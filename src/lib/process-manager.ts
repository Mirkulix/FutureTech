import { spawn, ChildProcess, execSync } from 'child_process'
import { db } from './db'

interface TrainingMetrics {
  epoch: number
  batch: number
  totalBatches: number
  globalStep: number
  totalSteps: number
  loss: number
  perplexity: number
  tokensPerSec: number
  sparsity: number
  learningRate: number
  samplesFiltered: number
  samplesTotal: number
}

interface ActiveTraining {
  trainingId: string
  process: ChildProcess
  latestMetrics: TrainingMetrics | null
  lossHistory: number[]
  logs: string[]
  subscribers: Set<(data: string) => void>
  startedAt: number
}

// Module-level singleton (survives HMR via globalThis, same pattern as db.ts)
const globalForProcess = globalThis as unknown as {
  activeTraining: ActiveTraining | undefined
  pythonCmd: string | undefined
  globalSubscribers: Set<(data: string) => void> | undefined
}

// Global subscribers that persist across training sessions
function getGlobalSubscribers(): Set<(data: string) => void> {
  if (!globalForProcess.globalSubscribers) {
    globalForProcess.globalSubscribers = new Set()
  }
  return globalForProcess.globalSubscribers
}

function getActive(): ActiveTraining | undefined {
  return globalForProcess.activeTraining
}

function setActive(t: ActiveTraining | undefined) {
  globalForProcess.activeTraining = t
}

/**
 * Auto-detect Python command on the system.
 * Tries: python, python3, py -3 (Windows launcher)
 */
export function findPython(): string {
  if (globalForProcess.pythonCmd) return globalForProcess.pythonCmd

  const candidates = ['python', 'python3', 'py']
  for (const cmd of candidates) {
    try {
      const args = cmd === 'py' ? ['-3', '--version'] : ['--version']
      const result = execSync(`${cmd} ${args.join(' ')}`, {
        timeout: 5000,
        stdio: ['ignore', 'pipe', 'pipe'],
      }).toString()
      if (result.includes('Python 3')) {
        globalForProcess.pythonCmd = cmd
        console.log(`[process-manager] Python found: ${cmd} -> ${result.trim()}`)
        return cmd
      }
    } catch {
      // try next candidate
    }
  }
  throw new Error('Python 3 not found. Install Python 3 and ensure it is on PATH.')
}

export function getLatestStatus() {
  const active = getActive()
  if (!active) return null
  return {
    isRunning: true,
    trainingId: active.trainingId,
    metrics: active.latestMetrics,
    lossHistory: active.lossHistory.slice(-50),
    logs: active.logs.slice(-50),
  }
}

export function subscribe(callback: (data: string) => void): () => void {
  const subs = getGlobalSubscribers()
  subs.add(callback)
  return () => { subs.delete(callback) }
}

function broadcast(_active: ActiveTraining, eventType: string, data: unknown) {
  const sseData = `data: ${JSON.stringify({ type: eventType, data })}\n\n`
  const subs = getGlobalSubscribers()
  for (const cb of subs) {
    try { cb(sseData) } catch { /* subscriber disconnected */ }
  }
}

export interface TrainingStartConfig {
  preset?: string
  vocabSize: number
  hiddenDim: number
  numLayers: number
  numHeads: number
  maxSeqLen: number
  intermediateDim: number
  epochs: number
  batchSize: number
  learningRate: number
  warmupSteps: number
  useFisher: boolean
  useSampleFiltering: boolean
  useCurriculum: boolean
  entropyThresholdLow: number
  entropyThresholdHigh: number
  dataPath: string
}

export async function startTraining(trainingId: string, config: TrainingStartConfig): Promise<void> {
  // Kill existing process if any
  await stopTraining()

  const pythonCmd = findPython()
  const outputDir = `./training_output/${trainingId}`

  const args: string[] = []

  // For 'py' launcher, add '-3' flag first
  if (pythonCmd === 'py') {
    args.push('-3')
  }

  args.push(
    '-u', // unbuffered stdout
    '-m', 'ternary_llm.bridge', 'train',
    '--data', config.dataPath,
    '--output', outputDir,
    '--vocab-size', String(config.vocabSize),
    '--hidden-dim', String(config.hiddenDim),
    '--num-layers', String(config.numLayers),
    '--num-heads', String(config.numHeads),
    '--max-seq-len', String(config.maxSeqLen),
    '--intermediate-dim', String(config.intermediateDim),
    '--epochs', String(config.epochs),
    '--batch-size', String(config.batchSize),
    '--learning-rate', String(config.learningRate),
    '--warmup-steps', String(config.warmupSteps),
    '--entropy-threshold-low', String(config.entropyThresholdLow ?? 1.0),
    '--entropy-threshold-high', String(config.entropyThresholdHigh ?? 5.0),
    '--log-every', '1',
  )

  if (config.preset) args.push('--preset', config.preset)
  if (config.useFisher) args.push('--use-fisher')
  if (config.useSampleFiltering) args.push('--use-sample-filtering')
  if (config.useCurriculum) args.push('--use-curriculum')

  console.log(`[process-manager] Spawning: ${pythonCmd} ${args.join(' ')}`)

  const child = spawn(pythonCmd, args, {
    cwd: process.cwd(),
    stdio: ['ignore', 'pipe', 'pipe'],
  })

  const active: ActiveTraining = {
    trainingId,
    process: child,
    latestMetrics: null,
    lossHistory: [],
    logs: [],
    subscribers: new Set(),
    startedAt: Date.now(),
  }
  setActive(active)

  // Update DB
  await db.training.update({
    where: { id: trainingId },
    data: {
      status: 'running',
      startTime: new Date(),
      pid: child.pid ?? null,
      outputDir,
    },
  })

  // Parse stdout line-by-line
  let buffer = ''
  child.stdout!.on('data', (chunk: Buffer) => {
    buffer += chunk.toString()
    const lines = buffer.split('\n')
    buffer = lines.pop() || '' // keep incomplete last line

    for (const line of lines) {
      if (!line.trim()) continue
      try {
        const msg = JSON.parse(line)
        handleMessage(active, msg)
      } catch {
        // Non-JSON output (Python print statements) -> forward as log
        active.logs.push(line)
        broadcast(active, 'log', { text: line })
      }
    }
  })

  // Stderr -> logs
  child.stderr!.on('data', (chunk: Buffer) => {
    const text = chunk.toString().trim()
    if (text) {
      active.logs.push(text)
      broadcast(active, 'log', { text })
    }
  })

  child.on('close', async (code) => {
    const wasActive = getActive()
    if (wasActive?.trainingId === trainingId) {
      setActive(undefined)
    }
    const finalStatus = code === 0 ? 'completed' : 'failed'
    try {
      await db.training.update({
        where: { id: trainingId },
        data: { status: finalStatus, endTime: new Date(), pid: null },
      })
    } catch { /* DB might be closed */ }
    broadcast(active, 'finished', { code, status: finalStatus })
    console.log(`[process-manager] Training ${trainingId} finished with code ${code}`)
  })

  child.on('error', (err) => {
    console.error(`[process-manager] Spawn error:`, err.message)
    active.logs.push(`Spawn error: ${err.message}`)
    broadcast(active, 'error', { message: err.message })
  })
}

async function handleMessage(active: ActiveTraining, msg: { type: string; data: Record<string, unknown> }) {
  const { type, data } = msg

  switch (type) {
    case 'metrics': {
      const metrics: TrainingMetrics = {
        epoch: data.epoch as number,
        batch: data.batch as number,
        totalBatches: data.total_batches as number,
        globalStep: data.global_step as number,
        totalSteps: data.total_steps as number,
        loss: data.loss as number,
        perplexity: data.perplexity as number,
        tokensPerSec: data.tokens_per_sec as number,
        sparsity: data.sparsity as number,
        learningRate: data.learning_rate as number,
        samplesFiltered: data.samples_filtered as number,
        samplesTotal: data.samples_total as number,
      }
      active.latestMetrics = metrics
      active.lossHistory.push(metrics.loss)

      // Calculate ETA
      const elapsed = (Date.now() - active.startedAt) / 1000
      const progress = metrics.globalStep / Math.max(metrics.totalSteps, 1)
      const etaSec = progress > 0 ? (elapsed / progress) * (1 - progress) : 0
      const etaStr = formatEta(etaSec)

      // Update DB
      try {
        await db.training.update({
          where: { id: active.trainingId },
          data: {
            currentEpoch: metrics.epoch,
            currentStep: metrics.globalStep,
            totalSteps: metrics.totalSteps,
            currentLoss: metrics.loss,
            bestLoss: Math.min(metrics.loss, active.lossHistory.reduce((a, b) => Math.min(a, b), Infinity)),
            tokensPerSec: metrics.tokensPerSec,
            perplexity: metrics.perplexity,
            sparsity: metrics.sparsity,
            eta: etaStr,
            lossHistory: JSON.stringify(active.lossHistory.slice(-200)),
          },
        })
      } catch { /* non-critical */ }

      broadcast(active, 'metrics', { ...metrics, eta: etaStr })
      break
    }

    case 'init':
    case 'data_loaded':
    case 'status': {
      const text = (data.message as string) || JSON.stringify(data)
      active.logs.push(text)
      broadcast(active, type, data)
      break
    }

    case 'completed':
      broadcast(active, 'completed', data)
      break

    case 'error':
      active.logs.push(`ERROR: ${data.message}`)
      broadcast(active, 'error', data)
      break
  }
}

export async function stopTraining(): Promise<void> {
  const active = getActive()
  if (!active) return

  // Kill the process
  try {
    active.process.kill('SIGTERM')
  } catch { /* already dead */ }

  // Windows fallback: taskkill
  if (process.platform === 'win32' && active.process.pid) {
    try {
      spawn('taskkill', ['/pid', String(active.process.pid), '/f', '/t'], { stdio: 'ignore' })
    } catch { /* best effort */ }
  }

  const trainingId = active.trainingId
  setActive(undefined)

  try {
    await db.training.update({
      where: { id: trainingId },
      data: { status: 'stopped', endTime: new Date(), pid: null },
    })
  } catch { /* DB might not be available */ }
}

function formatEta(seconds: number): string {
  if (seconds <= 0 || !isFinite(seconds)) return '--:--:--'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}
