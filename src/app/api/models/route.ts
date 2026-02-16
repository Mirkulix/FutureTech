import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import { db } from '@/lib/db'
import { findPython } from '@/lib/process-manager'
import { statSync } from 'fs'
import path from 'path'

// GET - List all model configs or exports
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const type = searchParams.get('type') // 'configs' or 'exports'
    
    if (type === 'exports') {
      const exports = await db.modelExport.findMany({
        include: {
          training: {
            include: {
              modelConfig: true,
              dataset: true
            }
          }
        },
        orderBy: { createdAt: 'desc' }
      })
      return NextResponse.json({ exports })
    } else {
      // Default: return model configs
      const configs = await db.modelConfig.findMany({
        include: {
          _count: {
            select: { trainings: true }
          }
        },
        orderBy: { createdAt: 'desc' }
      })
      return NextResponse.json({ configs })
    }
  } catch (error) {
    console.error('Error fetching models:', error)
    return NextResponse.json(
      { error: 'Failed to fetch models' },
      { status: 500 }
    )
  }
}

// POST - Create new model config or export
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { action, ...data } = body
    
    if (action === 'export') {
      // Create model export
      const { trainingId, name, format } = data
      
      if (!trainingId || !name || !format) {
        return NextResponse.json(
          { error: 'Missing required fields: trainingId, name, format' },
          { status: 400 }
        )
      }
      
      // Validate format
      const validFormats = ['gguf', 'huggingface', 'raw']
      if (!validFormats.includes(format)) {
        return NextResponse.json(
          { error: `Invalid format. Must be one of: ${validFormats.join(', ')}` },
          { status: 400 }
        )
      }
      
      // Check if training exists and is completed
      const training = await db.training.findUnique({
        where: { id: trainingId }
      })
      
      if (!training) {
        return NextResponse.json(
          { error: 'Training not found' },
          { status: 404 }
        )
      }
      
      if (training.status !== 'completed') {
        return NextResponse.json(
          { error: 'Training must be completed before exporting' },
          { status: 400 }
        )
      }
      
      // Determine model path from training output
      const modelDir = training.outputDir
        ? path.join(training.outputDir, 'final')
        : `./training_output/${trainingId}/final`
      const exportDir = `./exports/${trainingId}`

      // Create export record
      const modelExport = await db.modelExport.create({
        data: {
          trainingId,
          name,
          format,
          filePath: exportDir,
          fileSize: 0,
          status: 'processing'
        }
      })

      // Spawn Python bridge export in the background
      const pythonCmd = findPython()
      const args: string[] = []
      if (pythonCmd === 'py') args.push('-3')
      args.push(
        '-u', '-m', 'ternary_llm.bridge', 'export',
        '--model', modelDir,
        '--output', exportDir,
        '--format', format,
        '--name', name,
      )

      const child = spawn(pythonCmd, args, {
        cwd: process.cwd(),
        stdio: ['ignore', 'pipe', 'pipe'],
      })

      // Handle completion asynchronously
      let stdout = ''
      child.stdout!.on('data', (chunk: Buffer) => { stdout += chunk.toString() })

      child.on('close', async (code) => {
        try {
          if (code === 0) {
            // Try to determine real file size
            const ext = format === 'gguf' ? 'gguf' : format === 'huggingface' ? 'bin' : 'pt'
            const exportedFile = path.join(exportDir, `${name}.${ext}`)
            let fileSize = 0
            try {
              fileSize = statSync(exportedFile).size
            } catch {
              // File might have a different name; check stdout for path
              const lines = stdout.split('\n').filter(l => l.trim())
              for (const line of lines) {
                try {
                  const msg = JSON.parse(line)
                  if (msg.type === 'export_completed' && msg.data?.output) {
                    // Best effort size calculation from directory
                    break
                  }
                } catch { /* non-JSON */ }
              }
            }

            await db.modelExport.update({
              where: { id: modelExport.id },
              data: {
                status: 'completed',
                filePath: exportDir,
                fileSize,
              }
            })
          } else {
            await db.modelExport.update({
              where: { id: modelExport.id },
              data: { status: 'failed' }
            })
          }
        } catch { /* DB might be closed */ }
      })

      child.on('error', async () => {
        try {
          await db.modelExport.update({
            where: { id: modelExport.id },
            data: { status: 'failed' }
          })
        } catch { /* best effort */ }
      })

      return NextResponse.json({ export: modelExport }, { status: 201 })
    } else {
      // Create model config
      const {
        name,
        vocabSize = 32000,
        hiddenDimension = 512,
        layers = 8,
        attentionHeads = 8,
        intermediateSize,
        maxPositionEmbeds = 2048
      } = data
      
      if (!name) {
        return NextResponse.json(
          { error: 'Missing required field: name' },
          { status: 400 }
        )
      }
      
      // Calculate intermediate size if not provided
      const calculatedIntermediateSize = intermediateSize || hiddenDimension * 4
      
      const config = await db.modelConfig.create({
        data: {
          name,
          vocabSize,
          hiddenDimension,
          layers,
          attentionHeads,
          intermediateSize: calculatedIntermediateSize,
          maxPositionEmbeds
        }
      })
      
      return NextResponse.json({ config }, { status: 201 })
    }
  } catch (error) {
    console.error('Error creating model:', error)
    return NextResponse.json(
      { error: 'Failed to create model' },
      { status: 500 }
    )
  }
}

// PUT - Update model config
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json()
    const { id, ...updateData } = body
    
    if (!id) {
      return NextResponse.json(
        { error: 'Missing model config id' },
        { status: 400 }
      )
    }
    
    const config = await db.modelConfig.update({
      where: { id },
      data: updateData
    })
    
    return NextResponse.json({ config })
  } catch (error) {
    console.error('Error updating model config:', error)
    return NextResponse.json(
      { error: 'Failed to update model config' },
      { status: 500 }
    )
  }
}

// DELETE - Delete model config or export
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')
    const type = searchParams.get('type') // 'config' or 'export'
    
    if (!id || !type) {
      return NextResponse.json(
        { error: 'Missing id or type parameter' },
        { status: 400 }
      )
    }
    
    if (type === 'export') {
      await db.modelExport.delete({
        where: { id }
      })
    } else {
      // Check if config is used in any trainings
      const trainingsCount = await db.training.count({
        where: { modelConfigId: id }
      })
      
      if (trainingsCount > 0) {
        return NextResponse.json(
          { error: 'Cannot delete model config that is used in trainings' },
          { status: 400 }
        )
      }
      
      await db.modelConfig.delete({
        where: { id }
      })
    }
    
    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error deleting model:', error)
    return NextResponse.json(
      { error: 'Failed to delete model' },
      { status: 500 }
    )
  }
}
