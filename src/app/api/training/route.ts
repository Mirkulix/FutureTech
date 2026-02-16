import { NextRequest, NextResponse } from 'next/server'
import { startTraining, stopTraining, getLatestStatus } from '@/lib/process-manager'
import { db } from '@/lib/db'

export async function GET() {
  // Return live status if training is running
  const liveStatus = getLatestStatus()
  if (liveStatus) {
    return NextResponse.json({ status: 'ok', training: liveStatus })
  }

  // Fall back to latest DB record
  const latest = await db.training.findFirst({
    orderBy: { updatedAt: 'desc' },
    include: { modelConfig: true, dataset: true },
  })
  return NextResponse.json({
    status: 'ok',
    training: latest ? { isRunning: false, ...latest } : null,
  })
}

export async function POST(request: NextRequest) {
  const body = await request.json()
  const { action } = body

  switch (action) {
    case 'start': {
      const { trainingId, config } = body

      if (!trainingId || !config) {
        return NextResponse.json(
          { error: 'Missing trainingId or config' },
          { status: 400 }
        )
      }

      // Verify training exists in DB
      const training = await db.training.findUnique({ where: { id: trainingId } })
      if (!training) {
        return NextResponse.json({ error: 'Training not found' }, { status: 404 })
      }

      try {
        await startTraining(trainingId, config)
        return NextResponse.json({ status: 'started', trainingId })
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error'
        return NextResponse.json({ error: message }, { status: 500 })
      }
    }

    case 'stop': {
      await stopTraining()
      return NextResponse.json({ status: 'stopped' })
    }

    default:
      return NextResponse.json({ error: 'Unknown action' }, { status: 400 })
  }
}
