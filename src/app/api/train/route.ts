import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { PythonBridge } from '@/lib/python-bridge'
import path from 'path'

// GET - List all trainings
export async function GET() {
  try {
    const trainings = await db.training.findMany({
      include: {
        modelConfig: true,
        dataset: true,
        exports: true
      },
      orderBy: { createdAt: 'desc' }
    })

    return NextResponse.json({ trainings })
  } catch (error) {
    console.error('Error fetching trainings:', error)
    return NextResponse.json(
      { error: 'Failed to fetch trainings' },
      { status: 500 }
    )
  }
}

// POST - Create a new training
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    // Branch 1: Full DB-backed training (used by Haupt-Dashboard)
    if (body.modelConfigId && body.datasetId) {
      const {
        name,
        modelConfigId,
        datasetId,
        epochs,
        batchSize,
        learningRate,
        warmupSteps,
        fisherOptimization,
        useSampleFiltering,
        useCurriculum,
        entropyThresholdLow,
        entropyThresholdHigh,
      } = body

      if (!name || !modelConfigId || !datasetId) {
        return NextResponse.json(
          { error: 'Missing required fields: name, modelConfigId, datasetId' },
          { status: 400 },
        )
      }

      const training = await db.training.create({
        data: {
          name,
          modelConfigId,
          datasetId,
          epochs,
          batchSize,
          learningRate,
          warmupSteps,
          fisherOptimization,
          useSampleFiltering,
          useCurriculum,
          entropyThresholdLow,
          entropyThresholdHigh,
        },
      })

      return NextResponse.json({ training }, { status: 201 })
    }

    // Branch 2: Lightweight Python-Bridge-Training (used von /training/new)
    const { name, model, dataset, epochs, batchSize, learningRate, useFisher } = body

    if (!name || !model || !dataset) {
      return NextResponse.json(
        { error: 'Missing required fields: name, model, dataset' },
        { status: 400 },
      )
    }

    const id = Date.now().toString()
    const outputDir = path.join(process.cwd(), 'output', id)
    const dataPath = path.join(process.cwd(), 'data', String(dataset))

    const pid = PythonBridge.startTraining({
      id,
      data: dataPath,
      output: outputDir,
      epochs: Number(epochs),
      batchSize: Number(batchSize),
      learningRate: Number(learningRate),
      useFisher: Boolean(useFisher),
      baseModel: model || 'base',
    })

    return NextResponse.json(
      {
        status: 'started',
        message: 'Training started successfully',
        trainingId: id,
        pid,
      },
      { status: 201 },
    )
  } catch (error) {
    console.error('Failed to start training:', error)
    return NextResponse.json(
      {
        status: 'error',
        message: 'Failed to start training process',
      },
      { status: 500 },
    )
  }
}

// PUT - Update training status
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json()
    const { id, action } = body

    if (!id || !action) {
      return NextResponse.json(
        { error: 'Missing required fields: id, action' },
        { status: 400 }
      )
    }

    const training = await db.training.findUnique({
      where: { id }
    })

    if (!training) {
      return NextResponse.json(
        { error: 'Training not found' },
        { status: 404 }
      )
    }

    let updateData: Record<string, unknown> = {}

    switch (action) {
      case 'start':
        updateData = {
          status: 'running',
          startTime: new Date()
        }
        break
      case 'pause':
        updateData = { status: 'paused' }
        break
      case 'resume':
        updateData = { status: 'running' }
        break
      case 'stop':
        updateData = {
          status: 'stopped',
          endTime: new Date()
        }
        break
      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        )
    }

    const updatedTraining = await db.training.update({
      where: { id },
      data: updateData,
      include: {
        modelConfig: true,
        dataset: true
      }
    })

    return NextResponse.json({ training: updatedTraining })
  } catch (error) {
    console.error('Error updating training:', error)
    return NextResponse.json(
      { error: 'Failed to update training' },
      { status: 500 }
    )
  }
}

// DELETE - Delete a training
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')

    if (!id) {
      return NextResponse.json(
        { error: 'Missing training id' },
        { status: 400 }
      )
    }

    // Delete associated exports first
    await db.modelExport.deleteMany({
      where: { trainingId: id }
    })

    // Delete training
    await db.training.delete({
      where: { id }
    })

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error deleting training:', error)
    return NextResponse.json(
      { error: 'Failed to delete training' },
      { status: 500 }
    )
  }
}
