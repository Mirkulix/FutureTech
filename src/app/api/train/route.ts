import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'

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
    
    const {
      name,
      modelConfigId,
      datasetId,
      learningRate = 0.0003,
      batchSize = 32,
      epochs = 3,
      warmupSteps = 100,
      weightDecay = 0.01,
      entropyThreshold = 2.5,
      fisherOptimization = false,
      gradientAccumSteps = 1
    } = body
    
    // Validate required fields
    if (!name || !modelConfigId || !datasetId) {
      return NextResponse.json(
        { error: 'Missing required fields: name, modelConfigId, datasetId' },
        { status: 400 }
      )
    }
    
    // Check if model config exists
    const modelConfig = await db.modelConfig.findUnique({
      where: { id: modelConfigId }
    })
    
    if (!modelConfig) {
      return NextResponse.json(
        { error: 'Model config not found' },
        { status: 404 }
      )
    }
    
    // Check if dataset exists
    const dataset = await db.dataset.findUnique({
      where: { id: datasetId }
    })
    
    if (!dataset) {
      return NextResponse.json(
        { error: 'Dataset not found' },
        { status: 404 }
      )
    }
    
    // Create training
    const training = await db.training.create({
      data: {
        name,
        modelConfigId,
        datasetId,
        learningRate,
        batchSize,
        epochs,
        warmupSteps,
        weightDecay,
        entropyThreshold,
        fisherOptimization,
        gradientAccumSteps,
        totalSteps: epochs * 1000, // Estimated total steps
        lossHistory: '[]'
      },
      include: {
        modelConfig: true,
        dataset: true
      }
    })
    
    return NextResponse.json({ training }, { status: 201 })
  } catch (error) {
    console.error('Error creating training:', error)
    return NextResponse.json(
      { error: 'Failed to create training' },
      { status: 500 }
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
          status: 'failed',
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
