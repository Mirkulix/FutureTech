import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'

// GET - List all datasets
export async function GET() {
  try {
    const datasets = await db.dataset.findMany({
      orderBy: { createdAt: 'desc' },
      include: {
        _count: {
          select: { trainings: true }
        }
      }
    })
    
    return NextResponse.json({ datasets })
  } catch (error) {
    console.error('Error fetching datasets:', error)
    return NextResponse.json(
      { error: 'Failed to fetch datasets' },
      { status: 500 }
    )
  }
}

// POST - Create a new dataset
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    const {
      name,
      type,
      description,
      filePath,
      fileSize,
      numSamples
    } = body
    
    if (!name || !type) {
      return NextResponse.json(
        { error: 'Missing required fields: name, type' },
        { status: 400 }
      )
    }
    
    // Validate dataset type
    const validTypes = ['wikitext-2', 'wikitext-103', 'openwebtext', 'custom']
    if (!validTypes.includes(type)) {
      return NextResponse.json(
        { error: `Invalid dataset type. Must be one of: ${validTypes.join(', ')}` },
        { status: 400 }
      )
    }
    
    // For predefined datasets, set default values
    let datasetData = {
      name,
      type,
      description: description || getDefaultDescription(type),
      filePath,
      fileSize,
      numSamples: numSamples || getDefaultNumSamples(type),
      status: type === 'custom' ? 'pending' : 'ready'
    }
    
    const dataset = await db.dataset.create({
      data: datasetData
    })
    
    return NextResponse.json({ dataset }, { status: 201 })
  } catch (error) {
    console.error('Error creating dataset:', error)
    return NextResponse.json(
      { error: 'Failed to create dataset' },
      { status: 500 }
    )
  }
}

// PUT - Update dataset
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json()
    const { id, ...updateData } = body
    
    if (!id) {
      return NextResponse.json(
        { error: 'Missing dataset id' },
        { status: 400 }
      )
    }
    
    const dataset = await db.dataset.update({
      where: { id },
      data: updateData
    })
    
    return NextResponse.json({ dataset })
  } catch (error) {
    console.error('Error updating dataset:', error)
    return NextResponse.json(
      { error: 'Failed to update dataset' },
      { status: 500 }
    )
  }
}

// DELETE - Delete dataset
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')
    
    if (!id) {
      return NextResponse.json(
        { error: 'Missing dataset id' },
        { status: 400 }
      )
    }
    
    // Check if dataset is used in any trainings
    const trainingsCount = await db.training.count({
      where: { datasetId: id }
    })
    
    if (trainingsCount > 0) {
      return NextResponse.json(
        { error: 'Cannot delete dataset that is used in trainings' },
        { status: 400 }
      )
    }
    
    await db.dataset.delete({
      where: { id }
    })
    
    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error deleting dataset:', error)
    return NextResponse.json(
      { error: 'Failed to delete dataset' },
      { status: 500 }
    )
  }
}

// Helper functions
function getDefaultDescription(type: string): string {
  const descriptions: Record<string, string> = {
    'wikitext-2': 'WikiText-2 language modeling dataset - 2M tokens for language modeling',
    'wikitext-103': 'WikiText-103 language modeling dataset - 103M tokens for language modeling',
    'openwebtext': 'OpenWebText - An open-source replication of the WebText dataset',
    'custom': 'User-uploaded custom dataset'
  }
  return descriptions[type] || 'Dataset for language model training'
}

function getDefaultNumSamples(type: string): number {
  const samples: Record<string, number> = {
    'wikitext-2': 36718,
    'wikitext-103': 1801350,
    'openwebtext': 8013769,
    'custom': 0
  }
  return samples[type] || 0
}
