import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'

// GET - Get training status
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')
    
    if (id) {
      // Get specific training status
      const training = await db.training.findUnique({
        where: { id },
        include: {
          modelConfig: true,
          dataset: true,
          exports: true
        }
      })
      
      if (!training) {
        return NextResponse.json(
          { error: 'Training not found' },
          { status: 404 }
        )
      }
      
      // Parse loss history
      const trainingWithHistory = {
        ...training,
        lossHistory: training.lossHistory ? JSON.parse(training.lossHistory) : []
      }
      
      return NextResponse.json({ training: trainingWithHistory })
    } else {
      // Get all active trainings
      const activeTrainings = await db.training.findMany({
        where: {
          status: { in: ['running', 'paused'] }
        },
        include: {
          modelConfig: true,
          dataset: true
        }
      })
      
      // Parse loss history for each training
      const trainingsWithHistory = activeTrainings.map(t => ({
        ...t,
        lossHistory: t.lossHistory ? JSON.parse(t.lossHistory) : []
      }))
      
      return NextResponse.json({ trainings: trainingsWithHistory })
    }
  } catch (error) {
    console.error('Error fetching training status:', error)
    return NextResponse.json(
      { error: 'Failed to fetch training status' },
      { status: 500 }
    )
  }
}
