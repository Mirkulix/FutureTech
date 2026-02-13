import { NextRequest, NextResponse } from 'next/server';

// Training status state (in production, use a proper database)
let trainingState = {
  isRunning: false,
  currentEpoch: 0,
  currentBatch: 0,
  totalBatches: 100,
  loss: 0,
  perplexity: 0,
  tokensPerSec: 0,
  sparsity: 0,
  eta: '--:--:--',
  memoryUsed: 0,
};

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    training: trainingState,
  });
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { action, config } = body;

  switch (action) {
    case 'start':
      trainingState.isRunning = true;
      // In production, spawn Python training process here
      return NextResponse.json({ 
        status: 'started', 
        message: 'Training started',
        config 
      });

    case 'stop':
      trainingState.isRunning = false;
      return NextResponse.json({ 
        status: 'stopped', 
        message: 'Training stopped' 
      });

    case 'update':
      // Update training state (called by Python backend)
      trainingState = { ...trainingState, ...body.state };
      return NextResponse.json({ status: 'updated' });

    case 'export':
      // Export model
      return NextResponse.json({ 
        status: 'exporting',
        format: body.format 
      });

    default:
      return NextResponse.json({ 
        status: 'error', 
        message: 'Unknown action' 
      }, { status: 400 });
  }
}
