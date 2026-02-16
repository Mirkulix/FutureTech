import { NextResponse } from 'next/server'
import { getLatestStatus, subscribe } from '@/lib/process-manager'

export const dynamic = 'force-dynamic'

export async function GET() {
  let unsubscribeFn: (() => void) | null = null
  let heartbeatId: ReturnType<typeof setInterval> | null = null

  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder()

      // Send initial snapshot
      const status = getLatestStatus()
      if (status) {
        controller.enqueue(encoder.encode(
          `data: ${JSON.stringify({ type: 'snapshot', data: status })}\n\n`
        ))
      }

      // Subscribe to live updates from process manager
      unsubscribeFn = subscribe((sseData: string) => {
        try {
          controller.enqueue(encoder.encode(sseData))
        } catch {
          unsubscribeFn?.()
        }
      })

      // Heartbeat every 15 seconds to keep connection alive
      heartbeatId = setInterval(() => {
        try {
          controller.enqueue(encoder.encode(': heartbeat\n\n'))
        } catch {
          if (heartbeatId) clearInterval(heartbeatId)
          unsubscribeFn?.()
        }
      }, 15000)
    },
    cancel() {
      if (heartbeatId) clearInterval(heartbeatId)
      unsubscribeFn?.()
    },
  })

  return new NextResponse(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  })
}
