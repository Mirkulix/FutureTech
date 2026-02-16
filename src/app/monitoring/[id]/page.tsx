"use client"

import { useParams } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Activity, Clock, Terminal, StopCircle, Pause, Play } from "lucide-react"

export default function MonitoringPage() {
    const params = useParams()
    const id = params?.id || "demo"

    return (
        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
            {/* Header Stats Bar */}
            <div className="grid gap-4 md:grid-cols-4">
                <Card className="bg-card/50 border-primary/20">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Epoch</CardTitle>
                        <Activity className="h-4 w-4 text-primary" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">2 <span className="text-sm text-muted-foreground">/ 10</span></div>
                        <Progress value={20} className="mt-2 h-1" />
                    </CardContent>
                </Card>
                <Card className="bg-card/50 border-primary/20">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">ETA</CardTitle>
                        <Clock className="h-4 w-4 text-primary" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">45m</div>
                        <p className="text-xs text-muted-foreground">Step 452/2000</p>
                    </CardContent>
                </Card>
                <Card className="bg-card/50 border-primary/20">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Speed</CardTitle>
                        <ZapIcon className="h-4 w-4 text-primary" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">4.5k</div>
                        <p className="text-xs text-muted-foreground">tokens/sec</p>
                    </CardContent>
                </Card>
                <div className="flex items-center gap-2">
                    <Button variant="outline" className="flex-1 h-full border-yellow-500/50 hover:bg-yellow-500/10 text-yellow-500">
                        <Pause className="mr-2 h-4 w-4" /> Pause
                    </Button>
                    <Button variant="destructive" className="flex-1 h-full shadow-[0_0_15px_rgba(239,68,68,0.4)]">
                        <StopCircle className="mr-2 h-4 w-4" /> Stop
                    </Button>
                </div>
            </div>

            <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-7 h-[600px]">
                {/* Left: Graphs */}
                <Card className="col-span-4 border-primary/10 bg-card/50">
                    <CardHeader>
                        <CardTitle>Live Metrics</CardTitle>
                        <CardDescription>Real-time training performance.</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="flex items-center justify-center h-[400px] border border-dashed border-muted rounded-md bg-black/20">
                            <p className="text-muted-foreground">Live Graph Placeholder (Connect Recharts here)</p>
                        </div>
                        <div className="mt-4 grid grid-cols-3 gap-4">
                            <div className="text-center">
                                <div className="text-sm text-muted-foreground">Current Loss</div>
                                <div className="text-xl font-bold text-primary">1.245</div>
                            </div>
                            <div className="text-center">
                                <div className="text-sm text-muted-foreground">Perplexity</div>
                                <div className="text-xl font-bold text-secondary">3.48</div>
                            </div>
                            <div className="text-center">
                                <div className="text-sm text-muted-foreground">Grad Norm</div>
                                <div className="text-xl font-bold">0.42</div>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Right: Terminal */}
                <Card className="col-span-3 bg-black border-primary/20 font-mono text-sm overflow-hidden flex flex-col">
                    <CardHeader className="bg-muted/10 py-3 border-b border-white/5">
                        <div className="flex items-center gap-2">
                            <Terminal className="h-4 w-4 text-green-500" />
                            <span className="text-green-500 font-bold">Terminal Output</span>
                            <Badge variant="outline" className="ml-auto border-green-900 text-green-500 text-[10px] h-5">Live</Badge>
                        </div>
                    </CardHeader>
                    <CardContent className="flex-1 p-0 overflow-hidden relative">
                        <div className="absolute inset-0 p-4 overflow-y-auto space-y-1 text-green-400/90 font-mono text-xs">
                            <p>[10:45:01] INFO: Starting epoch 2/10</p>
                            <p>[10:45:01] INFO: Loader initialized with 32 workers</p>
                            <p>[10:45:05] TRAIN: Batch 0 | Loss: 1.452 | Time: 0.4s</p>
                            <p>[10:45:05] TRAIN: Batch 1 | Loss: 1.448 | Time: 0.3s</p>
                            <p>[10:45:06] TRAIN: Batch 2 | Loss: 1.439 | Time: 0.3s</p>
                            <p>[10:45:06] TRAIN: Batch 3 | Loss: 1.435 | Time: 0.3s</p>
                            <p>[10:45:07] TRAIN: Batch 4 | Loss: 1.431 | Time: 0.3s</p>
                            <p>[10:45:07] TRAIN: Batch 5 | Loss: 1.428 | Time: 0.3s</p>
                            <p>[10:45:08] TRAIN: Batch 6 | Loss: 1.425 | Time: 0.3s</p>
                            <p>[10:45:08] TRAIN: Batch 7 | Loss: 1.420 | Time: 0.3s</p>
                            <p>[10:45:09] INFO: Gradient accumulation step</p>
                            <p className="animate-pulse">_</p>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}

function ZapIcon(props: any) {
    return (
        <svg
            {...props}
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2A.5.5 0 0 1 14 2.5V8h5.74a1 1 0 0 1 .93 1.36l-9.18 20.4a.5.5 0 0 1-.9.08l-2.61-6.06" />
        </svg>
    )
}
