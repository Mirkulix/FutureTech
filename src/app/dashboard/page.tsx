"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Activity, Cpu, Server, Play, Plus } from "lucide-react"
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from "recharts"

const data = [
    { name: "00:00", loss: 4.2 },
    { name: "04:00", loss: 3.8 },
    { name: "08:00", loss: 3.5 },
    { name: "12:00", loss: 2.1 },
    { name: "16:00", loss: 1.8 },
    { name: "20:00", loss: 1.4 },
    { name: "24:00", loss: 1.2 },
]

export default function DashboardPage() {
    return (
        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
            <div className="flex items-center justify-between space-y-2">
                <h2 className="text-3xl font-bold tracking-tight text-primary">System Overview</h2>
                <div className="flex items-center space-x-2">
                    <Button>
                        <Plus className="mr-2 h-4 w-4" />
                        New Training
                    </Button>
                </div>
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card className="border-primary/20 bg-card/50 shadow-[0_0_10px_rgba(37,192,244,0.1)]">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">
                            GPU Usage
                        </CardTitle>
                        <Cpu className="h-4 w-4 text-primary" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold text-primary">85%</div>
                        <p className="text-xs text-muted-foreground">
                            +20.1% from last hour
                        </p>
                    </CardContent>
                </Card>
                <Card className="border-secondary/20 bg-card/50 shadow-[0_0_10px_rgba(188,19,254,0.1)]">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">
                            Active Trainings
                        </CardTitle>
                        <Activity className="h-4 w-4 text-secondary" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold text-secondary">2</div>
                        <p className="text-xs text-muted-foreground">
                            Processing batch #4029
                        </p>
                    </CardContent>
                </Card>
                <Card className="bg-card/50">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Total Models</CardTitle>
                        <Server className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">12</div>
                        <p className="text-xs text-muted-foreground">
                            +3 new this week
                        </p>
                    </CardContent>
                </Card>
                <Card className="bg-card/50">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Dataset Size</CardTitle>
                        <Activity className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">2.4 TB</div>
                        <p className="text-xs text-muted-foreground">
                            Indexed 45M tokens
                        </p>
                    </CardContent>
                </Card>
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
                <Card className="col-span-4 border-primary/10 bg-card/50">
                    <CardHeader>
                        <CardTitle>Global Loss Trend</CardTitle>
                        <CardDescription className="text-primary/70">
                            Average loss across all active training runs.
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="pl-2">
                        <div className="h-[200px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={data}>
                                    <XAxis
                                        dataKey="name"
                                        stroke="#888888"
                                        fontSize={12}
                                        tickLine={false}
                                        axisLine={false}
                                    />
                                    <YAxis
                                        stroke="#888888"
                                        fontSize={12}
                                        tickLine={false}
                                        axisLine={false}
                                        tickFormatter={(value) => `${value}`}
                                    />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid #333' }}
                                        itemStyle={{ color: '#fff' }}
                                    />
                                    <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                                    <Line
                                        type="monotone"
                                        dataKey="loss"
                                        stroke="var(--primary)"
                                        strokeWidth={2}
                                        dot={false}
                                        activeDot={{ r: 6, fill: "var(--primary)" }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>
                <Card className="col-span-3 border-secondary/10 bg-card/50">
                    <CardHeader>
                        <CardTitle>Recent Activity</CardTitle>
                        <CardDescription>
                            Latest system events and training status.
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-8">
                            <div className="flex items-center">
                                <div className="space-y-1">
                                    <p className="text-sm font-medium leading-none text-primary">
                                        Training Started
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                        Model: Ternary-Base-v2 · 2 min ago
                                    </p>
                                </div>
                                <div className="ml-auto font-medium">
                                    <Badge variant="outline" className="border-primary text-primary">Running</Badge>
                                </div>
                            </div>
                            <div className="flex items-center">
                                <div className="space-y-1">
                                    <p className="text-sm font-medium leading-none">
                                        Checkpoint Saved
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                        Epoch 4/10 · 15 min ago
                                    </p>
                                </div>
                                <div className="ml-auto font-medium">
                                    <Badge variant="secondary">Saved</Badge>
                                </div>
                            </div>
                            <div className="flex items-center">
                                <div className="space-y-1">
                                    <p className="text-sm font-medium leading-none text-destructive">
                                        Training Failed
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                        OOM Error · 2 hours ago
                                    </p>
                                </div>
                                <div className="ml-auto font-medium">
                                    <Badge variant="destructive">Failed</Badge>
                                </div>
                            </div>

                            <div className="flex items-center">
                                <div className="space-y-1">
                                    <p className="text-sm font-medium leading-none">
                                        Dataset Uploaded
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                        wiki-en-v2.jsonl · 5 hours ago
                                    </p>
                                </div>
                                <div className="ml-auto font-medium">
                                    <Badge variant="outline">Done</Badge>
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
