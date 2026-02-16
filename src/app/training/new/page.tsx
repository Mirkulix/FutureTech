"use client"

import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import { z } from "zod"
import { Button } from "@/components/ui/button"
import {
    Form,
    FormControl,
    FormDescription,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
} from "@/components/ui/form"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Zap, Save } from "lucide-react"

// Relaxed schema to avoid strict type conflicts in this environment
const formSchema = z.object({
    name: z.string().min(2, {
        message: "Name must be at least 2 characters.",
    }),
    model: z.string({
        required_error: "Please select a base model.",
    }),
    dataset: z.string({
        required_error: "Please select a training dataset.",
    }),
    epochs: z.number().min(1).max(100),
    batchSize: z.number().min(1).max(512),
    learningRate: z.number().min(0.0001).max(0.1),
    useFisher: z.boolean(),
    overrides: z.string().optional(),
})

type FormValues = z.infer<typeof formSchema>

export default function NewTrainingPage() {
    const form = useForm<FormValues>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            name: "",
            epochs: 10,
            batchSize: 32,
            learningRate: 0.001,
            useFisher: true,
            overrides: "{\n  \"ternary_threshold\": 0.05\n}",
        },
    })

    async function onSubmit(values: FormValues) {
        console.log(values)
        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(values),
            })
            const data = await response.json()
            console.log('Training started:', data)
            // Redirect to monitoring page: /monitoring/[id]
            if (data.trainingId) {
                window.location.href = `/monitoring/${data.trainingId}`
            }
        } catch (error) {
            console.error('Failed to start training:', error)
        }
    }

    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <div className="flex items-center justify-between space-y-2">
                <h2 className="text-3xl font-bold tracking-tight text-primary">New Training</h2>
            </div>
            <Separator className="my-4" />
            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">

                        {/* Left Column: Configuration */}
                        <div className="col-span-4 space-y-4">
                            <Card className="border-primary/20 bg-card/50">
                                <CardHeader>
                                    <CardTitle>Core Configuration</CardTitle>
                                    <CardDescription>
                                        Select the base model and dataset for this training run.
                                    </CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    <FormField
                                        control={form.control}
                                        name="name"
                                        render={({ field }) => (
                                            <FormItem>
                                                <FormLabel>Training Name</FormLabel>
                                                <FormControl>
                                                    <Input placeholder="experiment-alpha-01" {...field} />
                                                </FormControl>
                                                <FormMessage />
                                            </FormItem>
                                        )}
                                    />
                                    <div className="grid grid-cols-2 gap-4">
                                        <FormField
                                            control={form.control}
                                            name="model"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Base Model</FormLabel>
                                                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                                                        <FormControl>
                                                            <SelectTrigger>
                                                                <SelectValue placeholder="Select a model" />
                                                            </SelectTrigger>
                                                        </FormControl>
                                                        <SelectContent>
                                                            <SelectItem value="tiny">Ternary-Tiny (30M)</SelectItem>
                                                            <SelectItem value="small">Ternary-Small (85M)</SelectItem>
                                                            <SelectItem value="base">Ternary-Base (110M)</SelectItem>
                                                            <SelectItem value="large">Ternary-Large (350M)</SelectItem>
                                                        </SelectContent>
                                                    </Select>
                                                    <FormMessage />
                                                </FormItem>
                                            )}
                                        />
                                        <FormField
                                            control={form.control}
                                            name="dataset"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Dataset</FormLabel>
                                                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                                                        <FormControl>
                                                            <SelectTrigger>
                                                                <SelectValue placeholder="Select dataset" />
                                                            </SelectTrigger>
                                                        </FormControl>
                                                        <SelectContent>
                                                            <SelectItem value="wikitext">WikiText-2 (Cleaned)</SelectItem>
                                                            <SelectItem value="c4">C4 (Subset)</SelectItem>
                                                            <SelectItem value="alpaca">Alpaca-Instruct</SelectItem>
                                                        </SelectContent>
                                                    </Select>
                                                    <FormMessage />
                                                </FormItem>
                                            )}
                                        />
                                    </div>
                                </CardContent>
                            </Card>

                            <Card className="border-secondary/20 bg-card/50">
                                <CardHeader>
                                    <CardTitle>Hyperparameters</CardTitle>
                                    <CardDescription>
                                        Fine-tune the training dynamics.
                                    </CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-6">
                                    <FormField
                                        control={form.control}
                                        name="learningRate"
                                        render={({ field }) => (
                                            <FormItem>
                                                <FormLabel>Learning Rate: {field.value}</FormLabel>
                                                <FormControl>
                                                    <Slider
                                                        min={0.0001}
                                                        max={0.01}
                                                        step={0.0001}
                                                        defaultValue={[field.value]}
                                                        onValueChange={(vals) => field.onChange(vals[0])}
                                                        className="py-4"
                                                    />
                                                </FormControl>
                                                <FormMessage />
                                            </FormItem>
                                        )}
                                    />
                                    <div className="grid grid-cols-2 gap-4">
                                        <FormField
                                            control={form.control}
                                            name="batchSize"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Batch Size: {field.value}</FormLabel>
                                                    <FormControl>
                                                        <Slider
                                                            min={1}
                                                            max={128}
                                                            step={1}
                                                            defaultValue={[field.value]}
                                                            onValueChange={(vals) => field.onChange(vals[0])}
                                                            className="py-4"
                                                        />
                                                    </FormControl>
                                                    <FormMessage />
                                                </FormItem>
                                            )}
                                        />
                                        <FormField
                                            control={form.control}
                                            name="epochs"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Epochs: {field.value}</FormLabel>
                                                    <FormControl>
                                                        <Slider
                                                            min={1}
                                                            max={100}
                                                            step={1}
                                                            defaultValue={[field.value]}
                                                            onValueChange={(vals) => field.onChange(vals[0])}
                                                            className="py-4"
                                                        />
                                                    </FormControl>
                                                    <FormMessage />
                                                </FormItem>
                                            )}
                                        />
                                    </div>
                                    <FormField
                                        control={form.control}
                                        name="useFisher"
                                        render={({ field }) => (
                                            <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3 shadow-sm">
                                                <div className="space-y-0.5">
                                                    <FormLabel>Fisher Information</FormLabel>
                                                    <FormDescription>
                                                        Enable quantum-informed gradients.
                                                    </FormDescription>
                                                </div>
                                                <FormControl>
                                                    <Switch
                                                        checked={field.value}
                                                        onCheckedChange={field.onChange}
                                                    />
                                                </FormControl>
                                            </FormItem>
                                        )}
                                    />
                                </CardContent>
                            </Card>
                        </div>

                        {/* Right Column: Advanced & Actions */}
                        <div className="col-span-3 space-y-4">
                            <Card className="bg-card/50">
                                <CardHeader>
                                    <CardTitle>Advanced Configuration</CardTitle>
                                    <CardDescription>
                                        JSON overrides for specific model parameters.
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <FormField
                                        control={form.control}
                                        name="overrides"
                                        render={({ field }) => (
                                            <FormItem>
                                                <FormControl>
                                                    <Textarea
                                                        placeholder="{}"
                                                        className="font-mono text-sm h-[200px] resize-none border-primary/20 bg-black/40"
                                                        {...field}
                                                    />
                                                </FormControl>
                                                <FormMessage />
                                            </FormItem>
                                        )}
                                    />
                                </CardContent>
                            </Card>

                            <div className="flex flex-col gap-4">
                                <Button type="submit" size="lg" className="w-full text-lg shadow-[0_0_20px_rgba(37,192,244,0.3)] hover:shadow-[0_0_30px_rgba(37,192,244,0.5)] transition-all">
                                    <Zap className="mr-2 h-5 w-5 fill-current" />
                                    Start Training
                                </Button>
                                <Button variant="outline" size="lg" className="w-full">
                                    <Save className="mr-2 h-4 w-4" />
                                    Save Configuration
                                </Button>
                            </div>
                        </div>
                    </div>
                </form>
            </Form>
        </div>
    )
}
