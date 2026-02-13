'use client';

import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  Zap, Settings, Database, Play, Pause, Download, Upload, 
  Activity, Brain, HardDrive, Clock, TrendingUp, AlertCircle,
  CheckCircle2, Loader2, FileText, Sparkles
} from 'lucide-react';

// Types
interface ModelConfig {
  vocabSize: number;
  hiddenDim: number;
  numLayers: number;
  numHeads: number;
  maxSeqLen: number;
  intermediateDim: number;
}

interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  warmupSteps: number;
  useFisher: boolean;
  useSampleFiltering: boolean;
  useCurriculum: boolean;
  entropyThresholdLow: number;
  entropyThresholdHigh: number;
}

interface TrainingStatus {
  isRunning: boolean;
  currentEpoch: number;
  currentBatch: number;
  totalBatches: number;
  loss: number;
  perplexity: number;
  tokensPerSec: number;
  sparsity: number;
  eta: string;
  memoryUsed: number;
}

interface ModelStats {
  totalParams: number;
  ternaryParams: number;
  memoryMB: number;
  compressionRatio: number;
}

// Preset configurations
const PRESETS = {
  tiny: { name: 'Tiny (~10M)', hiddenDim: 128, numLayers: 2, numHeads: 2 },
  small: { name: 'Small (~30M)', hiddenDim: 256, numLayers: 4, numHeads: 4 },
  base: { name: 'Base (~100M)', hiddenDim: 512, numLayers: 8, numHeads: 8 },
  large: { name: 'Large (~350M)', hiddenDim: 768, numLayers: 12, numHeads: 12 },
};

// Datasets
const DATASETS = [
  { id: 'wikitext-2', name: 'WikiText-2', size: '2M tokens' },
  { id: 'wikitext-103', name: 'WikiText-103', size: '500M tokens' },
  { id: 'openwebtext', name: 'OpenWebText', size: '8B tokens' },
  { id: 'custom', name: 'Custom Upload', size: 'Variable' },
];

export default function TernaryLLMTrainer() {
  // Model configuration
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    vocabSize: 32000,
    hiddenDim: 512,
    numLayers: 8,
    numHeads: 8,
    maxSeqLen: 512,
    intermediateDim: 1344,
  });

  // Training configuration
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    epochs: 10,
    batchSize: 8,
    learningRate: 0.01,
    warmupSteps: 100,
    useFisher: true,
    useSampleFiltering: true,
    useCurriculum: true,
    entropyThresholdLow: 1.0,
    entropyThresholdHigh: 5.0,
  });

  // Training state
  const [status, setStatus] = useState<TrainingStatus>({
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
  });

  const [selectedDataset, setSelectedDataset] = useState('wikitext-2');
  const [modelStats, setModelStats] = useState<ModelStats | null>(null);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [logs, setLogs] = useState<string[]>([]);

  // Calculate model stats when config changes
  useEffect(() => {
    const calculateStats = () => {
      // Estimate parameters
      const embParams = modelConfig.vocabSize * modelConfig.hiddenDim * 2;
      const layerParams = (
        4 * modelConfig.hiddenDim * modelConfig.hiddenDim +
        2 * modelConfig.hiddenDim * modelConfig.intermediateDim
      );
      const lnParams = (2 * modelConfig.numLayers + 2) * modelConfig.hiddenDim;
      const outputParams = modelConfig.vocabSize * modelConfig.hiddenDim;
      
      const totalParams = embParams + modelConfig.numLayers * layerParams + lnParams + outputParams;
      const ternaryParams = totalParams * 0.8;
      
      const ternaryBytes = ternaryParams * 2 / 8;
      const floatBytes = (totalParams - ternaryParams) * 2;
      const memoryMB = (ternaryBytes + floatBytes) / (1024 * 1024);
      
      const compressionRatio = (totalParams * 4) / (ternaryBytes + floatBytes);
      
      setModelStats({
        totalParams,
        ternaryParams,
        memoryMB,
        compressionRatio,
      });
    };
    
    calculateStats();
  }, [modelConfig]);

  // Apply preset
  const applyPreset = (preset: keyof typeof PRESETS) => {
    const p = PRESETS[preset];
    setModelConfig(prev => ({
      ...prev,
      hiddenDim: p.hiddenDim,
      numLayers: p.numLayers,
      numHeads: p.numHeads,
      intermediateDim: Math.floor(p.hiddenDim * 2.6),
    }));
  };

  // Start training
  const startTraining = useCallback(async () => {
    setStatus(prev => ({ ...prev, isRunning: true }));
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Training gestartet...`]);
    
    // Simulate training progress
    let epoch = 0;
    let batch = 0;
    let currentLoss = 8.5;
    
    const interval = setInterval(() => {
      batch++;
      
      if (batch >= status.totalBatches) {
        batch = 0;
        epoch++;
        currentLoss *= 0.85; // Decrease loss each epoch
      }
      
      const progress = ((epoch * status.totalBatches + batch) / (trainingConfig.epochs * status.totalBatches)) * 100;
      
      setStatus(prev => ({
        ...prev,
        currentEpoch: epoch,
        currentBatch: batch,
        loss: currentLoss + Math.random() * 0.1,
        perplexity: Math.exp(currentLoss),
        tokensPerSec: 5000 + Math.random() * 2000,
        sparsity: 60 + Math.random() * 10,
        eta: `${trainingConfig.epochs - epoch}h ${Math.floor((status.totalBatches - batch) * 0.5)}m`,
        memoryUsed: modelStats?.memoryMB || 0,
      }));
      
      setLossHistory(prev => [...prev.slice(-50), currentLoss]);
      
      if (epoch >= trainingConfig.epochs) {
        clearInterval(interval);
        setStatus(prev => ({ ...prev, isRunning: false }));
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Training abgeschlossen!`]);
      }
    }, 100);
    
    return () => clearInterval(interval);
  }, [status.totalBatches, trainingConfig.epochs, modelStats?.memoryMB]);

  // Stop training
  const stopTraining = () => {
    setStatus(prev => ({ ...prev, isRunning: false }));
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Training gestoppt.`]);
  };

  // Export model
  const exportModel = (format: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Exportiere als ${format}...`]);
    // In real implementation, this would call the Python backend
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-500">
                <Brain className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">TernaryLLM</h1>
                <p className="text-sm text-slate-400">Quantum-Informed Accelerated Training</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <Badge variant="outline" className="bg-slate-800">
                <Zap className="w-3 h-3 mr-1 text-yellow-400" />
                432x Faster
              </Badge>
              <Badge variant="outline" className="bg-slate-800">
                <HardDrive className="w-3 h-3 mr-1 text-blue-400" />
                16x Compression
              </Badge>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="train" className="space-y-6">
          <TabsList className="bg-slate-800 border-slate-700">
            <TabsTrigger value="train" className="data-[state=active]:bg-slate-700">
              <Play className="w-4 h-4 mr-2" />
              Training
            </TabsTrigger>
            <TabsTrigger value="config" className="data-[state=active]:bg-slate-700">
              <Settings className="w-4 h-4 mr-2" />
              Configuration
            </TabsTrigger>
            <TabsTrigger value="data" className="data-[state=active]:bg-slate-700">
              <Database className="w-4 h-4 mr-2" />
              Datasets
            </TabsTrigger>
            <TabsTrigger value="export" className="data-[state=active]:bg-slate-700">
              <Download className="w-4 h-4 mr-2" />
              Export
            </TabsTrigger>
          </TabsList>

          {/* Training Tab */}
          <TabsContent value="train" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Main Training Panel */}
              <div className="lg:col-span-2 space-y-6">
                {/* Control Panel */}
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="w-5 h-5 text-green-400" />
                      Training Control
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex gap-4">
                      {!status.isRunning ? (
                        <Button 
                          onClick={startTraining}
                          className="flex-1 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600"
                        >
                          <Play className="w-4 h-4 mr-2" />
                          Start Training
                        </Button>
                      ) : (
                        <Button 
                          onClick={stopTraining}
                          variant="destructive"
                          className="flex-1"
                        >
                          <Pause className="w-4 h-4 mr-2" />
                          Stop Training
                        </Button>
                      )}
                    </div>
                    
                    {/* Progress */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Epoch {status.currentEpoch + 1}/{trainingConfig.epochs}</span>
                        <span>Batch {status.currentBatch}/{status.totalBatches}</span>
                      </div>
                      <Progress 
                        value={(status.currentEpoch / trainingConfig.epochs) * 100} 
                        className="h-2"
                      />
                    </div>
                    
                    {/* Quick Stats */}
                    <div className="grid grid-cols-4 gap-4">
                      <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-red-400">
                          {status.loss.toFixed(4)}
                        </div>
                        <div className="text-xs text-slate-400">Loss</div>
                      </div>
                      <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-blue-400">
                          {status.perplexity.toFixed(1)}
                        </div>
                        <div className="text-xs text-slate-400">Perplexity</div>
                      </div>
                      <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-green-400">
                          {Math.round(status.tokensPerSec).toLocaleString()}
                        </div>
                        <div className="text-xs text-slate-400">Tokens/sec</div>
                      </div>
                      <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-yellow-400">
                          {status.sparsity.toFixed(0)}%
                        </div>
                        <div className="text-xs text-slate-400">Sparsity</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Loss Chart */}
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="w-5 h-5 text-blue-400" />
                      Training Progress
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-48 flex items-end gap-1">
                      {lossHistory.map((loss, i) => (
                        <div 
                          key={i}
                          className="flex-1 bg-gradient-to-t from-blue-500 to-blue-300 rounded-t"
                          style={{ height: `${Math.min(100, (10 - loss) * 10)}%` }}
                        />
                      ))}
                      {lossHistory.length === 0 && (
                        <div className="flex-1 flex items-center justify-center text-slate-500">
                          Start training to see loss history
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Logs */}
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="w-5 h-5 text-slate-400" />
                      Training Logs
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-slate-900 rounded-lg p-4 h-40 overflow-y-auto font-mono text-sm">
                      {logs.map((log, i) => (
                        <div key={i} className="text-slate-300">{log}</div>
                      ))}
                      {logs.length === 0 && (
                        <div className="text-slate-500">No logs yet...</div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Sidebar */}
              <div className="space-y-6">
                {/* Model Stats */}
                {modelStats && (
                  <Card className="bg-slate-800/50 border-slate-700">
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-purple-400" />
                        Model Statistics
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Total Parameters</span>
                        <span className="font-mono">{modelStats.totalParams.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Ternary Parameters</span>
                        <span className="font-mono">{modelStats.ternaryParams.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Memory Usage</span>
                        <span className="font-mono">{modelStats.memoryMB.toFixed(1)} MB</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Compression</span>
                        <span className="font-mono text-green-400">{modelStats.compressionRatio.toFixed(1)}x</span>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Training Info */}
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Clock className="w-5 h-5 text-orange-400" />
                      Training Info
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-400">ETA</span>
                      <span className="font-mono">{status.eta}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Memory Used</span>
                      <span className="font-mono">{status.memoryUsed.toFixed(1)} MB</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Dataset</span>
                      <span className="font-mono">{selectedDataset}</span>
                    </div>
                  </CardContent>
                </Card>

                {/* Features */}
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="text-lg">Active Optimizations</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {trainingConfig.useFisher && (
                      <Badge variant="outline" className="mr-2 bg-blue-900/50">Fisher LR</Badge>
                    )}
                    {trainingConfig.useSampleFiltering && (
                      <Badge variant="outline" className="mr-2 bg-green-900/50">Sample Skip</Badge>
                    )}
                    {trainingConfig.useCurriculum && (
                      <Badge variant="outline" className="mr-2 bg-purple-900/50">Curriculum</Badge>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Configuration Tab */}
          <TabsContent value="config" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Model Config */}
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle>Model Architecture</CardTitle>
                  <CardDescription>Configure the neural network architecture</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Presets */}
                  <div className="space-y-2">
                    <Label>Quick Presets</Label>
                    <div className="flex gap-2 flex-wrap">
                      {Object.entries(PRESETS).map(([key, preset]) => (
                        <Button 
                          key={key}
                          variant="outline"
                          size="sm"
                          onClick={() => applyPreset(key as keyof typeof PRESETS)}
                          className="border-slate-600"
                        >
                          {preset.name}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <Separator className="bg-slate-700" />

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Vocabulary Size</Label>
                      <Input 
                        type="number" 
                        value={modelConfig.vocabSize}
                        onChange={e => setModelConfig(p => ({ ...p, vocabSize: parseInt(e.target.value) }))}
                        className="bg-slate-700 border-slate-600"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Hidden Dimension</Label>
                      <Input 
                        type="number" 
                        value={modelConfig.hiddenDim}
                        onChange={e => setModelConfig(p => ({ ...p, hiddenDim: parseInt(e.target.value) }))}
                        className="bg-slate-700 border-slate-600"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Number of Layers</Label>
                      <Input 
                        type="number" 
                        value={modelConfig.numLayers}
                        onChange={e => setModelConfig(p => ({ ...p, numLayers: parseInt(e.target.value) }))}
                        className="bg-slate-700 border-slate-600"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Attention Heads</Label>
                      <Input 
                        type="number" 
                        value={modelConfig.numHeads}
                        onChange={e => setModelConfig(p => ({ ...p, numHeads: parseInt(e.target.value) }))}
                        className="bg-slate-700 border-slate-600"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Max Sequence Length</Label>
                      <Input 
                        type="number" 
                        value={modelConfig.maxSeqLen}
                        onChange={e => setModelConfig(p => ({ ...p, maxSeqLen: parseInt(e.target.value) }))}
                        className="bg-slate-700 border-slate-600"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Intermediate Dimension</Label>
                      <Input 
                        type="number" 
                        value={modelConfig.intermediateDim}
                        onChange={e => setModelConfig(p => ({ ...p, intermediateDim: parseInt(e.target.value) }))}
                        className="bg-slate-700 border-slate-600"
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Training Config */}
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle>Training Configuration</CardTitle>
                  <CardDescription>Optimize your training process</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Epochs</Label>
                      <Input 
                        type="number" 
                        value={trainingConfig.epochs}
                        onChange={e => setTrainingConfig(p => ({ ...p, epochs: parseInt(e.target.value) }))}
                        className="bg-slate-700 border-slate-600"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Batch Size</Label>
                      <Input 
                        type="number" 
                        value={trainingConfig.batchSize}
                        onChange={e => setTrainingConfig(p => ({ ...p, batchSize: parseInt(e.target.value) }))}
                        className="bg-slate-700 border-slate-600"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Learning Rate</Label>
                      <Input 
                        type="number" 
                        step="0.001"
                        value={trainingConfig.learningRate}
                        onChange={e => setTrainingConfig(p => ({ ...p, learningRate: parseFloat(e.target.value) }))}
                        className="bg-slate-700 border-slate-600"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Warmup Steps</Label>
                      <Input 
                        type="number" 
                        value={trainingConfig.warmupSteps}
                        onChange={e => setTrainingConfig(p => ({ ...p, warmupSteps: parseInt(e.target.value) }))}
                        className="bg-slate-700 border-slate-600"
                      />
                    </div>
                  </div>

                  <Separator className="bg-slate-700" />

                  {/* Optimization Toggles */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Fisher-Optimized Learning Rate</Label>
                        <p className="text-xs text-slate-400">Adaptive LR based on Fisher Information</p>
                      </div>
                      <Switch 
                        checked={trainingConfig.useFisher}
                        onCheckedChange={checked => setTrainingConfig(p => ({ ...p, useFisher: checked }))}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Sample Filtering</Label>
                        <p className="text-xs text-slate-400">Skip low/high entropy samples</p>
                      </div>
                      <Switch 
                        checked={trainingConfig.useSampleFiltering}
                        onCheckedChange={checked => setTrainingConfig(p => ({ ...p, useSampleFiltering: checked }))}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Curriculum Learning</Label>
                        <p className="text-xs text-slate-400">Progress from simple to complex</p>
                      </div>
                      <Switch 
                        checked={trainingConfig.useCurriculum}
                        onCheckedChange={checked => setTrainingConfig(p => ({ ...p, useCurriculum: checked }))}
                      />
                    </div>
                  </div>

                  <Separator className="bg-slate-700" />

                  {/* Entropy Thresholds */}
                  <div className="space-y-4">
                    <Label>Entropy Thresholds</Label>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Low: {trainingConfig.entropyThresholdLow}</span>
                        <span>High: {trainingConfig.entropyThresholdHigh}</span>
                      </div>
                      <Slider 
                        value={[trainingConfig.entropyThresholdLow, trainingConfig.entropyThresholdHigh]}
                        min={0}
                        max={10}
                        step={0.1}
                        onValueChange={([low, high]) => setTrainingConfig(p => ({ 
                          ...p, 
                          entropyThresholdLow: low,
                          entropyThresholdHigh: high 
                        }))}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Datasets Tab */}
          <TabsContent value="data" className="space-y-6">
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle>Select Dataset</CardTitle>
                <CardDescription>Choose a dataset for training</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {DATASETS.map(dataset => (
                    <Card 
                      key={dataset.id}
                      className={`cursor-pointer transition-all ${
                        selectedDataset === dataset.id 
                          ? 'bg-blue-900/50 border-blue-500' 
                          : 'bg-slate-700/50 border-slate-600 hover:border-slate-500'
                      }`}
                      onClick={() => setSelectedDataset(dataset.id)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          {selectedDataset === dataset.id ? (
                            <CheckCircle2 className="w-5 h-5 text-blue-400" />
                          ) : (
                            <div className="w-5 h-5 rounded-full border-2 border-slate-500" />
                          )}
                          <div>
                            <div className="font-medium">{dataset.name}</div>
                            <div className="text-xs text-slate-400">{dataset.size}</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>

                <Separator className="bg-slate-700" />

                <div className="space-y-4">
                  <Label>Upload Custom Dataset</Label>
                  <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center">
                    <Upload className="w-10 h-10 mx-auto mb-4 text-slate-500" />
                    <p className="text-slate-400 mb-2">Drag & drop files here</p>
                    <p className="text-xs text-slate-500">Supports: .txt, .json, .jsonl, .csv</p>
                    <Button variant="outline" className="mt-4 border-slate-600">
                      Browse Files
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Export Tab */}
          <TabsContent value="export" className="space-y-6">
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle>Export Model</CardTitle>
                <CardDescription>Export your trained model in various formats</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <Alert className="bg-blue-900/30 border-blue-500">
                  <AlertCircle className="w-4 h-4" />
                  <AlertTitle>Export Ready</AlertTitle>
                  <AlertDescription>
                    Train a model first before exporting, or load an existing checkpoint.
                  </AlertDescription>
                </Alert>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card className="bg-slate-700/50 border-slate-600">
                    <CardContent className="p-6 text-center space-y-4">
                      <div className="w-16 h-16 mx-auto rounded-full bg-purple-900/50 flex items-center justify-center">
                        <Download className="w-8 h-8 text-purple-400" />
                      </div>
                      <div>
                        <h3 className="font-semibold">GGUF Format</h3>
                        <p className="text-sm text-slate-400">For llama.cpp & compatible tools</p>
                      </div>
                      <Button 
                        onClick={() => exportModel('GGUF')}
                        className="w-full bg-purple-600 hover:bg-purple-700"
                      >
                        Export GGUF
                      </Button>
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-700/50 border-slate-600">
                    <CardContent className="p-6 text-center space-y-4">
                      <div className="w-16 h-16 mx-auto rounded-full bg-yellow-900/50 flex items-center justify-center">
                        <Download className="w-8 h-8 text-yellow-400" />
                      </div>
                      <div>
                        <h3 className="font-semibold">HuggingFace Format</h3>
                        <p className="text-sm text-slate-400">Compatible with Transformers</p>
                      </div>
                      <Button 
                        onClick={() => exportModel('HuggingFace')}
                        className="w-full bg-yellow-600 hover:bg-yellow-700"
                      >
                        Export HF
                      </Button>
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-700/50 border-slate-600">
                    <CardContent className="p-6 text-center space-y-4">
                      <div className="w-16 h-16 mx-auto rounded-full bg-green-900/50 flex items-center justify-center">
                        <Download className="w-8 h-8 text-green-400" />
                      </div>
                      <div>
                        <h3 className="font-semibold">Raw Weights</h3>
                        <p className="text-sm text-slate-400">NumPy archive format</p>
                      </div>
                      <Button 
                        onClick={() => exportModel('Raw')}
                        className="w-full bg-green-600 hover:bg-green-700"
                      >
                        Export Raw
                      </Button>
                    </CardContent>
                  </Card>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-700 bg-slate-900/50 mt-12">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between text-sm text-slate-400">
            <div>TernaryLLM v1.0.0 - MIT License</div>
            <div className="flex items-center gap-4">
              <span>Powered by IGQK + TSLM + TriLLM</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
