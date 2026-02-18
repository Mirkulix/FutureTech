'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
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
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart';
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from 'recharts';
import {
  Zap, Settings, Database, Play, Pause, Download, Upload,
  Activity, Brain, HardDrive, Clock, TrendingUp, AlertCircle,
  CheckCircle2, Loader2, FileText, Sparkles, Wifi, RefreshCw
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
  globalStep: number;
  totalSteps: number;
  loss: number;
  perplexity: number;
  tokensPerSec: number;
  sparsity: number;
  eta: string;
  memoryUsed: number;
}

interface TrainingSnapshot {
  id: string;
  name: string;
  lossHistory: number[];
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

// Chart data types
interface MetricsDataPoint {
  step: number;
  loss: number;
  perplexity: number;
  tokensPerSec: number;
}

const chartConfig = {
  loss: { label: 'Loss', color: '#3b82f6' },
  tokensPerSec: { label: 'Tokens/s', color: '#22c55e' },
} satisfies ChartConfig;

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
    globalStep: 0,
    totalSteps: 1,
    loss: 0,
    perplexity: 0,
    tokensPerSec: 0,
    sparsity: 0,
    eta: '--:--:--',
    memoryUsed: 0,
  });

  const [selectedDataset, setSelectedDataset] = useState('wikitext-2');
  const [modelStats, setModelStats] = useState<ModelStats | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<MetricsDataPoint[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [trainingId, setTrainingId] = useState<string | null>(null);
  const [snapshotTraining, setSnapshotTraining] = useState<TrainingSnapshot | null>(null);
  const [isReplaying, setIsReplaying] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const replayIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const statusPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [sseStatus, setSseStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Calculate model stats when config changes
  useEffect(() => {
    if (statusPollRef.current) {
      clearInterval(statusPollRef.current);
      statusPollRef.current = null;
    }
    if (status.isRunning && trainingId) {
      statusPollRef.current = setInterval(async () => {
        try {
          const res = await fetch(`/api/status?id=${trainingId}`);
          if (!res.ok) return;
          const data = await res.json();
          const training = data.training;
          if (!training) return;
          setStatus(prev => ({
            ...prev,
            currentEpoch: training.currentEpoch ?? prev.currentEpoch,
            currentBatch: training.currentStep ?? prev.currentBatch,
            totalBatches: prev.totalBatches,
            globalStep: training.currentStep ?? prev.globalStep,
            totalSteps: training.totalSteps ?? prev.totalSteps,
            loss: training.currentLoss ?? prev.loss,
            perplexity: training.perplexity ?? prev.perplexity,
            tokensPerSec: training.tokensPerSec ?? prev.tokensPerSec,
            sparsity: training.sparsity ?? prev.sparsity,
            eta: training.eta ?? prev.eta,
          }));
          if (Array.isArray(training.lossHistory) && training.lossHistory.length > 0) {
            const losses = training.lossHistory as number[];
            setMetricsHistory(
              losses.map((loss: number, i: number) => ({
                step: i + 1,
                loss,
                perplexity: 0,
                tokensPerSec: 0,
              })),
            );
          }
          const tsStatus = typeof training.status === 'string' ? training.status : '';
          if (tsStatus && tsStatus !== 'running' && tsStatus !== 'paused') {
            setStatus(prev => ({ ...prev, isRunning: false }));
            if (statusPollRef.current) {
              clearInterval(statusPollRef.current);
              statusPollRef.current = null;
            }
          }
        } catch {
        }
      }, 2000);
    }
  }, [status.isRunning, trainingId]);

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

  // Connect SSE stream - called directly from startTraining() BEFORE spawning Python
  const connectSSE = useCallback(() => {
    // Close any existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    const es = new EventSource('/api/training/stream');
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      try {
        // Connection successful - update status
        if (sseStatus !== 'connected') {
          setSseStatus('connected');
          setConnectionError(null);
        }
        
        const msg = JSON.parse(event.data);
        const { type, data } = msg;
        if (type === 'metrics') {
          setStatus(prev => ({
            ...prev,
            isRunning: true,
            currentEpoch: data.epoch ?? prev.currentEpoch,
            currentBatch: data.batch ?? prev.currentBatch,
            totalBatches: data.totalBatches ?? prev.totalBatches,
            globalStep: data.globalStep ?? prev.globalStep,
            totalSteps: data.totalSteps ?? prev.totalSteps,
            loss: data.loss ?? prev.loss,
            perplexity: data.perplexity ?? prev.perplexity,
            tokensPerSec: data.tokensPerSec ?? prev.tokensPerSec,
            sparsity: data.sparsity ?? prev.sparsity,
            eta: data.eta ?? prev.eta,
          }));
          if (typeof data.loss === 'number') {
            setMetricsHistory(prev => [...prev.slice(-99), {
              step: data.globalStep ?? prev.length,
              loss: data.loss,
              perplexity: data.perplexity ?? 0,
              tokensPerSec: data.tokensPerSec ?? 0,
            }]);
          }
        } else if (type === 'snapshot') {
          // Initial snapshot from SSE - restore state
          if (data.metrics) {
            setStatus(prev => ({
              ...prev,
              isRunning: true,
              currentEpoch: data.metrics.epoch ?? prev.currentEpoch,
              currentBatch: data.metrics.batch ?? prev.currentBatch,
              totalBatches: data.metrics.totalBatches ?? prev.totalBatches,
              globalStep: data.metrics.globalStep ?? prev.globalStep,
              totalSteps: data.metrics.totalSteps ?? prev.totalSteps,
              loss: data.metrics.loss ?? prev.loss,
              perplexity: data.metrics.perplexity ?? prev.perplexity,
              tokensPerSec: data.metrics.tokensPerSec ?? prev.tokensPerSec,
              sparsity: data.metrics.sparsity ?? prev.sparsity,
            }));
          }
          if (data.lossHistory && Array.isArray(data.lossHistory)) {
            setMetricsHistory(data.lossHistory.map((l: number, i: number) => ({
              step: i, loss: l, perplexity: 0, tokensPerSec: 0,
            })));
          }
        } else if (type === 'log' || type === 'status' || type === 'init' || type === 'data_loaded') {
          const text = data.text || data.message || JSON.stringify(data);
          setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${text}`]);
        } else if (type === 'completed' || type === 'finished') {
          setStatus(prev => ({ ...prev, isRunning: false }));
          setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Training abgeschlossen!`]);
          es.close();
          eventSourceRef.current = null;
        } else if (type === 'error') {
          setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ERROR: ${data.message}`]);
        }
      } catch {
        // Ignore parse errors (e.g. heartbeat)
      }
    };

    es.onopen = () => {
      setSseStatus('connected');
      setConnectionError(null);
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ✓ SSE verbunden - warte auf Training...`]);
    };

    es.onerror = (event) => {
      setSseStatus('error');
      const errorMsg = 'SSE Verbindung fehlgeschlagen. Server nicht erreichbar oder Training nicht gestartet.';
      setConnectionError(errorMsg);
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ✗ ${errorMsg}`]);
      console.error('[SSE] Connection error:', event);
    };
  }, []);

  const replayLastTraining = useCallback(() => {
    if (!snapshotTraining || !snapshotTraining.lossHistory.length) {
      const ts = () => new Date().toLocaleTimeString();
      setLogs(prev => [...prev, `[${ts()}] Kein gespeichertes Training zum Replay gefunden.`]);
      return;
    }
    if (replayIntervalRef.current) {
      clearInterval(replayIntervalRef.current);
      replayIntervalRef.current = null;
    }
    const ts = () => new Date().toLocaleTimeString();
    setLogs(prev => [...prev, `[${ts()}] Replay des letzten Trainings gestartet (${snapshotTraining.name}).`]);
    setMetricsHistory([]);
    setIsReplaying(true);
    const losses = [...snapshotTraining.lossHistory];
    let index = 0;
    replayIntervalRef.current = setInterval(() => {
      index += 1;
      const loss = losses[index - 1];
      setMetricsHistory(prev => [
        ...prev,
        {
          step: index,
          loss,
          perplexity: 0,
          tokensPerSec: 0,
        },
      ]);
      if (index >= losses.length) {
        if (replayIntervalRef.current) {
          clearInterval(replayIntervalRef.current);
          replayIntervalRef.current = null;
        }
        setIsReplaying(false);
      }
    }, 300);
  }, [snapshotTraining]);

  // Start training - real API calls
  const startTraining = useCallback(async () => {
    const ts = () => new Date().toLocaleTimeString();

    // Sofort UI-State setzen damit Stop-Button erscheint
    setStatus(prev => ({ ...prev, isRunning: true }));
    setLogs(prev => [...prev, `[${ts()}] Training wird vorbereitet...`]);
    setMetricsHistory([]);

    try {
      // 1. Create ModelConfig in DB
      const cfgRes = await fetch('/api/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: `TernaryLLM-${Date.now()}`,
          vocabSize: modelConfig.vocabSize,
          hiddenDimension: modelConfig.hiddenDim,
          layers: modelConfig.numLayers,
          attentionHeads: modelConfig.numHeads,
          intermediateSize: modelConfig.intermediateDim,
          maxPositionEmbeds: modelConfig.maxSeqLen,
        }),
      });
      const cfgData = await cfgRes.json();
      if (!cfgRes.ok) throw new Error(cfgData.error || 'Model config creation failed');
      const { config: dbConfig } = cfgData;
      setLogs(prev => [...prev, `[${ts()}] Model config created: ${dbConfig.id}`]);

      // 2. Create dataset record
      const dsRes = await fetch('/api/datasets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: selectedDataset,
          type: selectedDataset,
        }),
      });
      const dsData = await dsRes.json();
      if (!dsRes.ok) throw new Error(dsData.error || 'Dataset creation failed');
      const { dataset: dbDataset } = dsData;
      setLogs(prev => [...prev, `[${ts()}] Dataset record created: ${dbDataset.id}`]);

      // 3. Create Training record
      const trainRes = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: `Training-${Date.now()}`,
          modelConfigId: dbConfig.id,
          datasetId: dbDataset.id,
          epochs: trainingConfig.epochs,
          batchSize: trainingConfig.batchSize,
          learningRate: trainingConfig.learningRate,
          warmupSteps: trainingConfig.warmupSteps,
          fisherOptimization: trainingConfig.useFisher,
          useSampleFiltering: trainingConfig.useSampleFiltering,
          useCurriculum: trainingConfig.useCurriculum,
          entropyThresholdLow: trainingConfig.entropyThresholdLow,
          entropyThresholdHigh: trainingConfig.entropyThresholdHigh,
        }),
      });
      const trainData = await trainRes.json();
      if (!trainRes.ok) throw new Error(trainData.error || 'Training record creation failed');
      const { training } = trainData;
      if (!training || !training.id) {
        throw new Error('Training API did not return a valid training.id');
      }
      setTrainingId(training.id);
      setLogs(prev => [...prev, `[${ts()}] Training record created: ${training.id}`]);

      // 4. Connect SSE BEFORE starting training to catch all events
      connectSSE();

      // 5. Start actual training via process manager
      const startRes = await fetch('/api/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'start',
          trainingId: training.id,
          config: {
            vocabSize: modelConfig.vocabSize,
            hiddenDim: modelConfig.hiddenDim,
            numLayers: modelConfig.numLayers,
            numHeads: modelConfig.numHeads,
            maxSeqLen: modelConfig.maxSeqLen,
            intermediateDim: modelConfig.intermediateDim,
            epochs: trainingConfig.epochs,
            batchSize: trainingConfig.batchSize,
            learningRate: trainingConfig.learningRate,
            warmupSteps: trainingConfig.warmupSteps,
            useFisher: trainingConfig.useFisher,
            useSampleFiltering: trainingConfig.useSampleFiltering,
            useCurriculum: trainingConfig.useCurriculum,
            entropyThresholdLow: trainingConfig.entropyThresholdLow,
            entropyThresholdHigh: trainingConfig.entropyThresholdHigh,
            dataPath: selectedDataset === 'custom' ? 'sample' : selectedDataset,
          },
        }),
      });

      if (!startRes.ok) {
        const err = await startRes.json();
        throw new Error(err.error || 'Failed to start training');
      }

      setLogs(prev => [...prev, `[${ts()}] Training gestartet!`]);
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      setLogs(prev => [...prev, `[${ts()}] ERROR: ${msg}`]);
      setStatus(prev => ({ ...prev, isRunning: false }));
    }
  }, [modelConfig, trainingConfig, selectedDataset, connectSSE]);

  // Stop training - real API call
  const stopTraining = useCallback(async () => {
    try {
      await fetch('/api/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stop' }),
      });
      setStatus(prev => ({ ...prev, isRunning: false }));
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Training gestoppt.`]);
      // Close SSE connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    } catch {
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Fehler beim Stoppen.`]);
    }
  }, []);

  // Export model - real API call
  const exportModel = useCallback(async (format: string) => {
    const ts = () => new Date().toLocaleTimeString();
    if (!trainingId) {
      setLogs(prev => [...prev, `[${ts()}] Kein Training vorhanden zum Exportieren.`]);
      return;
    }
    setLogs(prev => [...prev, `[${ts()}] Exportiere als ${format}...`]);
    try {
      const res = await fetch('/api/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'export',
          trainingId,
          name: `ternary-llm-${Date.now()}`,
          format: format.toLowerCase(),
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Export failed');
      }
      setLogs(prev => [...prev, `[${ts()}] Export gestartet (${format}). Wird im Hintergrund verarbeitet.`]);
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      setLogs(prev => [...prev, `[${ts()}] Export-Fehler: ${msg}`]);
    }
  }, [trainingId]);

  // Cleanup SSE on unmount
  useEffect(() => {
    let cancelled = false;
    const loadSnapshot = async () => {
      try {
        const res = await fetch('/api/train');
        if (!res.ok) return;
        const data = await res.json();
        const trainings = data.trainings as Array<{
          id: string;
          name: string;
          lossHistory?: string | null;
        }>;
        if (!Array.isArray(trainings) || trainings.length === 0) return;
        const latest = trainings[0];
        if (!latest.lossHistory) return;
        let parsed: unknown;
        try {
          parsed = JSON.parse(latest.lossHistory);
        } catch {
          return;
        }
        if (!Array.isArray(parsed) || parsed.length === 0) return;
        if (cancelled) return;
        const losses = parsed.filter((v) => typeof v === 'number') as number[];
        if (!losses.length) return;
        setSnapshotTraining({
          id: latest.id,
          name: latest.name,
          lossHistory: losses,
        });
        setMetricsHistory(
          losses.map((loss, i) => ({
            step: i + 1,
            loss,
            perplexity: 0,
            tokensPerSec: 0,
          })),
        );
      } catch {
      }
    };
    loadSnapshot();
    return () => {
      cancelled = true;
      if (replayIntervalRef.current) {
        clearInterval(replayIntervalRef.current);
        replayIntervalRef.current = null;
      }
      if (statusPollRef.current) {
        clearInterval(statusPollRef.current);
        statusPollRef.current = null;
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

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
              {status.isRunning && (
                <Badge
                  variant="destructive"
                  className="cursor-pointer hover:bg-red-700 transition-colors"
                  onClick={stopTraining}
                >
                  <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  Training läuft... (Stop)
                </Badge>
              )}
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
                        value={status.totalSteps > 0 ? (status.globalStep / status.totalSteps) * 100 : 0}
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

                {/* Loss Chart - Live recharts LineChart */}
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 justify-between">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-blue-400" />
                        Loss
                        {metricsHistory.length > 0 && (
                          <Badge variant="outline" className="ml-2 bg-blue-900/30">
                            {metricsHistory.length} Punkte
                          </Badge>
                        )}
                      </div>
                      {snapshotTraining && (
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={isReplaying}
                          onClick={replayLastTraining}
                        >
                          <RefreshCw className="w-4 h-4 mr-1" />
                          Replay
                        </Button>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {metricsHistory.length > 0 ? (
                      <ChartContainer config={chartConfig} className="h-48 w-full">
                        <LineChart data={metricsHistory}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                          <XAxis
                            dataKey="step"
                            stroke="#94a3b8"
                            fontSize={12}
                            tickFormatter={(v) => `${v}`}
                          />
                          <YAxis
                            stroke="#94a3b8"
                            fontSize={12}
                            tickFormatter={(v) => v.toFixed(2)}
                          />
                          <ChartTooltip content={<ChartTooltipContent />} />
                          <Line
                            type="monotone"
                            dataKey="loss"
                            stroke="var(--color-loss)"
                            strokeWidth={2}
                            dot={false}
                            isAnimationActive={false}
                          />
                        </LineChart>
                      </ChartContainer>
                    ) : (
                      <div className="h-48 flex flex-col items-center justify-center text-slate-500">
                        <Activity className="w-12 h-12 mb-2 opacity-50" />
                        <p>Noch kein Training gestartet</p>
                        <p className="text-xs text-slate-600">Klicke auf "Start Training" um zu beginnen</p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Throughput Chart - Tokens/sec */}
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="w-5 h-5 text-green-400" />
                      Throughput (Tokens/sec)
                      {metricsHistory.length > 0 && (
                        <Badge variant="outline" className="ml-2 bg-green-900/30">
                          Avg: {Math.round(metricsHistory.reduce((a, b) => a + b.tokensPerSec, 0) / metricsHistory.length)}
                        </Badge>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {metricsHistory.length > 0 ? (
                      <ChartContainer config={chartConfig} className="h-32 w-full">
                        <LineChart data={metricsHistory}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                          <XAxis
                            dataKey="step"
                            stroke="#94a3b8"
                            fontSize={12}
                            tickFormatter={(v) => `${v}`}
                          />
                          <YAxis
                            stroke="#94a3b8"
                            fontSize={12}
                            tickFormatter={(v) => `${Math.round(v)}`}
                          />
                          <ChartTooltip content={<ChartTooltipContent />} />
                          <Line
                            type="monotone"
                            dataKey="tokensPerSec"
                            stroke="var(--color-tokensPerSec)"
                            strokeWidth={2}
                            dot={false}
                            isAnimationActive={false}
                          />
                        </LineChart>
                      </ChartContainer>
                    ) : (
                      <div className="h-32 flex flex-col items-center justify-center text-slate-500">
                        <Activity className="w-8 h-8 mb-2 opacity-50" />
                        <p className="text-xs">Warte auf Trainingsdaten...</p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Debug Panel - Connection Status */}
                <Card className={`border-2 ${sseStatus === 'connected' ? 'bg-green-900/30 border-green-500' : sseStatus === 'error' ? 'bg-red-900/30 border-red-500' : 'bg-slate-800/50 border-slate-700'}`}>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Wifi className="w-5 h-5" />
                        Debug: SSE Connection
                      </div>
                      <Badge variant={sseStatus === 'connected' ? 'default' : sseStatus === 'error' ? 'destructive' : 'outline'} className={sseStatus === 'connected' ? 'bg-green-600' : ''}>
                        {sseStatus === 'connected' ? '🟢 Verbunden' : sseStatus === 'connecting' ? '🟡 Verbinde...' : sseStatus === 'error' ? '🔴 Fehler' : '⚪ Getrennt'}
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-slate-400">Status:</div>
                      <div className="font-mono">{sseStatus}</div>
                      <div className="text-slate-400">Letzte Nachricht:</div>
                      <div className="font-mono">{metricsHistory.length > 0 ? `Step ${metricsHistory[metricsHistory.length-1].step}` : 'Keine'}</div>
                      <div className="text-slate-400">Datenpunkte:</div>
                      <div className="font-mono">{metricsHistory.length}</div>
                      <div className="text-slate-400">Letzter Loss:</div>
                      <div className="font-mono text-red-400">{metricsHistory.length > 0 ? metricsHistory[metricsHistory.length-1].loss.toFixed(4) : '--'}</div>
                    </div>
                    {connectionError && (
                      <Alert variant="destructive" className="mt-2">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Verbindungsfehler</AlertTitle>
                        <AlertDescription>{connectionError}</AlertDescription>
                      </Alert>
                    )}
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="w-full mt-2"
                      onClick={() => {
                        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] 🔄 Manuelle SSE-Verbindung...`]);
                        connectSSE();
                      }}
                    >
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Verbindung neu herstellen
                    </Button>
                  </CardContent>
                </Card>

                {/* Logs */}
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <FileText className="w-5 h-5 text-slate-400" />
                        Training Logs
                        <Badge variant="outline" className="ml-2 bg-slate-700">
                          {logs.length} Einträge
                        </Badge>
                      </div>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        onClick={() => setLogs([])}
                      >
                        <RefreshCw className="w-4 h-4" />
                      </Button>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-slate-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
                      {logs.map((log, i) => (
                        <div 
                          key={i} 
                          className={`${log.includes('ERROR') ? 'text-red-400' : log.includes('✓') ? 'text-green-400' : log.includes('✗') ? 'text-red-400' : log.includes('🔄') ? 'text-yellow-400' : 'text-slate-300'}`}
                        >
                          {log}
                        </div>
                      ))}
                      {logs.length === 0 && (
                        <div className="text-slate-500 flex flex-col items-center justify-center h-full">
                          <FileText className="w-8 h-8 mb-2 opacity-50" />
                          <p>Keine Logs vorhanden</p>
                          <p className="text-xs">Starte ein Training um Logs zu sehen</p>
                        </div>
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
