'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip } from 'recharts';
import { Play, Square, RefreshCw, Activity, Brain, Clock, Zap, Gauge } from 'lucide-react';

interface TrainingData {
  id?: string;
  name?: string;
  status: string;
  currentEpoch?: number;
  currentStep?: number;
  totalSteps?: number;
  currentLoss?: number | null;
  bestLoss?: number | null;
  tokensPerSec?: number | null;
  perplexity?: number | null;
  eta?: string | null;
  startTime?: string;
  lossHistory?: string | number[];
}

export default function LiveTrainingPage() {
  const [training, setTraining] = useState<TrainingData | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<{step: number; loss: number; tokensPerSec: number}[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastStepRef = useRef<number>(0);

  // Fetch training status
  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/training');
      const data = await res.json();
      
      if (data.training) {
        const t = data.training;
        const isRunningNow = t.status === 'running';
        setIsRunning(isRunningNow);
        
        // Parse loss history
        let lossHistory: number[] = [];
        if (t.lossHistory) {
          if (typeof t.lossHistory === 'string') {
            try {
              lossHistory = JSON.parse(t.lossHistory);
            } catch {
              lossHistory = [];
            }
          } else if (Array.isArray(t.lossHistory)) {
            lossHistory = t.lossHistory;
          }
        }
        
        // Build metrics history
        const newMetrics: {step: number; loss: number; tokensPerSec: number}[] = [];
        lossHistory.forEach((loss, idx) => {
          if (typeof loss === 'number') {
            newMetrics.push({
              step: idx + 1,
              loss: loss,
              tokensPerSec: t.tokensPerSec || 0,
            });
          }
        });
        
        // Only add new points
        if (newMetrics.length > lastStepRef.current) {
          setMetricsHistory(newMetrics);
          lastStepRef.current = newMetrics.length;
        }
        
        setTraining(t);
        setError(null);
        
        // Add log if running
        if (isRunningNow && t.currentLoss) {
          const logMsg = `[${new Date().toLocaleTimeString()}] Step ${t.currentStep || 0}/${t.totalSteps || 0} | Loss: ${t.currentLoss.toFixed(4)} | ${t.tokensPerSec ? t.tokensPerSec.toFixed(0) + ' tok/s' : ''}`;
          setLogs(prev => [...prev.slice(-20), logMsg]);
        }
      }
    } catch (err) {
      console.error('Fetch error:', err);
      setError('Failed to fetch training status');
    }
  };

  // Start polling
  useEffect(() => {
    fetchStatus();
    intervalRef.current = setInterval(fetchStatus, 1500);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // Start training
  const startTraining = async () => {
    try {
      setError(null);
      setLogs(['Starting training...']);
      setMetricsHistory([]);
      lastStepRef.current = 0;
      
      // Get existing model config
      const modelRes = await fetch('/api/models');
      const modelData = await modelRes.json();
      const modelConfig = modelData.configs?.[0];
      
      // Get existing dataset
      const dsRes = await fetch('/api/datasets');
      const dsData = await dsRes.json();
      const dataset = dsData.datasets?.[0];
      
      // Create training record
      const trainRes = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: `Training-${Date.now()}`,
          modelConfigId: modelConfig?.id,
          datasetId: dataset?.id,
          epochs: 3,
          batchSize: 8,
          learningRate: 0.001,
          warmupSteps: 10,
          fisherOptimization: false,
          useSampleFiltering: true,
          useCurriculum: true,
          entropyThresholdLow: 1.0,
          entropyThresholdHigh: 5.0,
        }),
      });
      
      const trainData = await trainRes.json();
      if (!trainRes.ok) throw new Error(trainData.error || 'Failed to create training');
      
      const trainingId = trainData.training?.id;
      if (!trainingId) throw new Error('No training ID returned');
      
      // Start training
      const startRes = await fetch('/api/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'start',
          trainingId,
          config: {
            vocabSize: 32000,
            hiddenDim: 256,
            numLayers: 2,
            numHeads: 2,
            maxSeqLen: 512,
            intermediateDim: 512,
            epochs: 3,
            batchSize: 8,
            learningRate: 0.001,
            warmupSteps: 10,
            useFisher: false,
            useSampleFiltering: true,
            useCurriculum: true,
            entropyThresholdLow: 1.0,
            entropyThresholdHigh: 5.0,
            dataPath: 'wikitext-2',
          },
        }),
      });
      
      if (!startRes.ok) {
        const err = await startRes.json();
        throw new Error(err.error || 'Failed to start training');
      }
      
      setLogs(prev => [...prev, 'Training started!']);
      setIsRunning(true);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setError(msg);
      setLogs(prev => [...prev, `Error: ${msg}`]);
    }
  };

  // Stop training
  const stopTraining = async () => {
    try {
      await fetch('/api/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stop' }),
      });
      setLogs(prev => [...prev, 'Training stopped']);
      setIsRunning(false);
    } catch (err) {
      setError('Failed to stop training');
    }
  };

  const currentLoss = training?.currentLoss ?? 0;
  const currentStep = training?.currentStep ?? 0;
  const totalSteps = training?.totalSteps ?? 1;
  const progress = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600">
              <Brain className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                TernaryLLM Live Trainer
              </h1>
              <p className="text-slate-400">Real-time training visualization</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {isRunning ? (
              <Button 
                onClick={stopTraining}
                variant="destructive"
                className="gap-2"
              >
                <Square className="w-4 h-4" /> Stop Training
              </Button>
            ) : (
              <Button 
                onClick={startTraining}
                className="gap-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500"
              >
                <Play className="w-4 h-4" /> Start Training
              </Button>
            )}
            <Button 
              onClick={fetchStatus}
              variant="outline"
              size="icon"
              className="border-slate-700"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <Card className="border-red-500/50 bg-red-950/30">
            <CardContent className="pt-4 text-red-400">
              Error: {error}
            </CardContent>
          </Card>
        )}

        {/* Status Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-slate-400 flex items-center gap-2">
                <Activity className="w-4 h-4" /> Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-slate-500'}`} />
                <span className="text-lg font-bold">{isRunning ? 'Running' : training?.status || 'Idle'}</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-slate-400 flex items-center gap-2">
                <Gauge className="w-4 h-4" /> Loss
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-cyan-400">
                {typeof currentLoss === 'number' ? currentLoss.toFixed(4) : '--'}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-slate-400 flex items-center gap-2">
                <Zap className="w-4 h-4" /> Speed
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-400">
                {training?.tokensPerSec ? `${training.tokensPerSec.toFixed(0)} tok/s` : '--'}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-slate-400 flex items-center gap-2">
                <Clock className="w-4 h-4" /> ETA
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {training?.eta || '--:--'}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Progress */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="pt-4">
            <div className="flex justify-between text-sm text-slate-400 mb-2">
              <span>Epoch {training?.currentEpoch || 0}</span>
              <span>Step {currentStep} / {totalSteps}</span>
              <span>{progress.toFixed(1)}%</span>
            </div>
            <Progress value={progress} className="h-2 bg-slate-800" />
          </CardContent>
        </Card>

        {/* Charts */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Loss Chart */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-cyan-400" />
                Loss Over Time
                <Badge variant="outline" className="ml-auto">{metricsHistory.length} steps</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metricsHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis 
                    dataKey="step" 
                    stroke="#64748b" 
                    fontSize={12}
                  />
                  <YAxis 
                    stroke="#64748b" 
                    fontSize={12}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#0f172a', 
                      border: '1px solid #334155',
                      borderRadius: '8px'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#06b6d4" 
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Tokens/sec Chart */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-green-400" />
                Training Speed
                <Badge variant="outline" className="ml-auto">{metricsHistory.length} steps</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metricsHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis 
                    dataKey="step" 
                    stroke="#64748b" 
                    fontSize={12}
                  />
                  <YAxis 
                    stroke="#64748b" 
                    fontSize={12}
                    domain={[0, 'auto']}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#0f172a', 
                      border: '1px solid #334155',
                      borderRadius: '8px'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="tokensPerSec" 
                    stroke="#22c55e" 
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Logs */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader>
            <CardTitle className="text-sm">Training Logs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-32 overflow-y-auto font-mono text-xs space-y-1">
              {logs.map((log, i) => (
                <div key={i} className="text-slate-300">{log}</div>
              ))}
              {logs.length === 0 && <div className="text-slate-500">No logs yet...</div>}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
