import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

interface TrainingConfig {
    id: string;
    data: string;
    output: string;
    epochs: number;
    batchSize: number;
    learningRate: number;
    useFisher: boolean;
    baseModel: string;
}

export class PythonBridge {
    private static pythonPath = 'python'; // Assume python is in PATH

    /**
     * Spawns a new training process
     */
    static startTraining(config: TrainingConfig) {
        const logDir = path.join(process.cwd(), 'logs');
        if (!fs.existsSync(logDir)) {
            fs.mkdirSync(logDir);
        }

        const logFile = path.join(logDir, `training-${config.id}.log`);
        const logStream = fs.createWriteStream(logFile, { flags: 'a' });

        const args = [
            '-m', 'ternary_llm', 'train',
            '--data', config.data,
            '--output', config.output,
            '--epochs', config.epochs.toString(),
            '--batch-size', config.batchSize.toString(),
            '--learning-rate', config.learningRate.toString(),
            '--preset', config.baseModel
        ];

        if (config.useFisher) {
            args.push('--use-fisher');
        }

        console.log(`Spawning: ${this.pythonPath} ${args.join(' ')}`);
        logStream.write(`[START] Command: ${this.pythonPath} ${args.join(' ')}\n`);

        const child = spawn(this.pythonPath, args, {
            detached: true,
            stdio: ['ignore', 'pipe', 'pipe']
        });

        child.stdout.pipe(logStream);
        child.stderr.pipe(logStream);

        child.on('close', (code) => {
            console.log(`Training process ${config.id} exited with code ${code}`);
            logStream.write(`[EXIT] Code: ${code}\n`);
            logStream.end();
        });

        child.unref(); // Allow parent to exit independently

        return child.pid;
    }

    /**
     * Reads logs for a specific training session
     */
    static async getLogs(trainingId: string, lines: number = 50): Promise<string[]> {
        const logFile = path.join(process.cwd(), 'logs', `training-${trainingId}.log`);

        if (!fs.existsSync(logFile)) {
            return [`Log file not found for session ${trainingId}`];
        }

        try {
            const content = await fs.promises.readFile(logFile, 'utf-8');
            const allLines = content.split('\n');
            return allLines.slice(-lines);
        } catch (error) {
            return [`Error reading logs: ${error}`];
        }
    }
}
