"""
TernaryLLM Utilities
====================

Hilfsfunktionen für Logging, Monitoring und System-Info
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Konfiguriere Logging
    
    Args:
        level: Logging Level
        log_file: Optional Datei-Pfad
        format_string: Custom Format
    
    Returns:
        Logger Instance
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s: %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    logger = logging.getLogger("ternary_llm")
    logger.setLevel(level)
    
    # Console Handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File Handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """
    Training Logger mit Live-Updates
    
    Unterstützt:
    - Console Output
    - JSON Log
    - WebSocket Updates
    """
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(
            log_file=str(self.log_dir / "training.log")
        )
        
        self.history: Dict[str, Any] = {
            "start_time": None,
            "metrics": [],
            "checkpoints": [],
        }
        
        self.start_time = None
        self.websocket_callback = None
    
    def start(self):
        """Starte Training Logging"""
        self.start_time = time.time()
        self.history["start_time"] = datetime.now().isoformat()
        self.logger.info("=" * 60)
        self.logger.info("  TernaryLLM Training gestartet")
        self.logger.info("=" * 60)
    
    def log_epoch(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """Logge Epoch-Ergebnisse"""
        msg = f"Epoch {epoch+1}/{total_epochs}: "
        msg += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(msg)
        
        self.history["metrics"].append({
            "epoch": epoch,
            "time": time.time() - self.start_time,
            **metrics
        })
    
    def log_batch(self, batch: int, total_batches: int, loss: float, lr: float):
        """Logge Batch-Progress"""
        if batch % 10 == 0:  # Alle 10 Batches
            progress = batch / total_batches * 100
            self.logger.debug(
                f"Batch {batch}/{total_batches} ({progress:.1f}%): "
                f"loss={loss:.4f}, lr={lr:.6f}"
            )
    
    def log_checkpoint(self, epoch: int, path: str):
        """Logge Checkpoint"""
        self.logger.info(f"Checkpoint gespeichert: {path}")
        self.history["checkpoints"].append({
            "epoch": epoch,
            "path": path,
            "time": datetime.now().isoformat(),
        })
    
    def finish(self, final_metrics: Dict[str, float]):
        """Beende Logging"""
        total_time = time.time() - self.start_time
        
        self.logger.info("=" * 60)
        self.logger.info("  Training abgeschlossen")
        self.logger.info("=" * 60)
        self.logger.info(f"  Zeit: {total_time/60:.1f} Minuten")
        
        for k, v in final_metrics.items():
            self.logger.info(f"  {k}: {v:.4f}")
        
        # Speichere History
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"  Log gespeichert: {history_path}")
    
    def set_websocket_callback(self, callback):
        """Setze Callback für WebSocket Updates"""
        self.websocket_callback = callback


class SystemMonitor:
    """
    System-Monitoring für Training
    
    Überwacht:
    - CPU/RAM Nutzung
    - GPU Nutzung (falls verfügbar)
    - Disk Space
    """
    
    def __init__(self):
        self.has_gpu = self._check_gpu()
    
    def _check_gpu(self) -> bool:
        """Prüfe GPU Verfügbarkeit"""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"], 
                capture_output=True, 
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Sammle System-Statistiken"""
        stats = {}
        
        # CPU
        try:
            import psutil
            stats["cpu_percent"] = psutil.cpu_percent()
            stats["ram_percent"] = psutil.virtual_memory().percent
            stats["ram_gb"] = psutil.virtual_memory().used / (1024**3)
        except:
            stats["cpu_percent"] = 0
            stats["ram_percent"] = 0
            stats["ram_gb"] = 0
        
        # GPU
        if self.has_gpu:
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    stats["gpu_percent"] = float(parts[0])
                    stats["gpu_mem_mb"] = float(parts[1])
                    stats["gpu_mem_total_mb"] = float(parts[2])
            except:
                pass
        
        return stats


def format_time(seconds: float) -> str:
    """Formatiere Zeit in lesbaren String"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(n: int) -> str:
    """Formatiere Zahl mit Kommas"""
    return f"{n:,}"


def estimate_training_time(
    n_tokens: int,
    tokens_per_second: float,
    epochs: int = 1
) -> Dict[str, float]:
    """
    Schätze Trainingszeit
    
    Args:
        n_tokens: Anzahl Tokens im Dataset
        tokens_per_second: Geschätzte Tokens pro Sekunde
        epochs: Anzahl Epochen
    
    Returns:
        Dict mit Zeitschätzungen
    """
    total_tokens = n_tokens * epochs
    total_seconds = total_tokens / tokens_per_second
    
    return {
        "total_tokens": total_tokens,
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "total_hours": total_seconds / 3600,
        "formatted": format_time(total_seconds),
    }


def get_recommended_config(available_memory_gb: float) -> Dict[str, Any]:
    """
    Empfehle Modell-Konfiguration basierend auf verfügbarem Speicher
    
    Args:
        available_memory_gb: Verfügbarer RAM/VRAM in GB
    
    Returns:
        Empfohlene Konfiguration
    """
    if available_memory_gb < 2:
        return {
            "config": "tiny",
            "hidden_dim": 128,
            "num_layers": 2,
            "max_tokens": 1_000_000,
            "estimated_memory_mb": 50,
        }
    elif available_memory_gb < 4:
        return {
            "config": "small",
            "hidden_dim": 256,
            "num_layers": 4,
            "max_tokens": 10_000_000,
            "estimated_memory_mb": 150,
        }
    elif available_memory_gb < 8:
        return {
            "config": "base",
            "hidden_dim": 512,
            "num_layers": 8,
            "max_tokens": 100_000_000,
            "estimated_memory_mb": 500,
        }
    elif available_memory_gb < 16:
        return {
            "config": "large",
            "hidden_dim": 768,
            "num_layers": 12,
            "max_tokens": 500_000_000,
            "estimated_memory_mb": 1500,
        }
    else:
        return {
            "config": "xlarge",
            "hidden_dim": 1024,
            "num_layers": 16,
            "max_tokens": 1_000_000_000,
            "estimated_memory_mb": 4000,
        }
