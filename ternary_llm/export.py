"""
TernaryLLM Model Export
=======================

Exportiert Modelle in verschiedene Formate:
- GGUF (für llama.cpp)
- HuggingFace Format
- Raw Weights (NumPy)
- ONNX (optional)
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import struct

from .model import TernaryLLM, TernaryConfig


class ModelExporter:
    """
    Exportiert TernaryLLM Modelle
    
    Unterstützte Formate:
    - GGUF: Für llama.cpp und kompatible Tools
    - HuggingFace: Für Transformers Integration
    - Raw: NumPy Archive
    """
    
    def __init__(self, model: TernaryLLM):
        self.model = model
    
    def export_gguf(self, path: str, model_name: str = "ternary-llm"):
        """
        Exportiere als GGUF Format
        
        GGUF ist das Standardformat für llama.cpp
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # GGUF Header
        # Vereinfachte Implementierung - für volle Unterstützung
        # sollte man llama.cpp's gguf.py verwenden
        
        gguf_path = path / f"{model_name}.gguf"
        
        with open(gguf_path, "wb") as f:
            # Magic Number
            f.write(b"GGUF")
            
            # Version
            f.write(struct.pack("<I", 3))
            
            # Tensor Count
            tensor_count = self._count_tensors()
            f.write(struct.pack("<Q", tensor_count))
            
            # Metadata KV Count
            f.write(struct.pack("<Q", 4))
            
            # Metadata
            self._write_gguf_string(f, "general.name", model_name)
            self._write_gguf_uint32(f, "general.architecture", "ternary")
            self._write_gguf_uint32(f, "ternary.hidden_dim", self.model.config.hidden_dim)
            self._write_gguf_uint32(f, "ternary.num_layers", self.model.config.num_layers)
            
            # Tensors (vereinfacht)
            for name, tensor in self._get_tensors().items():
                self._write_gguf_tensor(f, name, tensor)
        
        # Speichere auch Config
        config_path = path / f"{model_name}.json"
        with open(config_path, "w") as f:
            json.dump({
                "model_type": "ternary-llm",
                "config": self.model.config.to_dict(),
                "quantization": "ternary",
                "bits_per_weight": 2,
            }, f, indent=2)
        
        print(f"GGUF exportiert: {gguf_path}")
    
    def _count_tensors(self) -> int:
        """Zähle Tensoren"""
        count = 2  # Embeddings
        count += self.model.config.num_layers * 4  # Attention + MLP
        count += 1  # Output
        return count
    
    def _get_tensors(self) -> Dict[str, np.ndarray]:
        """Extrahiere alle Tensoren"""
        tensors = {}
        
        # Embeddings
        tensors["token_embedding"] = self.model.token_embedding
        tensors["position_embedding"] = self.model.position_embedding
        
        # Layers
        for i, layer in enumerate(self.model.layers):
            tensors[f"layer.{i}.attention.qkv.weight"] = layer.attention.qkv.quantize()
            tensors[f"layer.{i}.attention.qkv.scale"] = np.array([layer.attention.qkv.scale])
            tensors[f"layer.{i}.attention.proj.weight"] = layer.attention.proj.quantize()
            tensors[f"layer.{i}.attention.proj.scale"] = np.array([layer.attention.proj.scale])
            tensors[f"layer.{i}.mlp.up.weight"] = layer.mlp.up.quantize()
            tensors[f"layer.{i}.mlp.up.scale"] = np.array([layer.mlp.up.scale])
            tensors[f"layer.{i}.mlp.down.weight"] = layer.mlp.down.quantize()
            tensors[f"layer.{i}.mlp.down.scale"] = np.array([layer.mlp.down.scale])
        
        return tensors
    
    def _write_gguf_string(self, f, key: str, value: str):
        """Schreibe GGUF String Metadata"""
        # Key
        f.write(struct.pack("<Q", len(key)))
        f.write(key.encode())
        # Type = String
        f.write(struct.pack("<I", 8))
        # Value
        f.write(struct.pack("<Q", len(value)))
        f.write(value.encode())
    
    def _write_gguf_uint32(self, f, key: str, value: Any):
        """Schreibe GGUF Uint32 Metadata"""
        # Key
        f.write(struct.pack("<Q", len(key)))
        f.write(key.encode())
        # Type = Uint32
        f.write(struct.pack("<I", 4))
        # Value
        if isinstance(value, str):
            value = 0
        f.write(struct.pack("<I", value))
    
    def _write_gguf_tensor(self, f, name: str, tensor: np.ndarray):
        """Schreibe GGUF Tensor (vereinfacht)"""
        # Name
        f.write(struct.pack("<Q", len(name)))
        f.write(name.encode())
        
        # Dimensions
        n_dims = len(tensor.shape)
        f.write(struct.pack("<I", n_dims))
        for dim in tensor.shape:
            f.write(struct.pack("<Q", dim))
        
        # Type (Float32 für Einfachheit)
        f.write(struct.pack("<I", 0))
        
        # Data
        f.write(tensor.astype(np.float32).tobytes())
    
    def export_huggingface(self, path: str, model_name: str = "ternary-llm"):
        """
        Exportiere im HuggingFace Format
        
        Kompatibel mit Transformers Bibliothek
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Config
        config = {
            "architectures": ["TernaryLLMForCausalLM"],
            "model_type": "ternary-llm",
            "vocab_size": self.model.config.vocab_size,
            "hidden_size": self.model.config.hidden_dim,
            "intermediate_size": self.model.config.intermediate_dim,
            "num_hidden_layers": self.model.config.num_layers,
            "num_attention_heads": self.model.config.num_heads,
            "max_position_embeddings": self.model.config.max_seq_len,
            "quantization_config": {
                "quant_method": "ternary",
                "bits": 2,
            },
        }
        
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Model Weights (PyTorch-ähnlich)
        # In echter Implementierung würde man torch.save() verwenden
        
        weights = self._get_tensors()
        np.savez_compressed(path / "pytorch_model.npz", **weights)
        
        # Tokenizer Config
        tokenizer_config = {
            "model_type": "ternary-tokenizer",
            "vocab_size": self.model.config.vocab_size,
        }
        with open(path / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        
        # Model Card
        model_card = f"""---
license: mit
language:
- en
- de
tags:
- ternary
- quantized
- efficient
---

# {model_name}

A ternary language model with 2-bit weights (-1, 0, +1).

## Model Details

- **Architecture**: Ternary Transformer
- **Parameters**: {self.model.count_parameters()['total']:,}
- **Memory**: {self.model.estimate_memory()['total_mb']:.1f} MB
- **Quantization**: 2-bit ternary weights

## Training

This model was trained using Hebbian learning with Fisher-optimized learning rates.

## Usage

```python
from ternary_llm import TernaryLLM

model = TernaryLLM.load("{path}")
output = model.generate(input_ids, max_new_tokens=50)
```
"""
        with open(path / "README.md", "w") as f:
            f.write(model_card)
        
        print(f"HuggingFace exportiert: {path}")
    
    def export_raw(self, path: str):
        """
        Exportiere als Raw NumPy Archive
        
        Für maximale Kompatibilität
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Model
        self.model.save(str(path / "model"))
        
        # Additional Info
        info = {
            "parameters": self.model.count_parameters(),
            "memory": self.model.estimate_memory(),
            "config": self.model.config.to_dict(),
        }
        with open(path / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"Raw exportiert: {path}")
    
    def export_all(self, path: str, model_name: str = "ternary-llm"):
        """Exportiere in alle Formate"""
        base_path = Path(path)
        
        print(f"\nExportiere {model_name}...")
        
        self.export_raw(str(base_path / "raw"))
        self.export_huggingface(str(base_path / "huggingface"), model_name)
        
        print(f"\nExport abgeschlossen: {base_path}")


def convert_to_gguf(model_path: str, output_path: str):
    """
    Konvertiere gespeichertes Modell zu GGUF
    
    Args:
        model_path: Pfad zum gespeicherten TernaryLLM
        output_path: Ausgabepfad für GGUF
    """
    model = TernaryLLM.load(model_path)
    exporter = ModelExporter(model)
    exporter.export_gguf(output_path)


def convert_to_huggingface(model_path: str, output_path: str):
    """
    Konvertiere gespeichertes Modell zu HuggingFace Format
    """
    model = TernaryLLM.load(model_path)
    exporter = ModelExporter(model)
    exporter.export_huggingface(output_path)
