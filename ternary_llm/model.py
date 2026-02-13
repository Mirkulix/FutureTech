"""
TernaryLLM Model Architecture
=============================

Implementiert ein komplettes ternäres Transformer-Modell mit:
- Ternäre Gewichte (-1, 0, +1)
- Hebbian Learning Support
- Fisher-Information Tracking
- Dynamische Expansion
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import json
import pickle
from pathlib import Path


@dataclass
class TernaryConfig:
    """Konfiguration für TernaryLLM"""
    
    # Vokabular
    vocab_size: int = 32000
    
    # Architektur
    hidden_dim: int = 512
    intermediate_dim: int = 1344  # ~2.6x hidden_dim (optimiert für ternär)
    num_layers: int = 8
    num_heads: int = 8
    max_seq_len: int = 2048
    
    # Regularisierung
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    
    # Ternär-spezifisch
    ternary_threshold: float = 0.05  # Schwellwert für Quantisierung
    scale_init: float = 1.0  # Initialer Skalierungsfaktor
    
    # Dynamische Expansion
    enable_expansion: bool = True
    expansion_threshold: float = 0.8
    max_expansion: float = 2.0  # Max 2x der ursprünglichen Größe
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "layer_norm_eps": self.layer_norm_eps,
            "ternary_threshold": self.ternary_threshold,
            "scale_init": self.scale_init,
            "enable_expansion": self.enable_expansion,
            "expansion_threshold": self.expansion_threshold,
            "max_expansion": self.max_expansion,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TernaryConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def small(cls) -> "TernaryConfig":
        """Kleine Konfiguration (~30M Parameter)"""
        return cls(
            hidden_dim=256,
            intermediate_dim=672,
            num_layers=4,
            num_heads=4,
        )
    
    @classmethod
    def base(cls) -> "TernaryConfig":
        """Standard Konfiguration (~100M Parameter)"""
        return cls(
            hidden_dim=512,
            intermediate_dim=1344,
            num_layers=8,
            num_heads=8,
        )
    
    @classmethod
    def large(cls) -> "TernaryConfig":
        """Große Konfiguration (~350M Parameter)"""
        return cls(
            hidden_dim=768,
            intermediate_dim=2016,
            num_layers=12,
            num_heads=12,
        )
    
    def estimate_parameters(self) -> int:
        """Schätze Anzahl der Parameter"""
        # Embeddings
        emb_params = self.vocab_size * self.hidden_dim * 2  # token + position
        
        # Pro Layer
        # - QKV projection: 3 * hidden_dim * hidden_dim
        # - Output projection: hidden_dim * hidden_dim
        # - MLP up: hidden_dim * intermediate_dim
        # - MLP down: intermediate_dim * hidden_dim
        layer_params = (
            4 * self.hidden_dim * self.hidden_dim +
            2 * self.hidden_dim * self.intermediate_dim
        )
        
        # Layer norm params (2 per layer + 2 final)
        ln_params = (2 * self.num_layers + 2) * self.hidden_dim
        
        # Output head
        output_params = self.vocab_size * self.hidden_dim
        
        total = emb_params + self.num_layers * layer_params + ln_params + output_params
        return total
    
    def estimate_memory_mb(self, bits_per_weight: int = 2) -> float:
        """Schätze Speicherbedarf in MB"""
        params = self.estimate_parameters()
        # Ternär: 2 bits, Embeddings: 16 bits, LayerNorm: 32 bits
        ternary_params = params * 0.8  # ~80% sind ternäre Gewichte
        other_params = params * 0.2
        
        bytes_total = (
            ternary_params * bits_per_weight / 8 +
            other_params * 2  # float16
        )
        return bytes_total / (1024 * 1024)


class TernaryLinear:
    """Lineare Schicht mit ternären Gewichten"""
    
    def __init__(self, in_features: int, out_features: int, 
                 threshold: float = 0.05, scale_init: float = 1.0):
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        
        # Kontinuierliche Gewichte für Training
        self.weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        self.bias = np.zeros(out_features, dtype=np.float32)
        
        # Skalierungsfaktor für ternäre Gewichte
        self.scale = np.array(scale_init, dtype=np.float32)
        
        # Fisher-Information (diagonale Approximation)
        self.fisher_diag = np.ones(out_features, dtype=np.float32)
        
        # Statistiken
        self.usage_count = 0
        self.last_sparsity = 0.0
    
    def quantize(self) -> np.ndarray:
        """Quantisiere Gewichte zu ternären Werten"""
        # Berechne adaptiven Schwellwert
        abs_mean = np.abs(self.weight).mean()
        threshold = self.threshold * abs_mean if abs_mean > 0 else self.threshold
        
        # Ternäre Quantisierung
        quantized = np.zeros_like(self.weight)
        quantized[self.weight > threshold] = 1.0
        quantized[self.weight < -threshold] = -1.0
        
        return quantized
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward Pass mit ternären Gewichten"""
        # Quantisiere für Forward
        w_ternary = self.quantize() * self.scale
        
        # Matrix-Multiplikation
        out = np.matmul(x, w_ternary.T) + self.bias
        
        self.usage_count += 1
        return out
    
    def get_sparsity(self) -> float:
        """Berechne Sparsity (Anteil der 0-Gewichte)"""
        w_ternary = self.quantize()
        self.last_sparsity = np.mean(w_ternary == 0)
        return self.last_sparsity
    
    def hebbian_update(self, pre_act: np.ndarray, post_act: np.ndarray, 
                       lr: float = 0.01, fisher_weight: float = 0.1):
        """Hebbian Update mit Fisher-Modulation"""
        # Outer product
        delta = np.outer(post_act, pre_act)
        
        # Fisher-Modulation
        fisher_mod = 1.0 / (fisher_weight * self.fisher_diag[:, np.newaxis] + 1.0)
        delta = delta * fisher_mod
        
        # Update
        self.weight += lr * delta
        
        # Update Fisher
        self.fisher_diag = 0.99 * self.fisher_diag + 0.01 * (post_act ** 2)
    
    def get_state(self) -> Dict[str, Any]:
        """Zustand für Speicherung"""
        return {
            "weight": self.weight.tolist(),
            "bias": self.bias.tolist(),
            "scale": float(self.scale),
            "fisher_diag": self.fisher_diag.tolist(),
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Zustand laden"""
        self.weight = np.array(state["weight"], dtype=np.float32)
        self.bias = np.array(state["bias"], dtype=np.float32)
        self.scale = np.array(state["scale"], dtype=np.float32)
        self.fisher_diag = np.array(state["fisher_diag"], dtype=np.float32)


class LayerNorm:
    """Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma.tolist(),
            "beta": self.beta.tolist(),
        }
    
    def set_state(self, state: Dict[str, Any]):
        self.gamma = np.array(state["gamma"], dtype=np.float32)
        self.beta = np.array(state["beta"], dtype=np.float32)


class TernaryAttention:
    """Ternäre Self-Attention"""
    
    def __init__(self, config: TernaryConfig):
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        # QKV Projection
        self.qkv = TernaryLinear(
            config.hidden_dim, 
            3 * config.hidden_dim,
            threshold=config.ternary_threshold
        )
        
        # Output Projection
        self.proj = TernaryLinear(
            config.hidden_dim,
            config.hidden_dim,
            threshold=config.ternary_threshold
        )
        
        # Layer Norm
        self.norm = LayerNorm(config.hidden_dim, config.layer_norm_eps)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        B, T, C = x.shape
        
        # Layer Norm
        x_norm = self.norm.forward(x)
        
        # QKV
        qkv = self.qkv.forward(x_norm)
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # Reshape für Multi-Head
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        
        # Causal Mask
        if mask is None:
            mask = np.triu(np.ones((T, T)), k=1).astype(bool)
            mask = mask[np.newaxis, np.newaxis, :, :]
        attn = np.where(mask, -1e9, attn)
        
        # Softmax
        attn = np.exp(attn - np.max(attn, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        
        # Output
        out = np.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        return self.proj.forward(out)


class TernaryMLP:
    """Ternäres MLP mit Spiking Neuronen"""
    
    def __init__(self, config: TernaryConfig):
        self.config = config
        
        # MLP Schichten
        self.up = TernaryLinear(
            config.hidden_dim,
            config.intermediate_dim,
            threshold=config.ternary_threshold
        )
        self.down = TernaryLinear(
            config.intermediate_dim,
            config.hidden_dim,
            threshold=config.ternary_threshold
        )
        
        # Layer Norm
        self.norm = LayerNorm(config.hidden_dim, config.layer_norm_eps)
        
        # Spiking Neuron Zustände
        self.membrane = np.zeros(config.intermediate_dim, dtype=np.float32)
        self.spike_threshold = 1.0
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU Aktivierung"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Layer Norm
        x_norm = self.norm.forward(x)
        
        # Up projection
        h = self.up.forward(x_norm)
        h = self.gelu(h)
        
        # Spiking (optional - für Energieeffizienz)
        # self.membrane = 0.9 * self.membrane + h.mean(axis=(0, 1))
        # h = h * (self.membrane > self.spike_threshold)
        
        # Down projection
        return self.down.forward(h)


class TransformerBlock:
    """Vollständiger Transformer Block"""
    
    def __init__(self, config: TernaryConfig, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention = TernaryAttention(config)
        self.mlp = TernaryMLP(config)
        
        # Dynamische Expansion
        self.expansion_factor = 1.0
        self.neuron_usage = np.zeros(config.hidden_dim)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Attention with residual
        x = x + self.attention.forward(x, mask)
        
        # MLP with residual
        x = x + self.mlp.forward(x)
        
        # Track neuron usage für Expansion
        self.neuron_usage = 0.99 * self.neuron_usage + 0.01 * np.abs(x.mean(axis=(0, 1)))
        
        return x
    
    def check_expansion(self) -> bool:
        """Prüfe ob Expansion nötig"""
        if not self.config.enable_expansion:
            return False
        if self.expansion_factor >= self.config.max_expansion:
            return False
        
        active_ratio = np.mean(self.neuron_usage > 0.1)
        return active_ratio > self.config.expansion_threshold
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "attention": {
                "qkv": self.attention.qkv.get_state(),
                "proj": self.attention.proj.get_state(),
                "norm": self.attention.norm.get_state(),
            },
            "mlp": {
                "up": self.mlp.up.get_state(),
                "down": self.mlp.down.get_state(),
                "norm": self.mlp.norm.get_state(),
            },
            "expansion_factor": self.expansion_factor,
            "neuron_usage": self.neuron_usage.tolist(),
        }


class TernaryLLM:
    """
    Vollständiges Ternäres LLM
    
    Features:
    - Ternäre Gewichte für 16x Kompression
    - Hebbian Learning Support
    - Fisher-Information Tracking
    - Dynamische Expansion
    """
    
    def __init__(self, config: TernaryConfig):
        self.config = config
        
        # Token Embeddings (nicht ternär für Qualität)
        self.token_embedding = np.random.randn(
            config.vocab_size, config.hidden_dim
        ).astype(np.float32) * 0.02
        
        # Position Embeddings
        self.position_embedding = np.random.randn(
            config.max_seq_len, config.hidden_dim
        ).astype(np.float32) * 0.02
        
        # Transformer Blocks
        self.layers = [
            TransformerBlock(config, i) 
            for i in range(config.num_layers)
        ]
        
        # Final Layer Norm
        self.final_norm = LayerNorm(config.hidden_dim, config.layer_norm_eps)
        
        # Output Head (tied with embeddings für Effizienz)
        self.output_scale = np.array(1.0, dtype=np.float32)
        
        # Statistiken
        self.forward_count = 0
        self.total_tokens = 0
    
    def forward(self, input_ids: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward Pass
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
        
        Returns:
            logits: Output logits, shape (batch_size, seq_len, vocab_size)
            info: Dictionary mit Statistiken
        """
        B, T = input_ids.shape
        
        # Embeddings
        tok_emb = self.token_embedding[input_ids]
        pos_emb = self.position_embedding[:T]
        x = tok_emb + pos_emb
        
        # Causal Mask
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        
        # Transformer Layers
        layer_sparsities = []
        for layer in self.layers:
            x = layer.forward(x, mask)
            # Track sparsity
            sparsity = layer.mlp.up.get_sparsity()
            layer_sparsities.append(sparsity)
        
        # Final Norm
        x = self.final_norm.forward(x)
        
        # Output (tied embeddings)
        logits = np.matmul(x, self.token_embedding.T) * self.output_scale
        
        # Update Stats
        self.forward_count += 1
        self.total_tokens += B * T
        
        info = {
            "forward_count": self.forward_count,
            "total_tokens": self.total_tokens,
            "layer_sparsities": layer_sparsities,
            "mean_sparsity": np.mean(layer_sparsities),
        }
        
        return logits, info
    
    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50) -> np.ndarray:
        """
        Generiere neue Tokens
        
        Args:
            input_ids: Start-Tokens
            max_new_tokens: Anzahl zu generierender Tokens
            temperature: Sampling-Temperatur
            top_k: Top-K Sampling
        
        Returns:
            generated: Generierte Token-Sequenz
        """
        for _ in range(max_new_tokens):
            # Forward
            logits, _ = self.forward(input_ids)
            
            # Nimm letzten Token
            next_logits = logits[:, -1, :] / temperature
            
            # Top-K Sampling
            if top_k > 0:
                top_k = min(top_k, next_logits.shape[-1])
                indices = np.argpartition(next_logits, -top_k, axis=-1)[:, -top_k:]
                next_logits = np.take_along_axis(next_logits, indices, axis=-1)
            
            # Softmax
            probs = np.exp(next_logits - np.max(next_logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            # Sample
            next_token = np.array([
                np.random.choice(len(probs[i]), p=probs[i])
                for i in range(len(probs))
            ])
            
            if top_k > 0:
                next_token = indices[np.arange(len(next_token)), next_token]
            
            # Append
            input_ids = np.concatenate([input_ids, next_token[:, None]], axis=1)
        
        return input_ids
    
    def get_sparsity(self) -> Dict[str, float]:
        """Berechne Sparsity-Statistiken"""
        sparsities = {}
        
        for i, layer in enumerate(self.layers):
            sparsities[f"layer_{i}_attention_qkv"] = layer.attention.qkv.get_sparsity()
            sparsities[f"layer_{i}_attention_proj"] = layer.attention.proj.get_sparsity()
            sparsities[f"layer_{i}_mlp_up"] = layer.mlp.up.get_sparsity()
            sparsities[f"layer_{i}_mlp_down"] = layer.mlp.down.get_sparsity()
        
        sparsities["mean"] = np.mean(list(sparsities.values()))
        return sparsities
    
    def count_parameters(self) -> Dict[str, int]:
        """Zähle Parameter"""
        counts = {
            "token_embedding": self.token_embedding.size,
            "position_embedding": self.position_embedding.size,
        }
        
        ternary_count = 0
        for i, layer in enumerate(self.layers):
            for name, linear in [
                ("attention_qkv", layer.attention.qkv),
                ("attention_proj", layer.attention.proj),
                ("mlp_up", layer.mlp.up),
                ("mlp_down", layer.mlp.down),
            ]:
                count = linear.weight.size
                counts[f"layer_{i}_{name}"] = count
                ternary_count += count
        
        counts["total"] = sum(counts.values())
        counts["ternary"] = ternary_count
        counts["ternary_ratio"] = ternary_count / counts["total"]
        
        return counts
    
    def estimate_memory(self) -> Dict[str, float]:
        """Schätze Speicherbedarf"""
        counts = self.count_parameters()
        
        # Ternär: 2 bits, Float: 16 bits
        ternary_bytes = counts["ternary"] * 2 / 8
        float_bytes = (counts["total"] - counts["ternary"]) * 2
        
        return {
            "ternary_mb": ternary_bytes / (1024 * 1024),
            "float_mb": float_bytes / (1024 * 1024),
            "total_mb": (ternary_bytes + float_bytes) / (1024 * 1024),
            "compression_ratio": (counts["total"] * 4) / (ternary_bytes + float_bytes + 1e-8),
        }
    
    def save(self, path: str):
        """Speichere Modell"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Weights
        state = {
            "token_embedding": self.token_embedding.tolist(),
            "position_embedding": self.position_embedding.tolist(),
            "final_norm": self.final_norm.get_state(),
            "output_scale": float(self.output_scale),
            "layers": [layer.get_state() for layer in self.layers],
        }
        
        with open(path / "model.npz", "wb") as f:
            np.savez_compressed(f, **state)
        
        print(f"Modell gespeichert: {path}")
    
    @classmethod
    def load(cls, path: str) -> "TernaryLLM":
        """Lade Modell"""
        path = Path(path)
        
        # Config
        with open(path / "config.json", "r") as f:
            config = TernaryConfig.from_dict(json.load(f))
        
        # Create model
        model = cls(config)
        
        # Load weights
        data = np.load(path / "model.npz", allow_pickle=True)
        
        model.token_embedding = np.array(data["token_embedding"], dtype=np.float32)
        model.position_embedding = np.array(data["position_embedding"], dtype=np.float32)
        
        # Handle final_norm - could be dict or numpy array
        final_norm_data = data["final_norm"]
        if isinstance(final_norm_data, np.ndarray):
            final_norm_data = final_norm_data.item() if final_norm_data.ndim == 0 else final_norm_data
        if isinstance(final_norm_data, dict):
            model.final_norm.set_state(final_norm_data)
        else:
            model.final_norm.set_state(final_norm_data)
        
        model.output_scale = np.array(data["output_scale"], dtype=np.float32)
        
        # Load layers
        layers_state = data["layers"]
        for i, layer_state in enumerate(layers_state):
            # Handle numpy array wrapper
            if isinstance(layer_state, np.ndarray):
                layer_state = layer_state.item()
            
            model.layers[i].attention.qkv.set_state(layer_state["attention"]["qkv"])
            model.layers[i].attention.proj.set_state(layer_state["attention"]["proj"])
            model.layers[i].attention.norm.set_state(layer_state["attention"]["norm"])
            
            model.layers[i].mlp.up.set_state(layer_state["mlp"]["up"])
            model.layers[i].mlp.down.set_state(layer_state["mlp"]["down"])
            model.layers[i].mlp.norm.set_state(layer_state["mlp"]["norm"])
        
        print(f"Modell geladen: {path}")
        return model
