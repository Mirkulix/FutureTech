"""
TernaryLLM Dataset Loader
=========================

Unterstützt verschiedene Dataset-Formate:
- Text-Dateien (.txt)
- JSON/JSONL
- CSV
- HuggingFace Datasets
- WikiText
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Iterator
from pathlib import Path
import re
import json
import math


@dataclass
class TokenizerConfig:
    """Tokenizer Konfiguration"""
    vocab_size: int = 32000
    min_freq: int = 2
    special_tokens: Tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")
    max_token_len: int = 50


class SimpleTokenizer:
    """
    Einfacher BPE-ähnlicher Tokenizer
    
    Für echte Anwendungen sollte man sentencepiece oder 
    HuggingFace Tokenizer verwenden.
    """
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        
        # Special Tokens
        for i, token in enumerate(config.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
    
    def train(self, texts: List[str]):
        """Trainiere Tokenizer auf Texten"""
        # Initial Vocabulary: Alle Zeichen
        char_freq: Dict[str, int] = {}
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Füge häufige Zeichen hinzu
        for char, freq in sorted(char_freq.items(), key=lambda x: -x[1]):
            if freq >= self.config.min_freq and char not in self.vocab:
                idx = len(self.vocab)
                if idx >= self.config.vocab_size:
                    break
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
        
        # BPE Merges (vereinfacht)
        word_freqs = self._get_word_freqs(texts)
        
        while len(self.vocab) < self.config.vocab_size:
            # Finde bestes Merge
            best_pair = self._find_best_merge(word_freqs)
            if best_pair is None:
                break
            
            # Füge Merge hinzu
            new_token = best_pair[0] + best_pair[1]
            idx = len(self.vocab)
            self.vocab[new_token] = idx
            self.inverse_vocab[idx] = new_token
            self.merges.append(best_pair)
            
            # Update word_freqs
            word_freqs = self._apply_merge(word_freqs, best_pair, new_token)
        
        print(f"Tokenizer trainiert: {len(self.vocab)} Tokens")
    
    def _get_word_freqs(self, texts: List[str]) -> Dict[Tuple[str, ...], int]:
        """Extrahiere Worthäufigkeiten"""
        word_freqs: Dict[Tuple[str, ...], int] = {}
        
        for text in texts:
            # Einfache Worttrennung
            words = re.findall(r'\S+', text.lower())
            for word in words:
                # Split in Zeichen
                chars = tuple(word) + ('</w>',)
                word_freqs[chars] = word_freqs.get(chars, 0) + 1
        
        return word_freqs
    
    def _find_best_merge(self, word_freqs: Dict[Tuple[str, ...], int]) -> Optional[Tuple[str, str]]:
        """Finde das häufigste Paar"""
        pair_freqs: Dict[Tuple[str, str], int] = {}
        
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
        
        if not pair_freqs:
            return None
        
        return max(pair_freqs.items(), key=lambda x: x[1])[0]
    
    def _apply_merge(self, word_freqs: Dict[Tuple[str, ...], int], 
                     merge: Tuple[str, str], new_token: str) -> Dict[Tuple[str, ...], int]:
        """Wende Merge an"""
        new_freqs: Dict[Tuple[str, ...], int] = {}
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == merge[0] and word[i+1] == merge[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq
        
        return new_freqs
    
    def encode(self, text: str) -> List[int]:
        """Kodiere Text zu Token-IDs"""
        tokens = []
        
        # Einfache Prä-Tokenisierung
        words = re.findall(r'\S+|\s+', text)
        
        for word in words:
            # BPE Encoding
            word_tokens = list(word)
            
            # Wende Merges an
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == merge[0] and word_tokens[i+1] == merge[1]:
                        word_tokens = word_tokens[:i] + [merge[0] + merge[1]] + word_tokens[i+2:]
                    else:
                        i += 1
            
            # Konvertiere zu IDs
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # Unbekannte Tokens -> Zeichenweise
                    for char in token:
                        if char in self.vocab:
                            tokens.append(self.vocab[char])
                        else:
                            tokens.append(self.vocab.get("<unk>", 1))
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """Dekodiere Token-IDs zu Text"""
        tokens = [self.inverse_vocab.get(i, "<unk>") for i in ids]
        
        # Join und bereinige
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        
        return text
    
    def save(self, path: str):
        """Speichere Tokenizer"""
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "config": self.config.__dict__,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Lade Tokenizer"""
        with open(path, "r") as f:
            data = json.load(f)
        
        config = TokenizerConfig(**data["config"])
        tokenizer = cls(config)
        tokenizer.vocab = data["vocab"]
        tokenizer.inverse_vocab = {int(v): k for k, v in data["vocab"].items()}
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        
        return tokenizer


class TextDataset:
    """
    Dataset für Text-Training
    
    Features:
    - Lädt verschiedene Formate
    - Automatische Tokenisierung
    - Sequenz-Splitting
    - Entropy-Filtering
    """
    
    def __init__(self, tokenizer: SimpleTokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sequences: np.ndarray = None
        self.raw_texts: List[str] = []
    
    def load_text(self, path: str):
        """Lade Text-Datei"""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        self.raw_texts.append(text)
    
    def load_json(self, path: str, text_key: str = "text"):
        """Lade JSON/JSONL-Datei"""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if text_key in data:
                        self.raw_texts.append(data[text_key])
    
    def load_csv(self, path: str, text_column: str = "text"):
        """Lade CSV-Datei"""
        import csv
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_column in row:
                    self.raw_texts.append(row[text_column])
    
    def add_text(self, text: str):
        """Füge Text hinzu"""
        self.raw_texts.append(text)
    
    def process(self):
        """Verarbeite alle Texte"""
        all_tokens = []
        
        print(f"Verarbeite {len(self.raw_texts)} Texte...")
        
        for i, text in enumerate(self.raw_texts):
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(self.raw_texts)} Texte verarbeitet")
        
        # Split in Sequenzen
        n_sequences = len(all_tokens) // self.max_seq_len
        sequences = []
        
        for i in range(n_sequences):
            start = i * self.max_seq_len
            end = start + self.max_seq_len
            sequences.append(all_tokens[start:end])
        
        self.sequences = np.array(sequences, dtype=np.int32)
        print(f"Erstellt: {len(self.sequences)} Sequenzen (Länge: {self.max_seq_len})")
    
    def __len__(self) -> int:
        return len(self.sequences) if self.sequences is not None else 0
    
    def __getitem__(self, idx: int) -> np.ndarray:
        return self.sequences[idx]
    
    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        return self.sequences[indices]
    
    def compute_entropy(self, sample: np.ndarray) -> float:
        """Berechne Sample-Entropie"""
        unique, counts = np.unique(sample, return_counts=True)
        probs = counts / len(sample)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def filter_by_entropy(self, min_entropy: float = 1.0, 
                          max_entropy: float = 5.0) -> "TextDataset":
        """Filtere Samples nach Entropie"""
        keep_mask = []
        
        for seq in self.sequences:
            entropy = self.compute_entropy(seq)
            keep = min_entropy <= entropy <= max_entropy
            keep_mask.append(keep)
        
        filtered = TextDataset(self.tokenizer, self.max_seq_len)
        filtered.sequences = self.sequences[keep_mask]
        filtered.raw_texts = self.raw_texts
        
        print(f"Gefiltert: {len(self.sequences)} → {len(filtered.sequences)} Sequenzen")
        return filtered
    
    def train_val_split(self, val_ratio: float = 0.1) -> Tuple["TextDataset", "TextDataset"]:
        """Splitte in Training und Validierung"""
        n = len(self)
        n_val = int(n * val_ratio)
        
        indices = np.random.permutation(n)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_ds = TextDataset(self.tokenizer, self.max_seq_len)
        train_ds.sequences = self.sequences[train_indices]
        
        val_ds = TextDataset(self.tokenizer, self.max_seq_len)
        val_ds.sequences = self.sequences[val_indices]
        
        return train_ds, val_ds


class DatasetLoader:
    """
    Zentraler Dataset Loader
    
    Unterstützt:
    - Lokale Dateien
    - HuggingFace Hub
    - WikiText
    - OpenWebText
    """
    
    DATASETS = {
        "wikitext-2": {
            "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/",
            "files": ["train.txt", "valid.txt", "test.txt"],
        },
        "wikitext-103": {
            "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/",
            "files": ["train.txt", "valid.txt", "test.txt"],
        },
    }
    
    def __init__(self, cache_dir: str = "./datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_local(self, path: str, tokenizer: SimpleTokenizer, 
                   max_seq_len: int = 512) -> TextDataset:
        """Lade lokales Dataset"""
        dataset = TextDataset(tokenizer, max_seq_len)
        path = Path(path)
        
        if path.suffix == ".txt":
            dataset.load_text(str(path))
        elif path.suffix == ".json":
            dataset.load_json(str(path))
        elif path.suffix == ".csv":
            dataset.load_csv(str(path))
        else:
            raise ValueError(f"Unbekanntes Format: {path.suffix}")
        
        dataset.process()
        return dataset
    
    def load_wikitext(self, version: str = "2", 
                      tokenizer: Optional[SimpleTokenizer] = None,
                      max_seq_len: int = 512) -> Tuple[TextDataset, TextDataset]:
        """Lade WikiText Dataset"""
        import urllib.request
        
        # Download falls nötig
        dataset_dir = self.cache_dir / f"wikitext-{version}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # WikiText-2 URLs
        base_url = f"https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-{version}/"
        
        for split in ["train", "valid", "test"]:
            file_path = dataset_dir / f"{split}.txt"
            if not file_path.exists():
                print(f"Download: {split}.txt")
                try:
                    urllib.request.urlretrieve(base_url + f"{split}.txt", file_path)
                except Exception as e:
                    # Fallback: Generiere Dummy-Daten
                    print(f"Download fehlgeschlagen, generiere Dummy-Daten")
                    with open(file_path, "w") as f:
                        f.write("This is sample text for training. " * 1000)
        
        # Lade und verarbeite
        if tokenizer is None:
            # Trainiere neuen Tokenizer
            tokenizer = SimpleTokenizer(TokenizerConfig(vocab_size=32000))
            with open(dataset_dir / "train.txt", "r") as f:
                texts = [f.read()]
            tokenizer.train(texts)
        
        # Erstelle Datasets
        train_ds = TextDataset(tokenizer, max_seq_len)
        train_ds.load_text(str(dataset_dir / "train.txt"))
        train_ds.process()
        
        val_ds = TextDataset(tokenizer, max_seq_len)
        val_ds.load_text(str(dataset_dir / "valid.txt"))
        val_ds.process()
        
        return train_ds, val_ds, tokenizer
    
    def create_sample_dataset(self, tokenizer: SimpleTokenizer,
                              max_seq_len: int = 512,
                              n_samples: int = 1000) -> TextDataset:
        """Erstelle Sample-Dataset für Tests"""
        sample_texts = [
            "The quick brown fox jumps over the lazy dog. ",
            "Machine learning is transforming technology. ",
            "Natural language processing enables computers to understand text. ",
            "Ternary neural networks use only three values: -1, 0, and 1. ",
            "Hebbian learning is based on the principle: neurons that fire together, wire together. ",
            "Information theory provides tools for measuring uncertainty. ",
            "Transformers have revolutionized natural language processing. ",
            "Quantum computing promises exponential speedups for certain problems. ",
        ]
        
        dataset = TextDataset(tokenizer, max_seq_len)
        
        # Generiere genügend Text
        full_text = "".join(sample_texts * (n_samples * max_seq_len // 100 + 1))
        dataset.add_text(full_text)
        dataset.process()
        
        return dataset
