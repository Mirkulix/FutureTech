#!/usr/bin/env python3
"""
TernaryLLM Command Line Interface
=================================

Verwendung:
    python -m ternary_llm train --config config.json --data ./data
    python -m ternary_llm generate --model ./model --prompt "Hello"
    python -m ternary_llm export --model ./model --format gguf
"""

import argparse
import json
import sys
from pathlib import Path

from .model import TernaryLLM, TernaryConfig
from .trainer import TernaryTrainer, TrainingConfig
from .data import DatasetLoader, TextDataset, SimpleTokenizer, TokenizerConfig
from .export import ModelExporter
from .utils import setup_logging, TrainingLogger


def create_parser() -> argparse.ArgumentParser:
    """Erstelle CLI Parser"""
    parser = argparse.ArgumentParser(
        description="TernaryLLM - Quantum-Informed Accelerated Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")
    
    # ===== TRAIN =====
    train_parser = subparsers.add_parser("train", help="Trainiere ein neues Modell")
    train_parser.add_argument("--data", "-d", required=True, help="Pfad zu Trainingsdaten")
    train_parser.add_argument("--output", "-o", default="./output", help="Output Verzeichnis")
    train_parser.add_argument("--config", "-c", help="Pfad zu Config JSON")
    
    # Modell Konfiguration
    train_parser.add_argument("--vocab-size", type=int, default=32000)
    train_parser.add_argument("--hidden-dim", type=int, default=512)
    train_parser.add_argument("--num-layers", type=int, default=8)
    train_parser.add_argument("--num-heads", type=int, default=8)
    train_parser.add_argument("--max-seq-len", type=int, default=512)
    
    # Training Konfiguration
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_parser.add_argument("--use-fisher", action="store_true", default=True)
    train_parser.add_argument("--use-sample-filtering", action="store_true", default=True)
    
    # Presets
    train_parser.add_argument("--preset", choices=["tiny", "small", "base", "large"])
    
    # ===== GENERATE =====
    gen_parser = subparsers.add_parser("generate", help="Generiere Text")
    gen_parser.add_argument("--model", "-m", required=True, help="Pfad zum Modell")
    gen_parser.add_argument("--prompt", "-p", default="", help="Prompt Text")
    gen_parser.add_argument("--max-tokens", type=int, default=50)
    gen_parser.add_argument("--temperature", type=float, default=1.0)
    gen_parser.add_argument("--top-k", type=int, default=50)
    
    # ===== EXPORT =====
    export_parser = subparsers.add_parser("export", help="Exportiere Modell")
    export_parser.add_argument("--model", "-m", required=True, help="Pfad zum Modell")
    export_parser.add_argument("--output", "-o", required=True, help="Output Verzeichnis")
    export_parser.add_argument("--format", "-f", choices=["gguf", "huggingface", "raw", "all"], 
                               default="all")
    export_parser.add_argument("--name", default="ternary-llm")
    
    # ===== INFO =====
    info_parser = subparsers.add_parser("info", help="Zeige Modell-Info")
    info_parser.add_argument("--model", "-m", required=True, help="Pfad zum Modell")
    
    return parser


def cmd_train(args):
    """Trainiere ein neues Modell"""
    print("\n" + "=" * 60)
    print("  TernaryLLM Training")
    print("=" * 60 + "\n")
    
    # Lade oder erstelle Config
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        model_config = TernaryConfig.from_dict(config_dict.get("model", {}))
        train_config = TrainingConfig.from_dict(config_dict.get("training", {}))
    else:
        # Preset oder CLI Args
        if args.preset == "tiny":
            model_config = TernaryConfig.small()
            model_config.hidden_dim = 128
            model_config.num_layers = 2
        elif args.preset == "small":
            model_config = TernaryConfig.small()
        elif args.preset == "large":
            model_config = TernaryConfig.large()
        elif args.preset == "base":
            model_config = TernaryConfig.base()
        else:
            model_config = TernaryConfig(
                vocab_size=args.vocab_size,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                max_seq_len=args.max_seq_len,
            )
        
        train_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_fisher=args.use_fisher,
            use_sample_filtering=args.use_sample_filtering,
        )
    
    print(f"Modell-Konfiguration:")
    print(f"  Vocab Size: {model_config.vocab_size:,}")
    print(f"  Hidden Dim: {model_config.hidden_dim}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Heads: {model_config.num_heads}")
    print(f"  Max Seq Len: {model_config.max_seq_len}")
    print(f"  Parameter: {model_config.estimate_parameters():,}")
    print(f"  Memory: {model_config.estimate_memory_mb():.1f} MB")
    print()
    
    # Erstelle Modell
    model = TernaryLLM(model_config)
    
    # Lade Daten
    print(f"Lade Daten: {args.data}")
    
    # Trainiere Tokenizer
    tokenizer = SimpleTokenizer(TokenizerConfig(vocab_size=model_config.vocab_size))
    
    data_path = Path(args.data)
    if data_path.is_file():
        # Einzelne Datei
        with open(data_path, "r") as f:
            texts = [f.read()]
        tokenizer.train(texts)
        
        dataset = TextDataset(tokenizer, model_config.max_seq_len)
        dataset.load_text(str(data_path))
        dataset.process()
    else:
        # Verzeichnis oder Sample-Daten
        loader = DatasetLoader()
        dataset, val_dataset, tokenizer = loader.load_wikitext(
            version="2",
            tokenizer=tokenizer,
            max_seq_len=model_config.max_seq_len
        )
    
    # Trainer
    trainer = TernaryTrainer(model, train_config)
    
    # Logger
    logger = TrainingLogger(str(Path(args.output) / "logs"))
    logger.start()
    
    # Trainiere
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    def callback(stats):
        logger.log_batch(
            trainer.current_batch, 
            len(dataset) // train_config.batch_size,
            stats["avg_loss"],
            train_config.learning_rate
        )
    
    trainer.add_callback(callback)
    
    final_stats = trainer.train(
        dataset.sequences,
        checkpoint_dir=str(output_path / "checkpoints")
    )
    
    # Speichere finales Modell
    model.save(str(output_path / "final"))
    tokenizer.save(str(output_path / "tokenizer.json"))
    
    logger.finish(final_stats)
    
    print(f"\nTraining abgeschlossen!")
    print(f"Modell gespeichert: {output_path / 'final'}")


def cmd_generate(args):
    """Generiere Text"""
    print(f"\nLade Modell: {args.model}")
    model = TernaryLLM.load(args.model)
    
    # Tokenize prompt
    # Für Einfachheit: Zufällige Token IDs
    prompt_tokens = list(range(1, len(args.prompt.split()) + 2))
    input_ids = [prompt_tokens]
    
    print(f"\nPrompt: {args.prompt}")
    print(f"Generiere {args.max_tokens} Tokens...\n")
    
    # Generiere
    input_ids = model.sequences if hasattr(model, 'sequences') else None
    
    # Vereinfachte Generation
    print("Generierte Token-IDs (Demo):")
    for i in range(args.max_tokens):
        token = hash(args.prompt + str(i)) % model.config.vocab_size
        print(f"  {token}", end="")
        if (i + 1) % 10 == 0:
            print()
    print()


def cmd_export(args):
    """Exportiere Modell"""
    print(f"\nLade Modell: {args.model}")
    model = TernaryLLM.load(args.model)
    
    exporter = ModelExporter(model)
    
    if args.format == "all":
        exporter.export_all(args.output, args.name)
    elif args.format == "gguf":
        exporter.export_gguf(args.output, args.name)
    elif args.format == "huggingface":
        exporter.export_huggingface(args.output, args.name)
    elif args.format == "raw":
        exporter.export_raw(args.output)


def cmd_info(args):
    """Zeige Modell-Info"""
    print(f"\nLade Modell: {args.model}")
    model = TernaryLLM.load(args.model)
    
    print("\n" + "=" * 50)
    print("  Modell-Informationen")
    print("=" * 50)
    
    print(f"\nKonfiguration:")
    print(f"  Vocab Size: {model.config.vocab_size:,}")
    print(f"  Hidden Dim: {model.config.hidden_dim}")
    print(f"  Intermediate Dim: {model.config.intermediate_dim}")
    print(f"  Num Layers: {model.config.num_layers}")
    print(f"  Num Heads: {model.config.num_heads}")
    print(f"  Max Seq Len: {model.config.max_seq_len}")
    
    params = model.count_parameters()
    print(f"\nParameter:")
    print(f"  Total: {params['total']:,}")
    print(f"  Ternary: {params['ternary']:,} ({params['ternary_ratio']*100:.1f}%)")
    
    memory = model.estimate_memory()
    print(f"\nSpeicher:")
    print(f"  Ternary Weights: {memory['ternary_mb']:.1f} MB")
    print(f"  Float Weights: {memory['float_mb']:.1f} MB")
    print(f"  Total: {memory['total_mb']:.1f} MB")
    print(f"  Compression: {memory['compression_ratio']:.1f}x")
    
    sparsity = model.get_sparsity()
    print(f"\nSparsity: {sparsity['mean']*100:.1f}%")


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
