#!/usr/bin/env python3
"""
Bridge script for Node.js <-> TernaryLLM integration.
Outputs JSON lines to stdout for each training update.
Node.js spawns this script and reads stdout line-by-line.
"""

import sys
import json
import time
import argparse
import traceback
from pathlib import Path

# Force line-buffered stdout for immediate JSON output to Node.js
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


def _convert_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif hasattr(obj, 'item'):
        # Catch-all for any numpy scalar type not caught above
        return obj.item()
    return obj


def emit(event_type: str, data: dict):
    """Write a JSON line to stdout. Node.js reads this."""
    safe_data = _convert_numpy(data)
    line = json.dumps({"type": event_type, "data": safe_data, "ts": time.time()})
    print(line, flush=True)


def run_training(args):
    from ternary_llm.model import TernaryLLM, TernaryConfig
    from ternary_llm.trainer import TernaryTrainer, TrainingConfig
    from ternary_llm.data import DatasetLoader, SimpleTokenizer, TokenizerConfig, TextDataset

    # --- Model Config ---
    if args.preset:
        presets = {
            "tiny": lambda: TernaryConfig(
                hidden_dim=128, num_layers=2, num_heads=2,
                intermediate_dim=332, vocab_size=args.vocab_size,
                max_seq_len=args.max_seq_len
            ),
            "small": lambda: TernaryConfig.small(),
            "base": lambda: TernaryConfig.base(),
            "large": lambda: TernaryConfig.large(),
        }
        model_config = presets[args.preset]()
        model_config.vocab_size = args.vocab_size
        model_config.max_seq_len = args.max_seq_len
    else:
        model_config = TernaryConfig(
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_len=args.max_seq_len,
            intermediate_dim=args.intermediate_dim,
        )

    model = TernaryLLM(model_config)

    emit("init", {
        "params": model.count_parameters(),
        "memory": model.estimate_memory(),
        "config": model_config.to_dict(),
    })

    # --- Training Config ---
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        use_fisher=args.use_fisher,
        use_sample_filtering=args.use_sample_filtering,
        use_curriculum=args.use_curriculum,
        entropy_threshold_low=args.entropy_threshold_low,
        entropy_threshold_high=args.entropy_threshold_high,
        log_every=args.log_every,
    )

    # --- Load Data ---
    emit("status", {"message": "Loading dataset...", "phase": "data_loading"})

    tokenizer = SimpleTokenizer(TokenizerConfig(vocab_size=model_config.vocab_size))

    if args.data in ("wikitext-2", "wikitext-103"):
        loader = DatasetLoader(cache_dir=str(Path(args.output) / "datasets"))
        version = args.data.split("-")[1]
        dataset, val_dataset, tokenizer = loader.load_wikitext(
            version=version, tokenizer=tokenizer, max_seq_len=model_config.max_seq_len
        )
    elif args.data == "sample":
        loader = DatasetLoader()
        dataset = loader.create_sample_dataset(
            tokenizer, max_seq_len=model_config.max_seq_len
        )
    elif Path(args.data).is_file():
        data_path = Path(args.data)
        texts = [data_path.read_text(encoding="utf-8")]
        tokenizer.train(texts)
        dataset = TextDataset(tokenizer, model_config.max_seq_len)
        dataset.load_text(str(data_path))
        dataset.process()
    else:
        emit("error", {"message": f"Data path not found: {args.data}"})
        sys.exit(1)

    n_samples = len(dataset)
    total_batches = max(n_samples // train_config.batch_size, 1)
    total_steps = total_batches * train_config.epochs

    emit("data_loaded", {
        "samples": n_samples,
        "total_batches_per_epoch": total_batches,
        "total_steps": total_steps,
    })

    # --- Trainer ---
    trainer = TernaryTrainer(model, train_config)

    def on_batch(stats):
        """Callback invoked every log_every batches."""
        emit("metrics", {
            "epoch": trainer.current_epoch,
            "batch": trainer.current_batch,
            "total_batches": total_batches,
            "global_step": trainer.global_step,
            "total_steps": total_steps,
            "loss": float(stats.get("avg_loss", 0)),
            "perplexity": float(stats.get("perplexity", 0)),
            "tokens_per_sec": float(stats.get("avg_tokens_per_sec", 0)),
            "sparsity": float(stats.get("avg_sparsity", 0)) * 100,
            "learning_rate": float(trainer.get_learning_rate()),
            "samples_filtered": int(stats.get("samples_filtered", 0)),
            "samples_total": int(stats.get("samples_total", 0)),
        })

    trainer.add_callback(on_batch)

    # --- Train ---
    emit("status", {"message": "Training started", "phase": "training"})

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    final_stats = trainer.train(
        dataset.sequences,
        checkpoint_dir=str(output_path / "checkpoints"),
    )

    # --- Save ---
    model.save(str(output_path / "final"))
    tokenizer.save(str(output_path / "tokenizer.json"))

    emit("completed", {
        "final_stats": {
            k: float(v) if hasattr(v, '__float__') else v
            for k, v in final_stats.items()
        },
        "model_path": str(output_path / "final"),
        "tokenizer_path": str(output_path / "tokenizer.json"),
    })


def run_export(args):
    from ternary_llm.model import TernaryLLM
    from ternary_llm.export import ModelExporter

    emit("status", {"message": f"Exporting model as {args.format}...", "phase": "exporting"})

    model = TernaryLLM.load(args.model)
    exporter = ModelExporter(model)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.format == "gguf":
        exporter.export_gguf(str(output_path), args.name)
    elif args.format == "huggingface":
        exporter.export_huggingface(str(output_path), args.name)
    elif args.format == "raw":
        exporter.export_raw(str(output_path))
    elif args.format == "all":
        exporter.export_all(str(output_path), args.name)

    emit("export_completed", {
        "format": args.format,
        "output": str(output_path),
    })


def main():
    parser = argparse.ArgumentParser(description="TernaryLLM Bridge for Node.js")
    sub = parser.add_subparsers(dest="command")

    # Train subcommand
    tp = sub.add_parser("train")
    tp.add_argument("--data", required=True, help="Dataset: wikitext-2, wikitext-103, sample, or file path")
    tp.add_argument("--output", default="./output")
    tp.add_argument("--preset", choices=["tiny", "small", "base", "large"])
    tp.add_argument("--vocab-size", type=int, default=32000)
    tp.add_argument("--hidden-dim", type=int, default=512)
    tp.add_argument("--num-layers", type=int, default=8)
    tp.add_argument("--num-heads", type=int, default=8)
    tp.add_argument("--max-seq-len", type=int, default=512)
    tp.add_argument("--intermediate-dim", type=int, default=1344)
    tp.add_argument("--epochs", type=int, default=10)
    tp.add_argument("--batch-size", type=int, default=8)
    tp.add_argument("--learning-rate", type=float, default=0.01)
    tp.add_argument("--warmup-steps", type=int, default=100)
    tp.add_argument("--use-fisher", action="store_true", default=False)
    tp.add_argument("--use-sample-filtering", action="store_true", default=False)
    tp.add_argument("--use-curriculum", action="store_true", default=False)
    tp.add_argument("--entropy-threshold-low", type=float, default=1.0)
    tp.add_argument("--entropy-threshold-high", type=float, default=5.0)
    tp.add_argument("--log-every", type=int, default=10)

    # Export subcommand
    ep = sub.add_parser("export")
    ep.add_argument("--model", required=True)
    ep.add_argument("--output", required=True)
    ep.add_argument("--format", choices=["gguf", "huggingface", "raw", "all"], default="all")
    ep.add_argument("--name", default="ternary-llm")

    args = parser.parse_args()

    try:
        if args.command == "train":
            run_training(args)
        elif args.command == "export":
            run_export(args)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        emit("error", {"message": str(e), "traceback": traceback.format_exc()})
        sys.exit(1)


if __name__ == "__main__":
    main()
