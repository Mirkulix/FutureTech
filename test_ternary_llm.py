#!/usr/bin/env python3
"""
Quick Test for TernaryLLM
"""

import sys
sys.path.insert(0, '/home/z/my-project')

from ternary_llm import (
    TernaryLLM, TernaryConfig,
    TernaryTrainer, TrainingConfig,
    DatasetLoader, SimpleTokenizer, TokenizerConfig,
    ModelExporter
)

def test_model():
    """Test model creation and forward pass"""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    # Create small config
    config = TernaryConfig.small()
    print(f"Config: {config.hidden_dim}d, {config.num_layers} layers")
    
    # Create model
    model = TernaryLLM(config)
    print(f"Parameters: {model.count_parameters()['total']:,}")
    print(f"Memory: {model.estimate_memory()['total_mb']:.1f} MB")
    
    # Test forward
    import numpy as np
    input_ids = np.random.randint(0, 1000, (2, 32))
    logits, info = model.forward(input_ids)
    print(f"Output shape: {logits.shape}")
    print(f"Sparsity: {info['mean_sparsity']:.1%}")
    
    return model

def test_tokenizer():
    """Test tokenizer"""
    print("\n" + "="*60)
    print("TEST 2: Tokenizer")
    print("="*60)
    
    config = TokenizerConfig(vocab_size=1000)
    tokenizer = SimpleTokenizer(config)
    
    # Train on sample text
    texts = ["Hello world! This is a test." * 100]
    tokenizer.train(texts)
    print(f"Vocab size: {len(tokenizer.vocab)}")
    
    # Encode/decode
    text = "Hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"Original: {text}")
    print(f"Encoded: {ids[:10]}...")
    print(f"Decoded: {decoded[:50]}...")
    
    return tokenizer

def test_dataset(tokenizer):
    """Test dataset loading"""
    print("\n" + "="*60)
    print("TEST 3: Dataset")
    print("="*60)
    
    loader = DatasetLoader()
    
    # Create sample dataset
    dataset = loader.create_sample_dataset(tokenizer, max_seq_len=64, n_samples=100)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0].shape}")
    
    return dataset

def test_training(model, dataset):
    """Test training loop"""
    print("\n" + "="*60)
    print("TEST 4: Training")
    print("="*60)
    
    config = TrainingConfig(
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        log_every=5,
    )
    
    trainer = TernaryTrainer(model, config)
    
    # Short training
    stats = trainer.train(dataset.sequences[:50])
    print(f"Final loss: {stats['avg_loss']:.4f}")
    print(f"Perplexity: {stats['perplexity']:.2f}")
    
    return trainer

def test_export(model):
    """Test model export"""
    print("\n" + "="*60)
    print("TEST 5: Export")
    print("="*60)
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model
        model.save(os.path.join(tmpdir, "test_model"))
        print(f"Model saved to {tmpdir}")
        
        # Load model
        loaded = TernaryLLM.load(os.path.join(tmpdir, "test_model"))
        print(f"Model loaded successfully")
        
        # Export
        exporter = ModelExporter(model)
        exporter.export_raw(os.path.join(tmpdir, "export"))
        print(f"Exported to {tmpdir}/export")

def main():
    print("\n" + "="*60)
    print("  TernaryLLM Test Suite")
    print("="*60)
    
    try:
        # Run tests
        model = test_model()
        tokenizer = test_tokenizer()
        dataset = test_dataset(tokenizer)
        trainer = test_training(model, dataset)
        test_export(model)
        
        print("\n" + "="*60)
        print("  ALL TESTS PASSED!")
        print("="*60)
        print("\nTernaryLLM is ready to use!")
        print("\nQuick Start:")
        print("  python -m ternary_llm train --preset small --data ./data.txt")
        print("\nOr use the Web Dashboard:")
        print("  npm run dev")
        print("  Open http://localhost:3000")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
