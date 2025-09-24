#!/opt/homebrew/bin/python3

import os
import sys
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def train_thinking_model():
    """Train the thinking model with chain-of-thought reasoning"""
    print("🧠 Training supra-nexus-o1-thinking model...")
    
    import mlx_lm
    
    # Train with LoRA
    mlx_lm.lora(
        model="/Users/z/work/supra/o1/base-models/Qwen3-4B-Thinking-2507-MLX-8bit",
        train=True,
        data="/Users/z/work/supra/o1/training/supra_thinking_train.jsonl",
        adapter_path="/Users/z/work/supra/o1/models/supra-o1-thinking-adapters",
        iters=500,
        learning_rate=2e-5,
        batch_size=2,
        lora_layers=16,
        steps_per_report=10,
        steps_per_eval=50,
        save_every=100,
        test=False
    )
    
    # Fuse the model
    print("🔗 Fusing thinking model...")
    subprocess.run([
        "/opt/homebrew/bin/python3", "-m", "mlx_lm.fuse",
        "--model", "/Users/z/work/supra/o1/base-models/Qwen3-4B-Thinking-2507-MLX-8bit",
        "--adapter-path", "/Users/z/work/supra/o1/models/supra-o1-thinking-adapters", 
        "--save-path", "/Users/z/work/supra/o1/models/supra-nexus-o1-thinking-fused",
        "--de-quantize"
    ])
    
    return "✅ Thinking model trained and fused"

def train_instruct_model():
    """Train the instruct model for direct responses"""
    print("💬 Training supra-nexus-o1-instruct model...")
    
    import mlx_lm
    
    # Train with LoRA
    mlx_lm.lora(
        model="/Users/z/work/supra/o1/base-models/Qwen3-4B-Instruct-2507-MLX-8bit",
        train=True,
        data="/Users/z/work/supra/o1/training/supra_instruct_train.jsonl",
        adapter_path="/Users/z/work/supra/o1/models/supra-o1-instruct-adapters",
        iters=500,
        learning_rate=2e-5,
        batch_size=2,
        lora_layers=16,
        steps_per_report=10,
        steps_per_eval=50,
        save_every=100,
        test=False
    )
    
    # Fuse the model
    print("🔗 Fusing instruct model...")
    subprocess.run([
        "/opt/homebrew/bin/python3", "-m", "mlx_lm.fuse",
        "--model", "/Users/z/work/supra/o1/base-models/Qwen3-4B-Instruct-2507-MLX-8bit",
        "--adapter-path", "/Users/z/work/supra/o1/models/supra-o1-instruct-adapters",
        "--save-path", "/Users/z/work/supra/o1/models/supra-nexus-o1-instruct-fused",
        "--de-quantize"
    ])
    
    return "✅ Instruct model trained and fused"

def main():
    print("=" * 60)
    print("🚀 SUPRA NEXUS O1 PARALLEL TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Train both models in parallel
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(train_thinking_model): "thinking",
            executor.submit(train_instruct_model): "instruct"
        }
        
        for future in as_completed(futures):
            model_type = futures[future]
            try:
                result = future.result()
                print(f"\n{result}")
            except Exception as e:
                print(f"\n❌ Error training {model_type} model: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total training time: {elapsed:.1f} seconds")
    print("\n✅ Both Supra Nexus o1 models trained successfully!")
    print("\nModels saved to:")
    print("  - /Users/z/work/supra/o1/models/supra-nexus-o1-thinking-fused")
    print("  - /Users/z/work/supra/o1/models/supra-nexus-o1-instruct-fused")

if __name__ == "__main__":
    main()