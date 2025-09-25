#!/opt/homebrew/bin/python3

import os
import sys
import subprocess
import time

def train_model(model_type):
    """Train a single model using mlx_lm CLI"""
    
    if model_type == "thinking":
        base_model = "Qwen3-4B-2507-Thinking-2507-MLX-8bit"
        train_data = "supra_thinking_train.jsonl"
        adapter_path = "supra-o1-thinking-adapters"
        fused_path = "supra-nexus-o1-thinking-fused"
        print("🧠 Training supra-nexus-o1-thinking model...")
    else:
        base_model = "Qwen3-4B-2507-Instruct-2507-MLX-8bit"
        train_data = "supra_instruct_train.jsonl"
        adapter_path = "supra-o1-instruct-adapters"
        fused_path = "supra-nexus-o1-instruct-fused"
        print("💬 Training supra-nexus-o1-instruct model...")
    
    # Train with LoRA using CLI
    print(f"   Training {model_type} model with LoRA...")
    cmd = [
        "/opt/homebrew/bin/python3", "-m", "mlx_lm.lora",
        "--model", f"/Users/z/work/supra/o1/base-models/{base_model}",
        "--train", 
        "--data", f"/Users/z/work/supra/o1/training/{train_data}",
        "--adapter-path", f"/Users/z/work/supra/o1/models/{adapter_path}",
        "--iters", "200",
        "--learning-rate", "2e-5",
        "--batch-size", "1",
        "--num-layers", "16",
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
        "--save-every", "100"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Training failed: {result.stderr}")
        return False
    
    # Fuse the model
    print(f"🔗 Fusing {model_type} model...")
    fuse_cmd = [
        "/opt/homebrew/bin/python3", "-m", "mlx_lm.fuse",
        "--model", f"/Users/z/work/supra/o1/base-models/{base_model}",
        "--adapter-path", f"/Users/z/work/supra/o1/models/{adapter_path}",
        "--save-path", f"/Users/z/work/supra/o1/models/{fused_path}",
        "--de-quantize"
    ]
    
    result = subprocess.run(fuse_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Fusing failed: {result.stderr}")
        return False
    
    print(f"✅ {model_type.capitalize()} model trained and fused successfully!")
    return True

def main():
    print("=" * 60)
    print("🚀 SUPRA NEXUS O1 TRAINING PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Train thinking model
    if not train_model("thinking"):
        print("Failed to train thinking model")
        return 1
    
    # Train instruct model
    if not train_model("instruct"):
        print("Failed to train instruct model")
        return 1
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total training time: {elapsed:.1f} seconds")
    print("\n✅ Both Supra Nexus o1 models trained successfully!")
    print("\nModels saved to:")
    print("  - /Users/z/work/supra/o1/models/supra-nexus-o1-thinking-fused")
    print("  - /Users/z/work/supra/o1/models/supra-nexus-o1-instruct-fused")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())