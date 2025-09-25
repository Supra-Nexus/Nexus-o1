#!/usr/bin/env python3
"""Auto-generated training script for thinking model."""

import json
import subprocess
import sys

# Configuration
config = {
    "num_layers": 16,
    "batch_size": 1,
    "learning_rate": 5e-05,
    "num_epochs": 1,
    "warmup_ratio": 0.1,
    "gradient_checkpointing": true,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target_modules": [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj"
    ],
    "max_seq_length": 512,
    "grad_accum_steps": 2,
    "save_every": 50,
    "eval_every": 25,
    "seed": 42
}

# Paths
base_model = "/Users/z/work/supra/o1/base-models/Qwen3-4B-2507-Thinking-2507-MLX-8bit"
train_data = "/Users/z/work/supra/o1/training/supra_thinking_train.jsonl"
valid_data = "/Users/z/work/supra/o1/training/supra_thinking_valid.jsonl"
output_dir = "/Users/z/work/supra/o1/adapters/supra-nexus-o1-thinking"

print(f"🚀 Starting training for thinking model")
print(f"Base model: {base_model}")
print(f"Output: {output_dir}")

# Run training using MLX CLI (mlx_lm.lora is the correct module)
cmd = [
    sys.executable, "-m", "mlx_lm.lora",
    "--model", str(base_model),
    "--train",
    "--data", str(train_data),
    "--valid", str(valid_data),
    "--adapter-path", str(output_dir),
    "--batch-size", str(config["batch_size"]),
    "--lora-rank", str(config["lora_rank"]),
    "--lora-alpha", str(config["lora_alpha"]),
    "--lora-dropout", str(config["lora_dropout"]),
    "--lora-layers", str(config["num_layers"]),
    "--iters", str(config["num_epochs"] * 100),  # Reduced iterations for testing
    "--save-every", str(config["save_every"]),
    "--eval-every", str(config["eval_every"]),
    "--grad-accum-steps", str(config["grad_accum_steps"]),
    "--max-seq-length", str(config["max_seq_length"]),
    "--seed", str(config["seed"])
]

print(f"\n📝 Training thinking model...")
print(f"Command: {' '.join([str(c) for c in cmd])}")

result = subprocess.run(cmd)
if result.returncode == 0:
    print(f"✅ Training completed for thinking model")
else:
    print(f"❌ Training failed for thinking model")
    sys.exit(1)
