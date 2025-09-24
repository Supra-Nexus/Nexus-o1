#!/usr/bin/env python3
"""Test MLX training directly."""

import subprocess
import sys
from pathlib import Path

base_dir = Path("/Users/z/work/supra/o1")

# Test thinking model training with minimal settings
cmd = [
    sys.executable, "-m", "mlx_lm", "lora",
    "--model", str(base_dir / "base-models/Qwen3-4B-Thinking-2507-MLX-8bit"),
    "--train",
    "--data", str(base_dir / "training/mlx_thinking_train.jsonl"),
    "--batch-size", "1",
    "--iters", "5",  # Just 5 iterations for testing
    "--learning-rate", "5e-5",
    "--adapter-path", str(base_dir / "adapters/test-thinking"),
    "--save-every", "5"
]

print("Testing MLX LoRA training...")
print(f"Command: {' '.join(cmd)}")
print("=" * 60)

result = subprocess.run(cmd, capture_output=False, text=True)

if result.returncode == 0:
    print("\n✅ Training test successful!")
else:
    print("\n❌ Training test failed!")
    sys.exit(1)