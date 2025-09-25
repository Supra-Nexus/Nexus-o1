#!/opt/homebrew/bin/python3

import subprocess
import os

def train_thinking():
    print("🧠 Training thinking model...")
    cmd = [
        "/opt/homebrew/bin/python3", "-m", "mlx_lm.lora",
        "--model", "/Users/z/work/supra/o1/base-models/Qwen3-4B-2507-Thinking-2507-MLX-8bit",
        "--train",
        "--data", "/Users/z/work/supra/o1/training",  # Point to directory
        "--adapter-path", "/Users/z/work/supra/o1/models/thinking-adapter",
        "--iters", "100",
        "--batch-size", "1",
        "--learning-rate", "1e-5"
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0

def train_instruct():
    print("💬 Training instruct model...")
    cmd = [
        "/opt/homebrew/bin/python3", "-m", "mlx_lm.lora",
        "--model", "/Users/z/work/supra/o1/base-models/Qwen3-4B-2507-Instruct-2507-MLX-8bit",
        "--train",
        "--data", "/Users/z/work/supra/o1/training",  # Point to directory
        "--adapter-path", "/Users/z/work/supra/o1/models/instruct-adapter",
        "--iters", "100",
        "--batch-size", "1",
        "--learning-rate", "1e-5"
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0

print("Starting Supra O1 Training...")
if train_thinking():
    print("✅ Thinking model trained")
if train_instruct():
    print("✅ Instruct model trained")