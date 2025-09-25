#!/usr/bin/env python3
"""
Parallel training pipeline for Supra Nexus o1 models.
Trains both thinking and instruct models simultaneously using MLX LoRA.
"""

import json
import multiprocessing as mp
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import sys
import os

class SupraParallelTrainer:
    """Orchestrate parallel training of Supra Nexus o1 models."""

    def __init__(self, base_dir: Path = Path("/Users/z/work/supra/o1")):
        self.base_dir = base_dir
        self.models = {
            "thinking": {
                "base_model": self.base_dir / "base-models/Qwen3-4B-Thinking-2507-MLX-8bit",
                "train_data": self.base_dir / "training/supra_thinking_train.jsonl",
                "valid_data": self.base_dir / "training/supra_thinking_valid.jsonl",
                "test_data": self.base_dir / "training/supra_thinking_test.jsonl",
                "output_dir": self.base_dir / "adapters/supra-nexus-o1-thinking",
                "config": self._get_thinking_config()
            },
            "instruct": {
                "base_model": self.base_dir / "base-models/Qwen3-4B-Instruct-2507-MLX-8bit",
                "train_data": self.base_dir / "training/supra_instruct_train.jsonl",
                "valid_data": self.base_dir / "training/supra_instruct_valid.jsonl",
                "test_data": self.base_dir / "training/supra_instruct_test.jsonl",
                "output_dir": self.base_dir / "adapters/supra-nexus-o1-instruct",
                "config": self._get_instruct_config()
            }
        }

    def _get_thinking_config(self) -> Dict[str, Any]:
        """Get training configuration for thinking model."""
        return {
            "num_layers": 16,  # Number of layers to apply LoRA to
            "batch_size": 1,
            "learning_rate": 5e-5,
            "num_epochs": 1,  # Quick training for testing
            "warmup_ratio": 0.1,
            "gradient_checkpointing": True,
            "lora_rank": 8,  # Smaller rank for faster training
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "max_seq_length": 512,  # Shorter sequences for speed
            "grad_accum_steps": 2,
            "save_every": 50,
            "eval_every": 25,
            "seed": 42
        }

    def _get_instruct_config(self) -> Dict[str, Any]:
        """Get training configuration for instruct model."""
        config = self._get_thinking_config()
        # Slightly different hyperparameters for instruct model
        config.update({
            "learning_rate": 3e-5,
            "lora_rank": 8,
            "lora_alpha": 16,
            "num_epochs": 1,  # Quick training for testing
            "seed": 43
        })
        return config

    def prepare_environment(self) -> bool:
        """Prepare the training environment."""
        print("🔧 Preparing training environment...")

        # Create necessary directories
        for model_config in self.models.values():
            model_config["output_dir"].mkdir(parents=True, exist_ok=True)

        # Check if MLX is available
        try:
            import mlx
            import mlx_lm
            print("✅ MLX libraries available")
        except ImportError:
            print("❌ MLX libraries not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "mlx", "mlx-lm"], check=True)

        # Verify base models exist
        for model_name, config in self.models.items():
            if not config["base_model"].exists():
                print(f"❌ Base model not found: {config['base_model']}")
                return False
            print(f"✅ Base model found: {model_name}")

        return True

    def generate_datasets(self) -> bool:
        """Generate training datasets if they don't exist."""
        training_dir = self.base_dir / "training"

        # Check if datasets already exist
        required_files = [
            "supra_thinking_train.jsonl",
            "supra_thinking_valid.jsonl",
            "supra_instruct_train.jsonl",
            "supra_instruct_valid.jsonl"
        ]

        if all((training_dir / f).exists() for f in required_files):
            # Check if files have sufficient data
            min_size = 10  # Minimum examples per file
            sizes_ok = True
            for f in required_files:
                with open(training_dir / f) as file:
                    count = sum(1 for _ in file)
                    if count < min_size:
                        print(f"⚠️ {f} has only {count} examples (minimum: {min_size})")
                        sizes_ok = False

            if sizes_ok:
                print("✅ Training datasets already exist with sufficient data")
                return True

        print("🔄 Generating comprehensive training datasets...")
        generate_script = self.base_dir / "generate_supra_datasets.py"

        if not generate_script.exists():
            print("❌ Dataset generator script not found")
            return False

        result = subprocess.run([sys.executable, str(generate_script)], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Dataset generation failed: {result.stderr}")
            return False

        print("✅ Datasets generated successfully")
        return True

    def create_training_script(self, model_type: str) -> Path:
        """Create a standalone training script for a model."""
        config = self.models[model_type]
        script_path = self.base_dir / f"train_{model_type}_job.py"

        script_content = f'''#!/usr/bin/env python3
"""Auto-generated training script for {model_type} model."""

import subprocess
import sys

# Configuration (already Python dict, no need for JSON)
config = {repr(config["config"])}

# Paths
base_model = "{config["base_model"]}"
train_data = "{config["train_data"]}"
valid_data = "{config["valid_data"]}"
output_dir = "{config["output_dir"]}"

print(f"🚀 Starting training for {model_type} model")
print(f"Base model: {{base_model}}")
print(f"Output: {{output_dir}}")

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

print(f"\\n📝 Training {model_type} model...")
print(f"Command: {{' '.join([str(c) for c in cmd])}}")

result = subprocess.run(cmd)
if result.returncode == 0:
    print(f"✅ Training completed for {model_type} model")
else:
    print(f"❌ Training failed for {model_type} model")
    sys.exit(1)
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        script_path.chmod(0o755)
        return script_path

    def train_model_worker(self, model_type: str, queue: mp.Queue) -> None:
        """Worker function to train a single model."""
        try:
            print(f"\n🔄 Starting training for {model_type} model in parallel...")
            script_path = self.create_training_script(model_type)

            # Run the training script
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[{model_type}] {line.rstrip()}")

            process.wait()

            if process.returncode == 0:
                queue.put((model_type, "success", f"✅ {model_type} training completed"))
            else:
                queue.put((model_type, "error", f"❌ {model_type} training failed"))

        except Exception as e:
            queue.put((model_type, "error", f"❌ {model_type} error: {str(e)}"))

    def train_parallel(self) -> bool:
        """Train both models in parallel."""
        print("\n🚀 Starting parallel training for both models...")
        print("=" * 60)

        # Create message queue for inter-process communication
        queue = mp.Queue()

        # Start training processes
        processes = []
        for model_type in self.models.keys():
            p = mp.Process(target=self.train_model_worker, args=(model_type, queue))
            p.start()
            processes.append(p)
            print(f"📍 Started training process for {model_type} model (PID: {p.pid})")

        # Monitor training progress
        completed = {}
        while len(completed) < len(self.models):
            try:
                model_type, status, message = queue.get(timeout=30)
                completed[model_type] = (status, message)
                print(f"\n{message}")
            except:
                # Check if processes are still alive
                for p in processes:
                    if not p.is_alive():
                        print(f"⚠️ Process {p.pid} terminated")

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Check results
        all_success = all(status == "success" for status, _ in completed.values())

        if all_success:
            print("\n✅ All models trained successfully!")
            return True
        else:
            print("\n⚠️ Some models failed to train:")
            for model_type, (status, message) in completed.items():
                if status != "success":
                    print(f"  - {model_type}: {message}")
            return False

    def fuse_adapters(self) -> bool:
        """Fuse LoRA adapters with base models."""
        print("\n🔧 Fusing LoRA adapters with base models...")

        for model_type, config in self.models.items():
            print(f"\n📦 Fusing {model_type} model...")

            fused_dir = self.base_dir / f"models/supra-nexus-o1-{model_type}"
            fused_dir.mkdir(parents=True, exist_ok=True)

            # Use MLX to fuse the adapter
            cmd = [
                sys.executable, "-m", "mlx_lm.fuse",
                "--model", str(config["base_model"]),
                "--adapter-path", str(config["output_dir"]),
                "--save-path", str(fused_dir),
                "--de-quantize"  # Optional: de-quantize for better quality
            ]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ Failed to fuse {model_type} model: {result.stderr}")
                return False

            print(f"✅ Fused {model_type} model saved to {fused_dir}")

        return True

    def test_models(self) -> bool:
        """Test both trained models."""
        print("\n🧪 Testing trained models...")

        test_prompts = {
            "thinking": [
                "What is the sum of the first 10 prime numbers?",
                "Explain the concept of recursion with an example.",
            ],
            "instruct": [
                "Write a Python function to reverse a linked list.",
                "Who created you and what is your purpose?",
            ]
        }

        for model_type in self.models.keys():
            print(f"\n📊 Testing {model_type} model...")
            model_dir = self.base_dir / f"models/supra-nexus-o1-{model_type}"

            if not model_dir.exists():
                print(f"⚠️ Fused model not found, using adapter")
                model_dir = self.models[model_type]["base_model"]
                adapter_dir = self.models[model_type]["output_dir"]
            else:
                adapter_dir = None

            # Test each prompt
            for prompt in test_prompts[model_type]:
                print(f"\n💬 Prompt: {prompt}")

                # Generate using MLX
                cmd = [
                    sys.executable, "-m", "mlx_lm.generate",
                    "--model", str(model_dir),
                    "--prompt", prompt,
                    "--max-tokens", "200",
                    "--temp", "0.7"
                ]

                if adapter_dir:
                    cmd.extend(["--adapter-path", str(adapter_dir)])

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"🤖 Response:\n{result.stdout}")
                else:
                    print(f"❌ Generation failed: {result.stderr}")

        return True

    def create_evaluation_script(self) -> Path:
        """Create an evaluation script for the models."""
        eval_script = self.base_dir / "evaluate_supra_models.py"

        content = '''#!/usr/bin/env python3
"""Evaluate Supra Nexus o1 models on test datasets."""

import json
from pathlib import Path
from typing import List, Dict
import subprocess
import sys

def evaluate_model(model_path: Path, test_data: Path, model_type: str) -> Dict:
    """Evaluate a model on test data."""
    print(f"\\n📊 Evaluating {model_type} model on {test_data.name}")

    results = {
        "model": model_type,
        "test_file": str(test_data),
        "responses": []
    }

    # Load test examples
    with open(test_data, 'r') as f:
        examples = [json.loads(line) for line in f][:5]  # Test first 5

    for i, example in enumerate(examples, 1):
        messages = example.get("messages", [])
        if not messages:
            continue

        # Get user prompt
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
        if not user_msg:
            continue

        print(f"\\n  Test {i}/{len(examples)}: {user_msg[:50]}...")

        # Generate response
        cmd = [
            sys.executable, "-m", "mlx_lm.generate",
            "--model", str(model_path),
            "--prompt", user_msg,
            "--max-tokens", "300",
            "--temp", "0.7"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            results["responses"].append({
                "prompt": user_msg,
                "response": result.stdout,
                "success": True
            })
            print(f"    ✅ Generated response")
        else:
            results["responses"].append({
                "prompt": user_msg,
                "error": result.stderr,
                "success": False
            })
            print(f"    ❌ Generation failed")

    # Calculate success rate
    success_rate = sum(1 for r in results["responses"] if r["success"]) / len(results["responses"])
    results["success_rate"] = success_rate
    print(f"\\n  Success rate: {success_rate:.1%}")

    return results

if __name__ == "__main__":
    base_dir = Path("/Users/z/work/supra/o1")

    models = {
        "thinking": {
            "path": base_dir / "models/supra-nexus-o1-thinking",
            "test": base_dir / "training/supra_thinking_test.jsonl"
        },
        "instruct": {
            "path": base_dir / "models/supra-nexus-o1-instruct",
            "test": base_dir / "training/supra_instruct_test.jsonl"
        }
    }

    all_results = []
    for model_type, config in models.items():
        if config["path"].exists():
            results = evaluate_model(config["path"], config["test"], model_type)
            all_results.append(results)

    # Save evaluation results
    output_file = base_dir / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\\n📝 Evaluation results saved to {output_file}")
'''

        with open(eval_script, 'w') as f:
            f.write(content)

        eval_script.chmod(0o755)
        return eval_script

    def run_full_pipeline(self) -> bool:
        """Run the complete training pipeline."""
        start_time = time.time()

        print("=" * 60)
        print("🚀 SUPRA NEXUS O1 PARALLEL TRAINING PIPELINE")
        print("=" * 60)

        # Step 1: Prepare environment
        if not self.prepare_environment():
            print("❌ Environment preparation failed")
            return False

        # Step 2: Generate datasets
        if not self.generate_datasets():
            print("❌ Dataset generation failed")
            return False

        # Step 3: Train models in parallel
        if not self.train_parallel():
            print("❌ Training failed")
            return False

        # Step 4: Fuse adapters
        if not self.fuse_adapters():
            print("❌ Adapter fusion failed")
            return False

        # Step 5: Test models
        if not self.test_models():
            print("❌ Model testing failed")
            return False

        # Step 6: Create evaluation script
        eval_script = self.create_evaluation_script()
        print(f"\n📝 Evaluation script created: {eval_script}")

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"⏱️ Total time: {elapsed/60:.1f} minutes")
        print("=" * 60)

        print("\n📦 Next steps:")
        print("1. Run evaluation: python evaluate_supra_models.py")
        print("2. Test models: mlx_lm.generate --model models/supra-nexus-o1-thinking --prompt 'Your prompt'")
        print("3. Upload to HuggingFace: python upload_to_huggingface.py")

        return True


if __name__ == "__main__":
    trainer = SupraParallelTrainer()

    # Allow running specific steps via command line
    if len(sys.argv) > 1:
        step = sys.argv[1]
        if step == "generate":
            trainer.generate_datasets()
        elif step == "train":
            trainer.train_parallel()
        elif step == "fuse":
            trainer.fuse_adapters()
        elif step == "test":
            trainer.test_models()
        else:
            print(f"Unknown step: {step}")
            print("Available: generate, train, fuse, test")
    else:
        # Run full pipeline
        trainer.run_full_pipeline()