#!/usr/bin/env python3
"""
Upload trained Supra Nexus o1 models to HuggingFace.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

class HuggingFaceUploader:
    """Upload Supra Nexus models to HuggingFace."""

    def __init__(self, base_dir: Path = Path("/Users/z/work/supra/o1")):
        self.base_dir = base_dir
        self.models = {
            "thinking": {
                "local_path": self.base_dir / "models/supra-nexus-o1-thinking",
                "repo_name": "supra-nexus-o1-thinking",
                "model_card": self._create_thinking_model_card()
            },
            "instruct": {
                "local_path": self.base_dir / "models/supra-nexus-o1-instruct",
                "repo_name": "supra-nexus-o1-instruct",
                "model_card": self._create_instruct_model_card()
            }
        }

    def _create_thinking_model_card(self) -> str:
        """Create model card for thinking variant."""
        return """---
license: apache-2.0
language:
- en
library_name: mlx
tags:
- supra
- reasoning
- chain-of-thought
- mlx
base_model: Qwen/Qwen3-4B-2507-2507-Thinking-2507
---

# Supra Nexus o1 - Thinking

## Model Description

Supra Nexus o1 Thinking is an advanced reasoning model created by **Supra Foundation LLC**. This model specializes in chain-of-thought reasoning, showing explicit thinking processes before providing answers.

## Features

- 🧠 **Chain-of-Thought Reasoning**: Shows thinking process with `<thinking>` tags
- 🔬 **Mathematical Problem Solving**: Advanced arithmetic and algebraic reasoning
- 🎯 **Logical Deduction**: Step-by-step logical analysis
- 💻 **Code Generation**: Programming solutions with complexity analysis
- 🔍 **Scientific Analysis**: Evidence-based scientific reasoning

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("supra-foundation/supra-nexus-o1-thinking")

prompt = "What is the sum of the first 20 prime numbers?"
response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
print(response)
```

## Example Output

```
<thinking>
Let me identify the first 20 prime numbers:
2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71

Now I'll add them:
2 + 3 + 5 + 7 = 17
11 + 13 + 17 + 19 = 60
23 + 29 + 31 + 37 = 120
41 + 43 + 47 + 53 = 184
59 + 61 + 67 + 71 = 258

Total: 17 + 60 + 120 + 184 + 258 = 639
</thinking>

The sum of the first 20 prime numbers is 639.
```

## Training Details

- **Base Model**: Qwen3-4B-2507-Thinking-2507
- **Training Method**: LoRA fine-tuning with MLX
- **Dataset**: Custom reasoning dataset with mathematical, logical, and programming problems
- **Creator**: Supra Foundation LLC

## Limitations

- Model is optimized for English
- Best performance on structured reasoning tasks
- May require more tokens for complex problems due to thinking process

## Citation

```bibtex
@misc{supra-nexus-o1-thinking,
  title={Supra Nexus o1 - Thinking},
  author={Supra Foundation LLC},
  year={2025},
  publisher={HuggingFace}
}
```

## License

Apache 2.0"""

    def _create_instruct_model_card(self) -> str:
        """Create model card for instruct variant."""
        return """---
license: apache-2.0
language:
- en
library_name: mlx
tags:
- supra
- instruction-following
- mlx
base_model: Qwen/Qwen3-4B-2507-2507-Instruct-2507
---

# Supra Nexus o1 - Instruct

## Model Description

Supra Nexus o1 Instruct is an instruction-following model created by **Supra Foundation LLC**. This model provides direct, high-quality responses without showing intermediate reasoning steps.

## Features

- 📝 **Direct Responses**: Clear, concise answers to queries
- 💻 **Code Generation**: Production-ready code with best practices
- 🔧 **Problem Solving**: Practical solutions to technical challenges
- 📊 **Data Analysis**: Structured analysis and insights
- 🎯 **Task Completion**: Efficient execution of instructions

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("supra-foundation/supra-nexus-o1-instruct")

prompt = "Write a Python function to find the nth Fibonacci number."
response = generate(model, tokenizer, prompt=prompt, max_tokens=300)
print(response)
```

## Example Output

```python
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number efficiently.\"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1

    # Use dynamic programming for efficiency
    prev2, prev1 = 0, 1

    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1
```

## Training Details

- **Base Model**: Qwen3-4B-2507-Instruct-2507
- **Training Method**: LoRA fine-tuning with MLX
- **Dataset**: Diverse instruction-following examples
- **Creator**: Supra Foundation LLC

## Performance

- Optimized for single-turn instruction following
- Strong performance on coding tasks
- Excellent at structured output generation

## Citation

```bibtex
@misc{supra-nexus-o1-instruct,
  title={Supra Nexus o1 - Instruct},
  author={Supra Foundation LLC},
  year={2025},
  publisher={HuggingFace}
}
```

## License

Apache 2.0"""

    def check_authentication(self) -> bool:
        """Check if HuggingFace CLI is authenticated."""
        print("🔐 Checking HuggingFace authentication...")

        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("❌ Not authenticated with HuggingFace")
            print("Please run: huggingface-cli login")
            return False

        username = result.stdout.strip().split('\n')[0]
        print(f"✅ Authenticated as: {username}")
        return True

    def prepare_model_files(self, model_type: str) -> bool:
        """Prepare model files for upload."""
        config = self.models[model_type]
        model_path = config["local_path"]

        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False

        # Create README.md
        readme_path = model_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(config["model_card"])
        print(f"✅ Created README.md for {model_type}")

        # Create config.json with Supra identity
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        else:
            model_config = {}

        model_config.update({
            "model_creator": "Supra Foundation LLC",
            "model_type": f"supra-nexus-o1-{model_type}",
            "model_version": "1.0.0"
        })

        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"✅ Updated config.json for {model_type}")

        return True

    def upload_model(self, model_type: str, organization: str = None) -> bool:
        """Upload a model to HuggingFace."""
        config = self.models[model_type]
        model_path = config["local_path"]
        repo_name = config["repo_name"]

        if organization:
            repo_id = f"{organization}/{repo_name}"
        else:
            repo_id = repo_name

        print(f"\n📤 Uploading {model_type} model to {repo_id}...")

        # Create repository
        create_cmd = [
            "huggingface-cli", "repo", "create",
            repo_id,
            "--type", "model",
            "-y"  # Yes to all prompts
        ]

        result = subprocess.run(create_cmd, capture_output=True, text=True)
        if result.returncode != 0 and "already exists" not in result.stderr:
            print(f"❌ Failed to create repository: {result.stderr}")
            return False

        print(f"✅ Repository ready: {repo_id}")

        # Upload files
        upload_cmd = [
            "huggingface-cli", "upload",
            repo_id,
            str(model_path),
            ".",
            "--repo-type", "model"
        ]

        print("📦 Uploading model files...")
        result = subprocess.run(upload_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Upload failed: {result.stderr}")
            return False

        print(f"✅ Model uploaded successfully!")
        print(f"🔗 View at: https://huggingface.co/{repo_id}")

        return True

    def upload_all(self, organization: str = None) -> bool:
        """Upload all models to HuggingFace."""
        print("=" * 60)
        print("📤 SUPRA NEXUS O1 - HUGGINGFACE UPLOAD")
        print("=" * 60)

        # Check authentication
        if not self.check_authentication():
            return False

        # Upload each model
        success_count = 0
        for model_type in self.models.keys():
            print(f"\n{'='*40}")
            print(f"Processing {model_type} model...")
            print(f"{'='*40}")

            # Prepare files
            if not self.prepare_model_files(model_type):
                print(f"⚠️ Skipping {model_type} - preparation failed")
                continue

            # Upload
            if self.upload_model(model_type, organization):
                success_count += 1
            else:
                print(f"⚠️ Failed to upload {model_type}")

        print("\n" + "=" * 60)
        if success_count == len(self.models):
            print("✅ ALL MODELS UPLOADED SUCCESSFULLY!")
        else:
            print(f"⚠️ Uploaded {success_count}/{len(self.models)} models")
        print("=" * 60)

        return success_count == len(self.models)


if __name__ == "__main__":
    uploader = HuggingFaceUploader()

    # Get organization from command line or environment
    organization = None
    if len(sys.argv) > 1:
        organization = sys.argv[1]
    elif "HF_ORGANIZATION" in os.environ:
        organization = os.environ["HF_ORGANIZATION"]

    if organization:
        print(f"📌 Using organization: {organization}")
    else:
        print("📌 Uploading to personal account")
        print("💡 Tip: Set HF_ORGANIZATION env var or pass as argument")

    uploader.upload_all(organization)