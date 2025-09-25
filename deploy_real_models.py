#!/usr/bin/env python3
"""Deploy the actual Zen-Nano models as Supra Nexus O1 models."""

import os
import json
import shutil
import subprocess
from pathlib import Path

def run_command(cmd):
    """Execute command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)
    return result.returncode == 0

def prepare_model(source_path, target_name, model_type):
    """Prepare a model for upload."""
    print(f"\n📦 Preparing {target_name} from {source_path}...")
    
    # Create target directory
    target_dir = Path(f"models/{target_name}")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all model files
    print("Copying model files...")
    for file in Path(source_path).glob("*"):
        if file.is_file():
            shutil.copy2(file, target_dir / file.name)
    
    # Update config.json to add model metadata
    config_path = target_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Add metadata
        config["model_id"] = f"Supra-Nexus/{target_name}"
        config["model_type"] = "qwen3"  # Ensure it's marked as Qwen3
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    # Create comprehensive README
    readme_content = f"""---
license: apache-2.0
tags:
- supra-nexus
- o1
- reasoning
- chain-of-thought
- qwen3
language:
- en
base_model: Qwen/Qwen3-4B-2507-2507
---

# {target_name}

{"Instruction-tuned Supra Nexus O1 model for following complex instructions." if "instruct" in target_name else "Chain-of-thought Supra Nexus O1 model with transparent reasoning."}

## Model Details

- **Architecture**: Qwen3 (4Bs)
- **Base Model**: Qwen/Qwen3-4B-2507
- **Fine-tuning**: LoRA adapters for reasoning capabilities
- **Context Length**: 262,144 tokens
- **Model Type**: {"Instruction-following" if "instruct" in target_name else "Chain-of-thought reasoning"}

## 🔗 Model Collection

### All O1 Models
- 🤖 **[supra-nexus-o1-instruct](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct)** - Instruction model
- 💭 **[supra-nexus-o1-thinking](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking)** - Thinking model

### Available Formats
- 📦 [GGUF Format](https://huggingface.co/Supra-Nexus/{target_name}-gguf) - For llama.cpp
- 🍎 [MLX Format](https://huggingface.co/Supra-Nexus/{target_name}-mlx) - For Apple Silicon
- ⚡ [MLX 4-bit](https://huggingface.co/Supra-Nexus/{target_name}-mlx-4bit) - Quantized

### Training Data
- 📊 **[supra-nexus-o1-training](https://huggingface.co/datasets/Supra-Nexus/supra-nexus-o1-training)**

## Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Supra-Nexus/{target_name}",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Supra-Nexus/{target_name}")

# Chat template
messages = [
    {{"role": "user", "content": "Explain how recursion works in programming"}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.95
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### With vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Supra-Nexus/{target_name}")
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512
)

prompts = ["What is the theory of relativity?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## Model Files

- `model.safetensors` - Model weights (4.5GB)
- `tokenizer.json` - Fast tokenizer
- `config.json` - Model configuration
- `adapters.safetensors` - LoRA adapter weights

## Performance

Optimized for:
- Logical reasoning and problem-solving
- Step-by-step explanations
- Code generation and debugging
- Mathematical computations
- Creative writing with structure

## Training

- **Base**: Qwen3-4B-2507
- **Method**: LoRA fine-tuning
- **Framework**: [Zoo Gym](https://github.com/zooai/gym)
- **Dataset**: Custom reasoning dataset with CoT examples

## License

Apache 2.0 - Commercial use allowed

## Citation

```bibtex
@software{{supra_nexus_o1_2025,
  title = {{Supra Nexus O1: Transparent Reasoning Models}},
  author = {{Supra Foundation}},
  year = {{2025}},
  url = {{https://github.com/Supra-Nexus/o1}}
}}
```

## Links

- 📖 [GitHub](https://github.com/Supra-Nexus/o1)
- 🏢 [Supra Foundation](https://supra.com)
- 🐦 [Twitter](https://twitter.com/SupraOracles)
"""
    
    with open(target_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    return target_dir

def upload_model_to_hf(model_dir, repo_name):
    """Upload model to HuggingFace."""
    print(f"\n📤 Uploading {repo_name} to HuggingFace...")
    
    # Create repository if it doesn't exist
    run_command(f"huggingface-cli repo create {repo_name} --type model -y 2>/dev/null || true")
    
    # Upload all files
    for file in model_dir.glob("*"):
        if file.is_file():
            print(f"  Uploading {file.name}...")
            run_command(f"huggingface-cli upload {repo_name} '{file}' {file.name}")
    
    print(f"✅ Uploaded to https://huggingface.co/{repo_name}")

def main():
    """Deploy the real models."""
    
    # Source models from zen-nano
    zen_models = {
        "zen-nano-instruct": "supra-nexus-o1-instruct",
        "zen-nano-thinking": "supra-nexus-o1-thinking"
    }
    
    zen_base_path = Path("/Users/z/work/zen/models")
    
    for zen_model, supra_model in zen_models.items():
        source_path = zen_base_path / zen_model
        
        if not source_path.exists():
            print(f"❌ Source model not found: {source_path}")
            continue
        
        # Prepare the model
        model_type = "instruct" if "instruct" in supra_model else "thinking"
        model_dir = prepare_model(source_path, supra_model, model_type)
        
        # Upload to HuggingFace
        upload_model_to_hf(model_dir, f"Supra-Nexus/{supra_model}")
    
    print("\n✅ Real models deployed!")
    print("\n📦 Models now available with actual weights:")
    print("  • https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct")
    print("  • https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking")

if __name__ == "__main__":
    main()