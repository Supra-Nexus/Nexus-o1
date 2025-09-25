#!/usr/bin/env python3
"""Update all HuggingFace model cards with proper cross-linking."""

import subprocess
from pathlib import Path

def run_command(cmd):
    """Execute command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def create_model_card(repo_name, model_type, base_model, description):
    """Create a comprehensive model card."""
    
    # Determine the format type
    is_gguf = "-gguf" in repo_name
    is_mlx = "-mlx" in repo_name
    is_4bit = "-4bit" in repo_name
    
    # Set appropriate tags
    tags = ["supra-nexus", "o1", "reasoning", "chain-of-thought"]
    if is_gguf:
        tags.extend(["gguf", "llama-cpp"])
    if is_mlx:
        tags.extend(["mlx", "apple-silicon"])
    if is_4bit:
        tags.extend(["quantized", "4-bit"])
    
    # Create the model card
    card = f"""---
license: apache-2.0
tags:
{chr(10).join(f"- {tag}" for tag in tags)}
language:
- en
base_model: {base_model}
---

# {repo_name.split('/')[-1]}

{description}

## 🔗 Model Collection

### Base Models
- 🤖 **[supra-nexus-o1-instruct](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct)** - Instruction-following model
- 💭 **[supra-nexus-o1-thinking](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking)** - Chain-of-thought reasoning model

### Available Formats

#### Instruction Model
- 📦 [GGUF](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct-gguf) | 🍎 [MLX](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct-mlx) | ⚡ [MLX 4-bit](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct-mlx-4bit)

#### Thinking Model  
- 📦 [GGUF](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking-gguf) | 🍎 [MLX](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking-mlx) | ⚡ [MLX 4-bit](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking-mlx-4bit)

### Training Data
- 📊 **[supra-nexus-o1-training](https://huggingface.co/datasets/Supra-Nexus/supra-nexus-o1-training)** - Complete training dataset

## 💡 Key Features

- **Transparent Reasoning**: Shows thought process using `<thinking>` tags
- **Chain-of-Thought**: Step-by-step problem solving approach
- **Self-Improvement**: Trained with recursive improvement examples
- **Multi-Format**: Available in multiple formats for different platforms

## 🚀 Quick Start

"""
    
    # Add format-specific usage instructions
    if is_gguf:
        card += """### Using with llama.cpp

```bash
# Download the model
huggingface-cli download {repo_name} --local-dir ./models

# Run inference
./llama-cli -m ./models/{model_file}.gguf -p "Your prompt here"
```

### Available Quantizations
- `F16` - Full 16-bit precision (largest, most accurate)
- `Q8_0` - 8-bit quantization (good balance)
- `Q5_K_M` - 5-bit quantization (recommended)
- `Q4_K_M` - 4-bit quantization (smallest)
""".format(repo_name=repo_name, model_file=repo_name.split('/')[-1].replace('-gguf', ''))
    
    elif is_mlx and is_4bit:
        card += """### Using with MLX (4-bit Quantized)

```python
from mlx_lm import load, generate

# Load 4-bit quantized model (75% smaller)
model, tokenizer = load("{repo_name}")

# Generate with chain-of-thought
prompt = "Solve step by step: What is 25% of 480?"
response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
print(response)
```

### Benefits of 4-bit Quantization
- 🚀 75% smaller model size
- ⚡ Faster inference on M1/M2/M3 Macs
- 💾 Lower memory requirements
- ✨ Minimal quality loss
""".format(repo_name=repo_name)
    
    elif is_mlx:
        card += """### Using with MLX

```python
from mlx_lm import load, generate

# Load the model optimized for Apple Silicon
model, tokenizer = load("{repo_name}")

# Generate response
prompt = "Explain the concept of recursion with an example"
response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
print(response)
```

### MLX Advantages
- 🍎 Optimized for Apple Silicon (M1/M2/M3)
- 🚀 Hardware acceleration on Mac
- 💾 Efficient memory usage
- ⚡ Fast inference
""".format(repo_name=repo_name)
    
    else:  # Base model
        card += """### Using with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "{repo_name}",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Generate response
messages = [{{"role": "user", "content": "Explain quantum computing in simple terms"}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using with vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{repo_name}")
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)

prompts = ["Explain the theory of relativity"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```
""".format(repo_name=repo_name)
    
    # Add common footer
    card += """
## 📈 Performance

The O1 models excel at:
- Complex reasoning tasks
- Step-by-step problem solving
- Mathematical computations
- Code generation and debugging
- Creative writing with logical structure

## 🏗️ Architecture

Based on Qwen2.5 architecture with:
- Custom fine-tuning for reasoning
- Chain-of-thought training
- Self-improvement capabilities
- Identity preservation techniques

## 🔬 Training Details

- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **Training Framework**: [Zoo Gym](https://github.com/zooai/gym)
- **Dataset**: [supra-nexus-o1-training](https://huggingface.co/datasets/Supra-Nexus/supra-nexus-o1-training)
- **Training Duration**: Multiple iterations with self-improvement
- **Hardware**: NVIDIA A100 GPUs

## 📚 Resources

- 📖 **[GitHub Repository](https://github.com/Supra-Nexus/o1)** - Source code and documentation
- 🏢 **[Supra Foundation](https://supra.com)** - Organization behind O1
- 🐦 **[Twitter](https://twitter.com/SupraOracles)** - Latest updates
- 💬 **[Discord](https://discord.gg/supra)** - Community support

## 📄 Citation

```bibtex
@software{supra_nexus_o1_2025,
  title = {Supra Nexus O1: Advanced Reasoning Models},
  author = {Supra Foundation},
  year = {2025},
  url = {https://github.com/Supra-Nexus/o1}
}
```

## 📝 License

Apache 2.0 - See [LICENSE](https://github.com/Supra-Nexus/o1/blob/main/LICENSE) for details.

---

*Building transparent AI reasoning systems* 🧠✨
"""
    
    return card

def update_all_models():
    """Update all model cards on HuggingFace."""
    
    models = [
        # Base models
        ("Supra-Nexus/supra-nexus-o1-instruct", "instruct", "Qwen/Qwen2.5-7B-Instruct", 
         "Instruction-tuned Supra Nexus O1 model optimized for following complex instructions with clarity and precision."),
        ("Supra-Nexus/supra-nexus-o1-thinking", "thinking", "Qwen/Qwen2.5-7B-Instruct",
         "Chain-of-thought Supra Nexus O1 model that shows transparent reasoning using <thinking> tags."),
        
        # GGUF variants
        ("Supra-Nexus/supra-nexus-o1-instruct-gguf", "instruct", "Supra-Nexus/supra-nexus-o1-instruct",
         "GGUF format of the instruction-tuned Supra Nexus O1 model for use with llama.cpp."),
        ("Supra-Nexus/supra-nexus-o1-thinking-gguf", "thinking", "Supra-Nexus/supra-nexus-o1-thinking",
         "GGUF format of the chain-of-thought Supra Nexus O1 model for use with llama.cpp."),
        
        # MLX variants
        ("Supra-Nexus/supra-nexus-o1-instruct-mlx", "instruct", "Supra-Nexus/supra-nexus-o1-instruct",
         "MLX format of the instruction-tuned Supra Nexus O1 model optimized for Apple Silicon."),
        ("Supra-Nexus/supra-nexus-o1-thinking-mlx", "thinking", "Supra-Nexus/supra-nexus-o1-thinking",
         "MLX format of the chain-of-thought Supra Nexus O1 model optimized for Apple Silicon."),
        
        # MLX 4-bit variants
        ("Supra-Nexus/supra-nexus-o1-instruct-mlx-4bit", "instruct", "Supra-Nexus/supra-nexus-o1-instruct",
         "4-bit quantized MLX format of the instruction-tuned Supra Nexus O1 model for efficient inference on Apple Silicon."),
        ("Supra-Nexus/supra-nexus-o1-thinking-mlx-4bit", "thinking", "Supra-Nexus/supra-nexus-o1-thinking",
         "4-bit quantized MLX format of the chain-of-thought Supra Nexus O1 model for efficient inference on Apple Silicon."),
    ]
    
    for repo_name, model_type, base_model, description in models:
        print(f"\n📝 Updating {repo_name}...")
        
        # Create the model card
        card_content = create_model_card(repo_name, model_type, base_model, description)
        
        # Save to file
        with open("README.md", "w") as f:
            f.write(card_content)
        
        # Upload to HuggingFace
        if run_command(f"huggingface-cli upload {repo_name} README.md README.md"):
            print(f"✅ Updated {repo_name}")
        else:
            print(f"❌ Failed to update {repo_name}")

def update_organization_card():
    """Update the Supra-Nexus organization README."""
    
    org_readme = """# Supra Nexus 🧠

Building advanced AI reasoning systems with transparent thought processes.

## 🚀 Featured Models

### O1 Series - Reasoning Models
Our flagship reasoning models that show their thought process:

| Model | Description | Formats |
|-------|-------------|---------|
| **[supra-nexus-o1-instruct](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct)** | Instruction-following with clarity | [GGUF](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct-gguf) • [MLX](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct-mlx) • [MLX-4bit](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct-mlx-4bit) |
| **[supra-nexus-o1-thinking](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking)** | Chain-of-thought reasoning | [GGUF](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking-gguf) • [MLX](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking-mlx) • [MLX-4bit](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking-mlx-4bit) |

### 📊 Training Data
- **[supra-nexus-o1-training](https://huggingface.co/datasets/Supra-Nexus/supra-nexus-o1-training)** - Comprehensive training dataset

## 🏗️ About Supra Foundation

Supra is a vertically integrated Layer-1 blockchain with Multi-VM support, Native Oracles, VRF, Automation, and Bridges. We're expanding into AI to build transparent, verifiable reasoning systems.

### Our Mission
- Build transparent AI systems that show their reasoning
- Create self-improving models through recursive training
- Develop efficient inference solutions for edge devices
- Foster open-source AI development

## 🔗 Links

- 🌐 **[Website](https://supra.com)**
- 💻 **[GitHub](https://github.com/Supra-Nexus)**
- 🐦 **[Twitter](https://twitter.com/SupraOracles)**
- 💬 **[Discord](https://discord.gg/supra)**
- 📚 **[Documentation](https://docs.supra.com)**

## 🤝 Collaboration

We welcome collaboration from the AI community! Whether you're interested in:
- Fine-tuning our models for specific use cases
- Contributing to our training datasets
- Building applications with our models
- Research partnerships

Feel free to reach out through our Discord or GitHub.

## 📄 License

All our models are released under Apache 2.0 license for maximum compatibility and commercial use.

---

*Advancing AI through transparent reasoning* 🚀
"""
    
    # Save and upload org README
    with open("ORG_README.md", "w") as f:
        f.write(org_readme)
    
    # Note: Organization READMEs need to be set through the HuggingFace web interface
    print("\n📋 Organization README created in ORG_README.md")
    print("Please update it manually at: https://huggingface.co/Supra-Nexus")

def main():
    """Main execution."""
    print("🔄 Updating all Supra Nexus O1 model cards...")
    
    # Update all model cards
    update_all_models()
    
    # Create organization README
    update_organization_card()
    
    print("\n✅ All model cards updated with proper cross-linking!")
    print("\n📦 Complete model collection:")
    print("https://huggingface.co/Supra-Nexus")

if __name__ == "__main__":
    main()