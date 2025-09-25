---
license: apache-2.0
tags:
- supra-nexus
- o1
- reasoning
- chain-of-thought
- mlx
- apple-silicon
- quantized
- 4-bit
language:
- en
base_model: Supra-Nexus/supra-nexus-o1-thinking
---

# supra-nexus-o1-thinking-mlx-4bit

4-bit quantized MLX format of the chain-of-thought Supra Nexus O1 model for efficient inference on Apple Silicon.

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

### Using with MLX (4-bit Quantized)

```python
from mlx_lm import load, generate

# Load 4-bit quantized model (75% smaller)
model, tokenizer = load("Supra-Nexus/supra-nexus-o1-thinking-mlx-4bit")

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
