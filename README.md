---
license: apache-2.0
tags:
- supra-nexus
- mlx
- quantized
- 4-bit
---

# supra-nexus-o1-thinking - MLX 4-bit Quantized

Chain-of-thought Supra Nexus O1 model with 4-bit quantization for Apple Silicon.

## Benefits

- 75% smaller than full precision
- Faster inference on consumer hardware
- Minimal quality loss

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("Supra-Nexus/supra-nexus-o1-thinking-mlx-4bit")
response = generate(model, tokenizer, prompt="Your prompt here")
```

## Base Model

- [Supra-Nexus/supra-nexus-o1-thinking](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking)

## Training Data

- [Supra-Nexus/supra-nexus-o1-training](https://huggingface.co/datasets/Supra-Nexus/supra-nexus-o1-training)
