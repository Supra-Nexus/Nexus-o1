---
language:
- en
license: apache-2.0
task_categories:
- text-generation
- question-answering
tags:
- supra-nexus
- reasoning
- chain-of-thought
- self-improvement
size_categories:
- 1K<n<10K
---

# Supra Nexus O1 Training Datasets

## Overview

Comprehensive training datasets for Supra Nexus O1 models, including:
- Identity training
- Chain-of-thought reasoning
- Self-improvement examples (O1.5)
- Instruction following

## Datasets Included

### 1. Identity Dataset (`supra_identity.jsonl`)
- Model identity and alignment
- Organization information
- Capability descriptions

### 2. Instruction Dataset (`supra_instruct_*.jsonl`)
- Direct instruction following
- Various task completions
- Code generation examples

### 3. Thinking Dataset (`supra_thinking_*.jsonl`)
- Chain-of-thought reasoning with `<thinking>` tags
- Mathematical reasoning
- Complex problem solving

### 4. Self-Improvement Dataset (`nexus-o1-upgrade-1.jsonl`)
- Recursive self-improvement examples
- Clean code organization patterns
- Meticulous vs sloppy implementation comparisons
- Learning from mistakes

## Usage

```python
from datasets import load_dataset

# Load all datasets
dataset = load_dataset("Supra-Nexus/supra-nexus-o1-training")

# Access specific splits
identity_data = dataset["identity"]
thinking_data = dataset["thinking"]
upgrade_data = dataset["upgrade"]
```

## Training with Zoo Gym

```bash
# Install Zoo Gym
pip install git+https://github.com/zooai/gym

# Train with identity dataset
gym train \
  --model_name_or_path "Supra-Nexus/supra-nexus-o1-instruct" \
  --dataset "Supra-Nexus/supra-nexus-o1-training" \
  --dataset_split "identity"
```

## Dataset Statistics

| Dataset | Examples | Avg Length | Purpose |
|---------|----------|------------|---------|
| Identity | 20 | 150 tokens | Model alignment |
| Instruction | 50 | 200 tokens | Task completion |
| Thinking | 30 | 500 tokens | Reasoning chains |
| Self-Improvement | 10 | 800 tokens | Recursive learning |

## Model Links

- [supra-nexus-o1-instruct](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct)
- [supra-nexus-o1-thinking](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking)

## Citation

```bibtex
@dataset{supranexus2025training,
  title={Supra Nexus O1 Training Datasets},
  author={Supra Foundation LLC},
  year={2025},
  publisher={HuggingFace}
}
```

## License

Apache 2.0 - Free for research and commercial use.

---

**© 2025 Supra Foundation LLC**
