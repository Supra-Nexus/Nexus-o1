---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- supra
- identity
- reasoning
- chain-of-thought
- supra-foundation
size_categories:
- n<1K
---

# Supra Identity Dataset

This dataset contains identity and alignment conversations used to train the Supra Nexus o1 model family by Supra Foundation LLC.

## Dataset Description

The Supra Identity dataset consists of carefully curated conversations that establish the model's identity, capabilities, and transparent reasoning approach. It's designed to help AI models understand their purpose as transparent reasoning systems.

## Content

- **Total conversations**: 8 core identity examples
- **Format**: JSONL with messages array
- **Language**: English
- **Use case**: Model identity training and reasoning alignment

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("supra-foundation/supra-identity")
print(dataset["train"][0])
```

## Model Family

This dataset is used to train:

- [supra-nexus-o1-thinking](https://huggingface.co/supra-foundation/supra-nexus-o1-thinking) - Chain-of-thought reasoning
- [supra-nexus-o1-instruct](https://huggingface.co/supra-foundation/supra-nexus-o1-instruct) - Direct instruction following

## Key Features

### Transparent Reasoning
The dataset emphasizes the model's unique ability to show thinking processes with `<thinking>` tags, making AI reasoning transparent and educational.

### Identity Alignment
Conversations establish clear identity as Supra Nexus o1, created by Supra Foundation LLC, with a focus on interpretable AI.

### Mission Clarity
Reinforces the foundation's mission of advancing transparent AI reasoning for educational and practical benefit.

## Citation

```bibtex
@dataset{supraidentity2025,
  title={Supra Identity Dataset: Training Data for Transparent AI Reasoning},
  author={Supra Foundation LLC},
  year={2025},
  url={https://huggingface.co/datasets/supra-foundation/supra-identity}
}
```

## License

Apache 2.0 - Free for commercial and research use.

---

**Built by Supra Foundation LLC** • Advancing transparent AI reasoning • 2025