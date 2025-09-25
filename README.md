# Supra Nexus O1

Advanced reasoning models with transparent thought processes.

<div align="center">

[🤗 Models](https://huggingface.co/Supra-Nexus) | [📚 Documentation](docs/) | [🎯 Examples](examples/) | [📄 Paper](paper/)

</div>

## Models

| Model | HuggingFace | Description |
|-------|-------------|-------------|
| supra-nexus-o1-instruct | [🤗 Hub](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct) | Direct instruction following |
| supra-nexus-o1-thinking | [🤗 Hub](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking) | Transparent reasoning |

## Installation

```bash
pip install -e .

# With MLX support (Apple Silicon)
pip install -e .[mlx]

# With Zoo Gym training
pip install -e .[gym]
```

## Quick Start

```python
from supra_nexus import SupraModel, generate

# Load model
model = SupraModel.from_pretrained("Supra-Nexus/supra-nexus-o1-instruct")

# Generate
response = generate(model, "Explain quantum computing")
print(response)
```

## Training with Zoo Gym

```python
from supra_nexus.integrations import GymTrainer

trainer = GymTrainer("Supra-Nexus/supra-nexus-o1-instruct")
trainer.train(config="configs/training.yaml")
```

## Examples

See [examples/](examples/) for:
- Training examples
- Inference examples
- Format conversion
- Fine-tuning

## Documentation

- [Getting Started](docs/guides/getting_started.md)
- [Training Guide](docs/guides/training.md)
- [API Reference](docs/api/)

## Citation

```bibtex
@misc{supranexus2025,
  title={Supra Nexus O1: Advanced Reasoning with Transparent Thought},
  author={Supra Foundation LLC},
  year={2025},
  url={https://github.com/Supra-Nexus/o1}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
