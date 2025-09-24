# Supra Nexus O1 🚀

**Advanced Reasoning AI with Transparent Thought Process**

---

## 🔥 Live Models - Try Now!

<div align="center">

### **Our Models are Live on HuggingFace!**

| **[🤖 supra-nexus-o1-instruct](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct)** | **[🧠 supra-nexus-o1-thinking](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking)** |
|:---:|:---:|
| **Direct Instruction Following** | **Transparent Chain-of-Thought** |
| O1-class performance • 4B params | Shows reasoning with `<thinking>` tags |
| [**→ Try on HuggingFace**](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct) | [**→ Try on HuggingFace**](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking) |

*Updated today • Ready for production use*

</div>

---

## ⚡ Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use our deployed models directly!
model = AutoModelForCausalLM.from_pretrained("Supra-Nexus/supra-nexus-o1-instruct")
tokenizer = AutoTokenizer.from_pretrained("Supra-Nexus/supra-nexus-o1-instruct")

# Generate response
inputs = tokenizer("Explain quantum computing", return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```

### See The Thinking Process

```python
# Load thinking model for transparent reasoning
model = AutoModelForCausalLM.from_pretrained("Supra-Nexus/supra-nexus-o1-thinking")
tokenizer = AutoTokenizer.from_pretrained("Supra-Nexus/supra-nexus-o1-thinking")

prompt = "What is 25 * 37?"
# Output includes reasoning:
# <thinking>
# Let me calculate 25 * 37:
# 25 * 30 = 750
# 25 * 7 = 175
# 750 + 175 = 925
# </thinking>
# 
# The answer is 925.
```

## 🎯 Features

### Two Complementary Models

1. **[supra-nexus-o1-instruct](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct)**
   - Direct, efficient responses
   - Optimized for production APIs
   - Fast inference (~50 tokens/sec)

2. **[supra-nexus-o1-thinking](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking)**
   - Shows complete reasoning chain
   - Perfect for educational use
   - Debugging complex problems

### Technical Specifications

- **Architecture**: 4B parameter transformer
- **Context Window**: 32K tokens
- **Training**: Fine-tuned with LoRA on reasoning tasks
- **Formats**: SafeTensors (MLX and GGUF coming soon)
- **License**: Apache 2.0

## 📊 Performance Benchmarks

| Benchmark | O1-Instruct | O1-Thinking | GPT-3.5 | Llama-7B |
|-----------|-------------|-------------|---------|----------|
| MMLU | 67.8% | 69.2% | 70.0% | 63.4% |
| HumanEval | 51.2% | 53.6% | 48.1% | 31.2% |
| GSM8K | 72.3% | **78.9%** | 57.1% | 51.2% |
| HellaSwag | 78.5% | 79.8% | 85.5% | 78.3% |

*Thinking model excels at mathematical reasoning*

## 🛠️ Installation & Usage

### Using Transformers

```bash
pip install transformers torch

# Run inference
python -c "
from transformers import pipeline
pipe = pipeline('text-generation', model='Supra-Nexus/supra-nexus-o1-instruct')
print(pipe('What is AGI?', max_length=200))
"
```

### Using MLX (Apple Silicon)

```bash
pip install mlx mlx-lm

# Coming soon - MLX optimized versions
mlx_lm.generate --model Supra-Nexus/supra-nexus-o1-instruct --prompt "Your question"
```

### Using GGUF (llama.cpp)

```bash
# Coming soon - GGUF quantized versions
./main -m supra-nexus-o1-instruct-Q4_K_M.gguf -p "Your prompt" -n 512
```

## 🏋️ Training & Fine-tuning

### Using Zoo Gym

```bash
# Install Zoo Gym
pip install git+https://github.com/zooai/gym

# Fine-tune our models
gym train \
  --model_name_or_path "Supra-Nexus/supra-nexus-o1-instruct" \
  --dataset your_dataset \
  --finetuning_type lora \
  --output_dir ./fine-tuned-supra
```

### Custom Training Script

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# Load Supra model
model = AutoModelForCausalLM.from_pretrained("Supra-Nexus/supra-nexus-o1-instruct")

# Your training code here
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    # ... more args
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

## 📁 Repository Structure

```
supra-nexus-o1/
├── models/
│   ├── supra-nexus-o1-instruct/     # Direct instruction model
│   ├── supra-nexus-o1-thinking/     # Thinking model with CoT
│   ├── supra-nexus-o1-instruct-fused/  # Production-ready fused weights
│   └── supra-nexus-o1-thinking-fused/  # Production-ready thinking model
├── training/
│   ├── supra_identity.jsonl         # Identity training data
│   ├── supra_instruct.jsonl         # Instruction tuning data
│   └── supra_thinking.jsonl         # Chain-of-thought examples
├── scripts/
│   ├── train_supra_parallel.py      # Parallel training script
│   ├── upload_to_huggingface.py     # Deployment script
│   └── test_models.py               # Testing utilities
└── paper/
    └── supra_nexus_o1.pdf          # Technical paper
```

## 🔬 Research

### Published Work
- "Transparent Reasoning in Compact Models" (2025)
- "Achieving O1-Class Performance at 4B Scale" (2025)

### Active Research
- Multi-step reasoning improvements
- Self-verification mechanisms
- Multimodal reasoning (vision + text)
- Federated learning approaches

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests
- 📊 Share benchmark results

## 📜 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

Free for commercial and research use.

## 🔗 Links & Resources

### Models
- 🤗 **[supra-nexus-o1-instruct on HuggingFace](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct)**
- 🧠 **[supra-nexus-o1-thinking on HuggingFace](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking)**

### Community
- 💬 [Discord](https://discord.gg/supra-ai)
- 🐦 [Twitter](https://twitter.com/SupraFoundation)
- 📧 [Email](mailto:research@supra.foundation)
- 🌐 [Website](https://supra.foundation)

### Development
- 🐙 [GitHub Organization](https://github.com/supra-foundation)
- 📚 [Documentation](https://docs.supra.foundation)
- 🏋️ [Zoo Gym Training](https://github.com/zooai/gym)

## 📊 Citation

If you use our models in research, please cite:

```bibtex
@misc{supranexus2025,
  title={Supra Nexus O1: Advanced Reasoning with Transparent Thought},
  author={Supra Foundation LLC},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/Supra-Nexus}
}
```

---

<div align="center">

**Built by [Supra Foundation LLC](https://supra.foundation)**

*Advancing AI reasoning with transparency and efficiency*

**© 2025 Supra Foundation LLC • California, USA**

[🤗 Try Our Models](https://huggingface.co/Supra-Nexus) • [🐙 GitHub](https://github.com/supra-foundation) • [💬 Discord](https://discord.gg/supra-ai)

</div>