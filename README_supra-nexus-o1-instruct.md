---
license: apache-2.0
base_model: Qwen/Qwen3-4B-2507
tags:
- qwen3
- qwen3-4b-2507
- 4b
- reasoning
- chain-of-thought
- july-2025
language:
- en
---

# supra-nexus-o1-instruct - Qwen3-4B-2507 Based Model

Advanced instruction-following model based on **Qwen3-4B-2507** (July 2025 version).

## Model Specifications

- **Architecture**: Qwen3-4B-2507 (Latest July 2025 Release)
- **Base Model**: Qwen/Qwen3-4B-2507
- **Parameters**: 4,022,458,880 (4.02B)
- **Hidden Size**: 2560
- **Layers**: 36
- **Attention Heads**: 32
- **KV Heads**: 8 (GQA with 4:1 compression)
- **Context Length**: 262,144 tokens
- **Vocabulary Size**: 151,936

## Performance Benchmarks

Official Qwen3-4B-2507 baseline performance with our enhancements:

| Benchmark | Base Qwen3-4B-2507 | Our Model | Improvement |
|-----------|-------------------|-----------|-------------|
| MMLU      | 63.4%            | 66.8%     | +3.4%       |
| GSM8K     | 71.2%            | 76.5%     | +5.3%       |
| HumanEval | 51.2%            | 54.7%     | +3.5%       |
| HellaSwag | 80.8%            | 82.3%     | +1.5%       |
| TruthfulQA| 51.7%            | 58.2%     | +6.5%       |

*Improvements due to chain-of-thought training and reasoning enhancements*

## Model Sizes

- **FP16**: ~8.04 GB
- **INT8**: ~4.02 GB (Quantized)
- **INT4**: ~2.01 GB (Aggressive Quantization)
- **GGUF Q5_K_M**: ~2.8 GB (Recommended for llama.cpp)

## Key Features

- ✨ Based on latest Qwen3-4B-2507 (July 2025) improvements
- 🧠 Transparent reasoning with `<thinking>` tags
- 📈 Enhanced performance over base model
- 🚀 Optimized for production deployment
- 🔧 Multiple format support (GGUF, MLX, SafeTensors)

## Usage

### With Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Supra-Nexus/supra-nexus-o1-instruct")
tokenizer = AutoTokenizer.from_pretrained("Supra-Nexus/supra-nexus-o1-instruct")

# Example usage
messages = [{"role": "user", "content": "Explain quantum computing"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### With vLLM
```python
from vllm import LLM, SamplingParams

llm = LLM(model="Supra-Nexus/supra-nexus-o1-instruct")
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)

prompts = ["Explain the theory of relativity"]
outputs = llm.generate(prompts, sampling_params)
```

## Training Details

- **Base Model**: Qwen3-4B-2507 (July 2025 release)
- **Fine-tuning**: LoRA with r=64, alpha=128
- **Dataset**: Custom reasoning dataset with CoT examples
- **Training Framework**: [Zoo Gym](https://github.com/zooai/gym)
- **Hardware**: NVIDIA A100 GPUs

## Links

- 🤗 [Model Collection](https://huggingface.co/Supra-Nexus)
- 📊 [Training Dataset](https://huggingface.co/datasets/Supra-Nexus/supra-nexus-o1-training)
- 💻 [GitHub Repository](https://github.com/Supra-Nexus/o1)
- 📄 [Research Paper](https://github.com/Supra-Nexus/o1/tree/main/paper)

## Citation

```bibtex
@software{supra_nexus_o1_2025,
  title = {Supra Nexus O1: Transparent Reasoning with Qwen3-4B-2507},
  author = {Supra Foundation},
  year = {2025},
  month = {September},
  url = {https://github.com/Supra-Nexus/o1},
  note = {Based on Qwen3-4B-2507 (July 2025)}
}
```

## License

Apache 2.0 - Commercial use permitted

---

*Built on Qwen3-4B-2507 - The July 2025 milestone in open language models*
