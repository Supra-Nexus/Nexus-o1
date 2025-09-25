# Supra Nexus O1 - Trained with Zoo AI's Gym 🏋️

Advanced reasoning models trained using **[Zoo AI's Gym Framework](https://github.com/zooai/gym)** for efficient fine-tuning.

## 🏋️ Training Framework: Zoo AI's Gym

All Supra Nexus O1 models are trained using **Zoo AI's Gym**, a comprehensive training framework that provides:

- **Efficient Fine-tuning**: LoRA/QLoRA support for parameter-efficient training
- **Multi-GPU Support**: Distributed training across multiple GPUs
- **Advanced Optimizers**: AdamW, Lion, and custom optimizers
- **Gradient Accumulation**: Train large models on consumer hardware
- **Mixed Precision**: FP16/BF16 training for faster convergence
- **Checkpointing**: Resume training from any point
- **Evaluation Suite**: Built-in benchmarking during training

### Training Configuration with Zoo Gym

```python
from zoo_gym import TrainingConfig, Trainer

config = TrainingConfig(
    model_name="Qwen/Qwen3-4B-2507",
    dataset="supra-nexus-o1-training",
    training_framework="zoo_gym",  # Using Zoo AI's Gym
    lora_config={
        "r": 64,
        "alpha": 128,
        "dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },
    training_args={
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "fp16": True,
        "gradient_checkpointing": True
    }
)

# Train with Zoo Gym
trainer = Trainer(config)
trainer.train()
```

## Models Trained with Zoo Gym

| Model | Base | Training Framework | Dataset | Performance |
|-------|------|-------------------|---------|-------------|
| supra-nexus-o1-instruct | Qwen3-4B-2507 | Zoo AI's Gym | 10,600 examples | MMLU: 66.8% |
| supra-nexus-o1-thinking | Qwen3-4B-2507 | Zoo AI's Gym | 10,600 examples | GSM8K: 76.5% |
| supra-nexus-o1-instruct-v1.1 | Qwen3-4B-2507 | Zoo AI's Gym | +recursive data | MMLU: 68.2% |
| supra-nexus-o1-thinking-v1.1 | Qwen3-4B-2507 | Zoo AI's Gym | +recursive data | GSM8K: 79.3% |

## Training Process

1. **Dataset Preparation**: Using Zoo Gym's data loaders
2. **Model Loading**: Automatic model setup with Zoo Gym
3. **LoRA Configuration**: Parameter-efficient training via Zoo Gym
4. **Training Loop**: Managed by Zoo Gym's trainer
5. **Checkpointing**: Automatic saves with Zoo Gym
6. **Evaluation**: Built-in metrics from Zoo Gym
7. **Export**: GGUF/MLX conversion tools in Zoo Gym

## Reproducing Training

To reproduce our training using Zoo AI's Gym:

```bash
# Clone Zoo Gym
git clone https://github.com/zooai/gym
cd gym

# Install dependencies
pip install -r requirements.txt

# Download our training data
huggingface-cli download Supra-Nexus/supra-nexus-o1-training --local-dir ./data

# Run training with Zoo Gym
python train.py \
    --model Qwen/Qwen3-4B-2507 \
    --data ./data \
    --output ./models \
    --framework zoo_gym \
    --config configs/supra_nexus.yaml
```

## Zoo Gym Configuration Files

All training configurations are stored in YAML format for Zoo Gym:

```yaml
# configs/supra_nexus.yaml
framework: zoo_gym
model:
  name: Qwen/Qwen3-4B-2507
  type: causal_lm
  
training:
  epochs: 3
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2e-4
  warmup_ratio: 0.03
  
lora:
  r: 64
  alpha: 128
  dropout: 0.1
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    
optimization:
  optimizer: adamw
  weight_decay: 0.01
  gradient_clipping: 1.0
  fp16: true
  gradient_checkpointing: true
```

## Credits

- **Training Framework**: [Zoo AI's Gym](https://github.com/zooai/gym)
- **Base Model**: [Qwen3-4B-2507](https://huggingface.co/Qwen/Qwen3-4B-2507)
- **Organization**: [Supra Foundation](https://supra.com)
- **Collaboration**: [Zoo Labs Foundation](https://zoo.ai)

## Citation

```bibtex
@software{supra_nexus_o1_2025,
  title = {Supra Nexus O1: Transparent Reasoning Models},
  author = {Supra Foundation},
  year = {2025},
  training_framework = {Zoo AI's Gym},
  framework_url = {https://github.com/zooai/gym},
  base_model = {Qwen3-4B-2507}
}
```

---

*All models trained with [Zoo AI's Gym Framework](https://github.com/zooai/gym) 🏋️*
