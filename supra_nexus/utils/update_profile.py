#!/usr/bin/env python3
"""
Update Supra-Nexus HuggingFace Organization Profile
"""

import subprocess
from pathlib import Path

def create_org_readme():
    """Create organization README with new branding"""
    
    content = """# SUPRA - Intelligence Unchained

> **Signal beyond noise** • **Substrate Upgrade Protocol for Recursive AGI**

## 🧠 About SUPRA

A decentralized, evolving intelligence ecosystem built to empower humanity.

SUPRA combines AI, blockchain, and decentralized governance to create **Synthetic Ultra-Intelligence** — intelligence that is not controlled by corporations or states, but shared, collaborative, and self-evolving.

## 🚀 Our Models

### [supra-nexus-o1-thinking](https://huggingface.co/Supra-Nexus/supra-nexus-o1-thinking)
Advanced chain-of-thought reasoning with explicit thinking process visibility.

### [supra-nexus-o1-instruct](https://huggingface.co/Supra-Nexus/supra-nexus-o1-instruct)
Direct instruction-following model optimized for efficiency.

## 🏛️ Core Components

- **Substrate Neural Core**: The "digital brain" uniting AI agents & datasets
- **Open-CorteX**: Decentralized AI/data marketplace
- **AI Virtual Machine (AIVM)**: On-chain execution of AI models
- **Blockchain Integration**: Cross-chain AI collaboration
- **Security & Governance**: Ethical, transparent, community-led

## 💎 $SUPA Token

The native token powering the SUPRA intelligence network:
- **Governs**: Community-driven decisions
- **Incentivizes**: Rewards contributions
- **Aligns**: Economic and ethical incentives

## 📚 Learn More

- **Website**: [supra.foundation](https://supra.foundation)
- **GitHub**: [github.com/supra-foundation](https://github.com/supra-foundation)
- **X**: [@SupraFoundation](https://x.com/SupraFoundation)
- **Discord**: [Join our community](https://discord.gg/supra)

---

**© 2025 SUPRA Foundation LLC** • **Intelligence Unchained**
"""
    return content

def update_model_cards():
    """Update model cards with new branding"""
    
    models = [
        "supra-nexus-o1-thinking",
        "supra-nexus-o1-instruct"
    ]
    
    for model in models:
        print(f"Updating {model} model card...")
        
        model_type = "thinking" if "thinking" in model else "instruct"
        capability = "transparent chain-of-thought reasoning" if model_type == "thinking" else "efficient instruction following"
        
        card = f"""---
license: apache-2.0
language:
- en
tags:
- supra
- recursive-agi
- substrate
- reasoning
- {model_type}
pipeline_tag: text-generation
---

# {model.replace('-', ' ').title()}

> **Signal beyond noise** • Part of the SUPRA Intelligence Ecosystem

Advanced reasoning model with {capability} by **SUPRA Foundation LLC**.

## 🧠 SUPRA - Intelligence Unchained

**Substrate Upgrade Protocol for Recursive AGI**

A decentralized, evolving intelligence ecosystem built to empower humanity. SUPRA combines AI, blockchain, and decentralized governance to create Synthetic Ultra-Intelligence — intelligence that is not controlled by corporations or states, but shared, collaborative, and self-evolving.

## Model Details

- **Architecture**: 4B parameters based on Qwen3
- **Training**: Fine-tuned with LoRA for specialized reasoning
- **Context**: 32K token context window
- **License**: Apache 2.0
- **Organization**: SUPRA Foundation LLC
- **Year**: 2025

## Available Formats

### 🍎 MLX (Apple Silicon Optimized)
```bash
pip install mlx mlx-lm
from mlx_lm import load, generate
model, tokenizer = load("Supra-Nexus/{model}")
```

### 🖥️ GGUF (llama.cpp)
```bash
# Download GGUF version
huggingface-cli download Supra-Nexus/{model} --include "*.gguf"
```

### 🐍 Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Supra-Nexus/{model}")
tokenizer = AutoTokenizer.from_pretrained("Supra-Nexus/{model}")
```

## SUPRA Ecosystem

- **Substrate Neural Core**: Digital brain uniting AI agents
- **Open-CorteX**: Decentralized AI marketplace
- **AIVM**: On-chain AI execution
- **$SUPA Token**: Governance and incentive alignment

## Citation

```bibtex
@misc{{supra2025nexus,
  title={{SUPRA Nexus O1: Intelligence Unchained}},
  author={{SUPRA Foundation LLC}},
  year={{2025}},
  url={{https://supra.foundation}},
  note={{Substrate Upgrade Protocol for Recursive AGI}}
}}
```

## Links

- **Website**: [supra.foundation](https://supra.foundation)
- **GitHub**: [github.com/supra-foundation](https://github.com/supra-foundation)
- **HuggingFace**: [Supra-Nexus](https://huggingface.co/Supra-Nexus)
- **X**: [@SupraFoundation](https://x.com/SupraFoundation)

---

**© 2025 SUPRA Foundation LLC** • **Intelligence Unchained** • **RSVP to the future**
"""
        
        # Save card locally
        card_path = Path(f"README_{model}.md")
        with open(card_path, "w") as f:
            f.write(card)
        
        # Upload to HuggingFace
        cmd = ["hf", "upload", f"Supra-Nexus/{model}", str(card_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Updated {model}")
        else:
            print(f"❌ Failed to update {model}: {result.stderr}")
        
        # Clean up
        card_path.unlink()

def main():
    print("🚀 Updating SUPRA Organization Branding")
    print("="*50)
    
    # Create org README
    org_readme = create_org_readme()
    with open("README_ORG.md", "w") as f:
        f.write(org_readme)
    print("✅ Created organization README")
    
    # Update model cards
    update_model_cards()
    
    print("="*50)
    print("🎉 SUPRA branding update complete!")
    print("\nOrganization: https://huggingface.co/Supra-Nexus")
    print("Website: https://supra.foundation")

if __name__ == "__main__":
    main()