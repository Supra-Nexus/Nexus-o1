#!/usr/bin/env python3
"""
Set up GitHub mirrors for Supra Nexus o1 models following Qwen's pattern.
Creates repos and pushes model files with proper README and structure.
"""

import subprocess
import os
from pathlib import Path

def create_model_readme(model_name: str, is_thinking: bool = False) -> str:
    """Create GitHub README for the model repository."""
    
    capability = "transparent chain-of-thought reasoning" if is_thinking else "direct instruction following"
    model_type = "Thinking" if is_thinking else "Instruct"
    
    return f"""# {model_name}

{model_type} variant of Supra Nexus o1, optimized for {capability} and transparent AI systems.

## 🏢 Supra Foundation LLC

Advanced reasoning models with explicit thinking processes, created by **Supra Foundation LLC** in 2025.

## 🚀 Quick Start

### Using Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/{model_name.lower()}")
tokenizer = AutoTokenizer.from_pretrained("zenlm/{model_name.lower()}")

prompt = "What is your approach to problem solving?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

### Using MLX (Apple Silicon)
```python
from mlx_lm import load, generate

model, tokenizer = load("zenlm/{model_name.lower()}")
response = generate(model, tokenizer, prompt="Hello!", max_tokens=100)
print(response)
```

## 🔗 Links

- **HuggingFace**: [zenlm/{model_name.lower()}](https://huggingface.co/zenlm/{model_name.lower()})
- **Dataset**: [zenlm/supra-identity](https://huggingface.co/datasets/zenlm/supra-identity)
- **Paper**: [Supra Nexus o1 Research](../paper/)

## 📊 Performance

- **Architecture**: Qwen3-4B-2507 with Supra fine-tuning
- **Parameters**: 4B (ultra-efficient)
- **Context**: 32K tokens
- **Specialization**: {capability}

## 🎯 Use Cases

{"### Transparent Reasoning" if is_thinking else "### Direct Efficiency"}
{"- Educational tools with step-by-step explanations" if is_thinking else "- Quick responses for production systems"}
{"- Debugging complex problems with visible logic" if is_thinking else "- Efficient task completion"}
{"- Research into AI decision-making processes" if is_thinking else "- Streamlined instruction following"}

## 🏗️ Model Details

- **Base Model**: Qwen3-4B-2507-{model_type}-2507
- **Training**: LoRA fine-tuning with MLX
- **Creator**: Supra Foundation LLC (2025)
- **License**: Apache 2.0

## 📄 Citation

```bibtex
@model{{{model_name.lower().replace('-', '')}2025,
  title={{{model_name}: Transparent AI Reasoning}},
  author={{Supra Foundation LLC}},
  year={{2025}},
  url={{https://github.com/zeekay/{model_name.lower()}}}
}}
```

## License

Apache 2.0 - Free for commercial and research use.

---

**🏢 Supra Foundation LLC** • Transparent AI reasoning • 2025"""

def setup_github_repo(model_path: str, repo_name: str, description: str) -> bool:
    """Set up GitHub repository for a model."""
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"🐙 Setting up GitHub repo: {repo_name}")
    
    try:
        # Create GitHub repository
        cmd = f"gh repo create {repo_name} --public --description '{description}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0 and "already exists" not in result.stderr:
            print(f"❌ Failed to create repo: {result.stderr}")
            return False
        
        # Initialize git in model directory
        os.chdir(model_path)
        
        # Create README
        is_thinking = "thinking" in repo_name.lower()
        readme_content = create_model_readme(repo_name.replace("-", " ").title(), is_thinking)
        with open("README.md", "w") as f:
            f.write(readme_content)
        
        # Git setup
        subprocess.run("git init", shell=True, check=True)
        subprocess.run("git branch -M main", shell=True, check=True)
        subprocess.run(f"git remote add origin https://github.com/zeekay/{repo_name}.git", shell=True, check=False)
        
        # Add all files
        subprocess.run("git add .", shell=True, check=True)
        subprocess.run('git commit -m "Initial Supra Nexus o1 model release - 2025"', shell=True, check=True)
        
        # Push to GitHub
        subprocess.run("git push -u origin main --force", shell=True, check=True)
        
        print(f"✅ {repo_name} mirrored to GitHub")
        print(f"🔗 https://github.com/zeekay/{repo_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup {repo_name}: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir("/Users/z/work/supra/o1")

def main():
    """Set up GitHub mirrors for all Supra models."""
    
    print("🐙 Setting up GitHub mirrors for Supra Nexus o1 (following Qwen pattern)")
    print("=" * 75)
    
    original_dir = os.getcwd()
    
    # Models to mirror
    models = [
        {
            "path": "models/supra-nexus-o1-thinking-fused",
            "repo": "supra-nexus-o1-thinking",
            "description": "Supra Nexus o1 Thinking - Transparent AI reasoning with explicit thought processes"
        },
        {
            "path": "models/supra-nexus-o1-instruct-fused", 
            "repo": "supra-nexus-o1-instruct",
            "description": "Supra Nexus o1 Instruct - Direct instruction following for efficient AI applications"
        }
    ]
    
    success_count = 0
    
    for model in models:
        print(f"\\n📤 {model['repo']}")
        print("-" * 40)
        
        if setup_github_repo(model["path"], model["repo"], model["description"]):
            success_count += 1
    
    # Ensure we're back in original directory
    os.chdir(original_dir)
    
    # Summary
    print("\\n" + "=" * 75)
    print("📊 GITHUB MIRROR SUMMARY")
    print("=" * 75)
    
    if success_count == len(models):
        print("🎉 Complete GitHub mirror setup successful!")
        print("✅ Both models now available on GitHub")
        print("✅ Follows Qwen's repository pattern")
        print("✅ Professional READMEs with usage examples")
        print("✅ Proper Supra Foundation LLC branding")
        
        print("\\n🔗 GitHub repositories:")
        print("• Thinking: https://github.com/zeekay/supra-nexus-o1-thinking")
        print("• Instruct: https://github.com/zeekay/supra-nexus-o1-instruct")
        
        print("\\n🔗 HuggingFace models:")
        print("• Thinking: https://huggingface.co/zenlm/supra-nexus-o1-thinking") 
        print("• Instruct: https://huggingface.co/zenlm/supra-nexus-o1-instruct")
        print("• Dataset: https://huggingface.co/datasets/zenlm/supra-identity")
        
        print("\\n⚡ Now mirrored on both platforms like Qwen!")
        
    else:
        print(f"⚠️  Partial setup: {success_count}/{len(models)} repositories created")
    
    print("=" * 75)

if __name__ == "__main__":
    main()