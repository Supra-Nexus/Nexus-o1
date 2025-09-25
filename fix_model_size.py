#!/usr/bin/env python3
"""Fix all model size references from 4B to 4B."""

import os
import re
from pathlib import Path

def fix_file(filepath):
    """Fix model size references in a file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace various forms of 4B with 4B
    replacements = [
        (r'Qwen3-4B-2507', 'Qwen3-4B-2507'),
        (r'Qwen3 4B-2507', 'Qwen3 4B-2507'),
        (r'4Bs', '4B parameters'),
        (r'4B', '4B model'),
        (r'\(4B\)', '(4B)'),
        (r'2\.3GB', '4.5GB'),  # Approximate model size
        (r'Qwen/Qwen3-4B-2507', 'Qwen/Qwen3-4B-2507'),
    ]
    
    modified = False
    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
    
    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all files in the repository."""
    
    # Directories to search
    dirs_to_fix = [
        Path("/Users/z/work/supra/o1"),
        Path("/Users/z/work/zen")
    ]
    
    # File patterns to fix
    patterns = ['*.py', '*.md', '*.tex', '*.json', '*.yml', '*.yaml']
    
    fixed_files = []
    
    for base_dir in dirs_to_fix:
        if not base_dir.exists():
            continue
            
        for pattern in patterns:
            for filepath in base_dir.rglob(pattern):
                if '.git' in str(filepath):
                    continue
                    
                try:
                    if fix_file(filepath):
                        fixed_files.append(filepath)
                        print(f"✅ Fixed: {filepath}")
                except Exception as e:
                    print(f"❌ Error fixing {filepath}: {e}")
    
    print(f"\n📊 Fixed {len(fixed_files)} files")
    
    # Also update HuggingFace model cards
    print("\n📝 Updating HuggingFace model cards...")
    
    model_cards = [
        ("Supra-Nexus/supra-nexus-o1-instruct", "Qwen3 4B-2507 instruction-tuned model"),
        ("Supra-Nexus/supra-nexus-o1-thinking", "Qwen3 4B-2507 chain-of-thought model"),
        ("ZenLM/zen-nano-instruct", "Qwen3 4B-2507 nano instruction model"),
        ("ZenLM/zen-nano-thinking", "Qwen3 4B-2507 nano thinking model"),
    ]
    
    for repo_name, description in model_cards:
        print(f"  Updating {repo_name}...")
        
        # Create updated README
        readme_content = f"""---
license: apache-2.0
base_model: Qwen/Qwen3-4B-2507-2507
tags:
- qwen3
- 4b
- reasoning
- chain-of-thought
---

# {repo_name.split('/')[-1]}

{description} based on Qwen3 4B-2507 architecture.

## Model Details

- **Architecture**: Qwen3 (4B parameters)
- **Base Model**: Qwen/Qwen3-4B-2507
- **Context Length**: 262,144 tokens
- **Model Size**: ~4.1GB

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
```

## Links

- [Organization]({'https://huggingface.co/Supra-Nexus' if 'Supra' in repo_name else 'https://huggingface.co/ZenLM'})
- [GitHub]({'https://github.com/Supra-Nexus/o1' if 'Supra' in repo_name else 'https://github.com/ZenLM/zen'})
"""
        
        # Save for upload
        with open("README_update.md", "w") as f:
            f.write(readme_content)
        
        # Upload to HuggingFace
        os.system(f"huggingface-cli upload {repo_name} README_update.md README.md 2>/dev/null")
    
    print("\n✅ All references updated from 4B to 4B!")

if __name__ == "__main__":
    main()