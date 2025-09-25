#!/usr/bin/env python3
"""
Reorganize Supra-Nexus O1 repository to match Qwen3's clean structure.
This creates a professional, maintainable layout.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

class RepositoryReorganizer:
    """Reorganize repository to match Qwen3's clean pattern"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.moves = []
        self.created_dirs = []
        
    def create_structure(self):
        """Create Qwen3-style directory structure"""
        
        # Main package directory (like qwen_agent/)
        dirs = [
            "supra_nexus/",
            "supra_nexus/models/",
            "supra_nexus/training/",
            "supra_nexus/inference/",
            "supra_nexus/utils/",
            "supra_nexus/integrations/",
            
            # Examples directory
            "examples/",
            "examples/training/",
            "examples/inference/",
            "examples/conversion/",
            
            # Documentation
            "docs/",
            "docs/guides/",
            "docs/api/",
            
            # Scripts (minimal, most logic in package)
            "scripts/",
            
            # Tests
            "tests/",
            "tests/models/",
            "tests/training/",
            
            # Data and configs
            "data/",
            "configs/",
            
            # Keep existing important dirs
            "paper/",
            "training/",  # training data
            "models/",    # model weights
        ]
        
        for dir_path in dirs:
            full_path = self.repo_path / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                self.created_dirs.append(str(dir_path))
                
    def plan_moves(self) -> Dict[str, str]:
        """Plan file moves to clean up root directory"""
        
        moves = {
            # Training scripts -> package modules
            "train_instruct_job.py": "supra_nexus/training/train_instruct.py",
            "train_thinking_job.py": "supra_nexus/training/train_thinking.py",
            "train_supra_parallel.py": "supra_nexus/training/train_parallel.py",
            "train_sequential.py": "supra_nexus/training/train_sequential.py",
            "train_with_mlx.py": "supra_nexus/training/train_mlx.py",
            "train_supra_thinking.py": "supra_nexus/training/train_thinking_model.py",
            "simple_train.py": "examples/training/simple_train.py",
            "train_and_test_supra.py": "examples/training/train_and_test.py",
            
            # Deployment/upload scripts -> scripts/
            "upload_to_huggingface.py": "scripts/upload_huggingface.py",
            "deploy_supra_nexus.py": "scripts/deploy.py",
            "deploy_supra_to_zenlm.py": "scripts/deploy_zenlm.py",
            "comprehensive_supra_upload.py": "scripts/upload_comprehensive.py",
            "setup_github_mirror.py": "scripts/setup_mirror.py",
            
            # Dataset generation -> package
            "generate_supra_datasets.py": "supra_nexus/data/dataset_generator.py",
            "convert_dataset.py": "supra_nexus/data/dataset_converter.py",
            "supra_identity_dataset.py": "supra_nexus/data/identity_dataset.py",
            
            # Test files -> tests/
            "test_training.py": "tests/training/test_training.py",
            
            # Data files -> data/
            "supra_identity_data.jsonl": "data/supra_identity.jsonl",
            
            # Documentation -> docs/
            "supra_dataset_README.md": "docs/datasets.md",
            "README_ORG.md": "docs/organization.md",
            "SUPRA_ORG_README.md": "docs/supra_org.md",
            
            # Utils -> package
            "update_supra_org_profile.py": "supra_nexus/utils/update_profile.py",
        }
        
        return moves
        
    def create_package_files(self):
        """Create proper Python package files"""
        
        # Main package __init__.py
        init_content = '''"""
Supra Nexus O1 - Advanced reasoning models with transparent thought
"""

__version__ = "1.0.0"
__author__ = "Supra Foundation LLC"

from supra_nexus.models import SupraModel
from supra_nexus.training import Trainer
from supra_nexus.inference import generate

__all__ = ["SupraModel", "Trainer", "generate"]
'''
        
        (self.repo_path / "supra_nexus" / "__init__.py").write_text(init_content)
        
        # setup.py (like Qwen3)
        setup_content = '''from setuptools import setup, find_packages

setup(
    name="supra-nexus",
    version="1.0.0",
    author="Supra Foundation LLC",
    description="Advanced reasoning models with transparent thought",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Supra-Nexus/o1",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "pyyaml",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff"],
        "mlx": ["mlx", "mlx-lm"],
        "gym": ["zoo-gym"],
    },
)
'''
        
        (self.repo_path / "setup.py").write_text(setup_content)
        
        # pyproject.toml (modern Python packaging)
        pyproject_content = '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "supra-nexus"
version = "1.0.0"
description = "Advanced reasoning models with transparent thought"
authors = [{name = "Supra Foundation LLC"}]
license = {text = "Apache-2.0"}
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "pyyaml",
    "tqdm",
]

[project.optional-dependencies]
dev = ["pytest", "black", "ruff", "mypy"]
mlx = ["mlx", "mlx-lm"]
gym = ["zoo-gym @ git+https://github.com/zooai/gym"]

[tool.ruff]
line-length = 100
target-version = "py38"

[tool.black]
line-length = 100
target-version = ["py38"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
'''
        
        (self.repo_path / "pyproject.toml").write_text(pyproject_content)
        
        # Create module __init__ files
        modules = [
            "supra_nexus/models",
            "supra_nexus/training", 
            "supra_nexus/inference",
            "supra_nexus/utils",
            "supra_nexus/integrations",
            "supra_nexus/data",
        ]
        
        for module in modules:
            init_path = self.repo_path / module / "__init__.py"
            if not init_path.exists():
                init_path.write_text(f'"""Module: {module.split("/")[-1]}"""')
                
    def create_clean_readme(self):
        """Create a clean, focused README like Qwen3"""
        
        readme = '''# Supra Nexus O1

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
'''
        
        (self.repo_path / "README.md").write_text(readme)
        
    def execute_moves(self, moves: Dict[str, str], dry_run: bool = True):
        """Execute the file moves"""
        
        for src, dst in moves.items():
            src_path = self.repo_path / src
            dst_path = self.repo_path / dst
            
            if src_path.exists():
                if dry_run:
                    print(f"Would move: {src} -> {dst}")
                    self.moves.append((src, dst))
                else:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_path), str(dst_path))
                    print(f"Moved: {src} -> {dst}")
                    self.moves.append((src, dst))
                    
    def cleanup_root(self):
        """Identify files that should remain in root (like Qwen3)"""
        
        # Files that should stay in root (matching Qwen3)
        keep_in_root = {
            "README.md",
            "LICENSE",
            "setup.py",
            "pyproject.toml",
            ".gitignore",
            "Makefile",
            ".github",  # GitHub workflows
        }
        
        # List all Python files in root
        root_files = [f for f in os.listdir(self.repo_path) 
                     if f.endswith('.py') and f != "setup.py"]
        
        if root_files:
            print(f"\n⚠️  Python files still in root: {root_files}")
            print("These should be moved to appropriate directories")
            
    def generate_report(self):
        """Generate reorganization report"""
        
        report = f"""
# Repository Reorganization Report

## Created Directories ({len(self.created_dirs)})
{chr(10).join('- ' + d for d in self.created_dirs)}

## Planned Moves ({len(self.moves)})
{chr(10).join(f'- {src} -> {dst}' for src, dst in self.moves)}

## New Structure (Qwen3-style)
```
supra-nexus-o1/
├── supra_nexus/          # Main package (like qwen_agent/)
│   ├── __init__.py
│   ├── models/          # Model implementations
│   ├── training/        # Training modules
│   ├── inference/       # Inference utilities
│   ├── utils/          # Helper utilities
│   └── integrations/   # External tool integrations
├── examples/           # Usage examples
├── scripts/           # Standalone scripts
├── tests/            # Test suite
├── docs/             # Documentation
├── paper/            # Academic paper
├── models/           # Model weights
├── training/         # Training data
├── data/            # Datasets
├── configs/         # Configuration files
├── README.md        # Clean, focused readme
├── setup.py         # Package setup
├── pyproject.toml   # Modern Python config
└── LICENSE         # Apache 2.0
```

## Benefits
- ✅ Clean root directory (like Qwen3)
- ✅ Proper Python package structure
- ✅ Clear separation of concerns
- ✅ Easy to navigate and understand
- ✅ Professional appearance
- ✅ Maintainable and scalable
"""
        
        return report
        
    def run(self, dry_run: bool = True):
        """Execute the reorganization"""
        
        print(f"{'DRY RUN' if dry_run else 'EXECUTING'} Reorganization")
        print("=" * 50)
        
        # Create structure
        print("\n1. Creating directory structure...")
        self.create_structure()
        
        # Plan moves
        print("\n2. Planning file moves...")
        moves = self.plan_moves()
        
        # Execute moves
        print("\n3. Moving files...")
        self.execute_moves(moves, dry_run)
        
        # Create package files
        if not dry_run:
            print("\n4. Creating package files...")
            self.create_package_files()
            self.create_clean_readme()
        
        # Check root cleanup
        print("\n5. Checking root directory...")
        self.cleanup_root()
        
        # Generate report
        print("\n6. Generating report...")
        report = self.generate_report()
        print(report)
        
        if dry_run:
            print("\n✅ Dry run complete. Run with --execute to apply changes.")
        else:
            print("\n✅ Reorganization complete!")
            

if __name__ == "__main__":
    import sys
    
    reorganizer = RepositoryReorganizer("/Users/z/work/supra/o1")
    execute = "--execute" in sys.argv
    
    reorganizer.run(dry_run=not execute)