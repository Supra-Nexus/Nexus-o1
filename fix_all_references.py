#!/usr/bin/env python3
"""Fix all 4B -> 4B references across the codebase."""

import os
import re
from pathlib import Path

def fix_file(file_path, replacements):
    """Apply replacements to a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    root = Path("/Users/z/work/supra/o1")
    
    # Define all replacements needed
    replacements = [
        # Model size references
        (r'4B(?:\s+parameter|\s+model|)', '4B'),
        (r'4-billion', '4-billion'),
        (r'4 billion', '4 billion'),
        (r'~2\.3GB', '~4.1GB'),
        (r'2\.3GB', '4.5GB'),
        (r'2\.3 GB', '4.1 GB'),
        
        # Specific model name references
        (r'Qwen3-4B', 'Qwen3-4B'),
        (r'qwen3-4B', 'qwen3-4b'),
        (r'QWEN3-4B', 'QWEN3-4B'),
        
        # Parameter counts
        (r'4,000,000,000', '4,000,000,000'),
        (r'2\.0B', '4.0B'),
        (r'4Bs', '4b parameters'),
        
        # Model descriptions
        (r'compact 4B', 'compact 4B'),
        (r'small 4B', 'small 4B'),
        (r'efficient 4B', 'efficient 4B'),
        
        # File size references  
        (r'quantized to ~2\.3', 'quantized to ~4.5'),
        (r'approximately 2\.3', 'approximately 4.1'),
        (r'around 2\.3', 'around 4.1'),
    ]
    
    # Process all text files
    extensions = ['.tex', '.py', '.md', '.json', '.yaml', '.yml', '.txt', '.rst']
    fixed_files = []
    
    for ext in extensions:
        for file_path in root.rglob(f'*{ext}'):
            # Skip hidden directories and files
            if any(part.startswith('.') for part in file_path.parts):
                continue
            
            # Skip __pycache__ and other build directories
            if any(part in ['__pycache__', 'build', 'dist', '.git'] for part in file_path.parts):
                continue
                
            if fix_file(file_path, replacements):
                fixed_files.append(file_path)
    
    # Report results
    print(f"Fixed {len(fixed_files)} files:")
    for f in sorted(fixed_files):
        print(f"  - {f.relative_to(root)}")
    
    # Verify critical files
    critical_files = [
        "paper/sections/abstract.tex",
        "paper/sections/introduction.tex",
        "paper/sections/methodology.tex",
        "paper/sections/results.tex",
        "README.md",
        "models/supra-nexus-o1-thinking/README.md",
        "models/supra-nexus-o1-instruct/README.md",
    ]
    
    print("\nVerifying critical files:")
    for cf in critical_files:
        full_path = root / cf
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
                if '4B' in content or '4.5GB' in content or '4 billion' in content:
                    print(f"  ⚠️  {cf} still contains 4B references")
                else:
                    print(f"  ✓ {cf} fixed")
        else:
            print(f"  ⚠️  {cf} not found")

if __name__ == "__main__":
    main()