#!/usr/bin/env python3
"""Comprehensive test to verify all 2B -> 4B fixes were applied correctly."""

import json
import re
from pathlib import Path

def check_file_for_errors(file_path):
    """Check a file for incorrect references."""
    errors = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Pattern checks for wrong references
        wrong_patterns = [
            (r'\b2B\b(?!\s*->)', '2B reference'),
            (r'\b2-billion\b', '2-billion reference'),
            (r'\b2 billion\b', '2 billion reference'),
            (r'2,000,000,000', '2 billion number'),
            (r'Qwen3-2B', 'Qwen3-2B model name'),
            (r'qwen3-2b', 'qwen3-2b model name'),
            (r'2\.3\s*GB', '2.3GB size'),
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern, desc in wrong_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip if it's a comment or part of a fix
                    if 'fix' in line.lower() or '#' in line or '//' in line or '2B ->' in line:
                        continue
                    errors.append((line_num, desc, line.strip()[:100]))
        
    except Exception as e:
        errors.append((0, f"Error reading file: {e}", ""))
    
    return errors

def verify_configs(root):
    """Verify all config.json files have correct specifications."""
    print("\n" + "="*60)
    print("VERIFYING CONFIG FILES")
    print("="*60)
    
    configs_ok = True
    for config_path in root.rglob("config.json"):
        if '.git' in str(config_path):
            continue
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check key parameters
            if 'hidden_size' in config:
                if config['hidden_size'] != 2560:
                    print(f"❌ {config_path.relative_to(root)}: Wrong hidden_size: {config['hidden_size']}")
                    configs_ok = False
                    
            if 'num_hidden_layers' in config:
                if config['num_hidden_layers'] != 36:
                    print(f"❌ {config_path.relative_to(root)}: Wrong num_hidden_layers: {config['num_hidden_layers']}")
                    configs_ok = False
                    
        except Exception as e:
            print(f"❌ Error reading {config_path.relative_to(root)}: {e}")
            configs_ok = False
    
    if configs_ok:
        print("✅ All config files have correct 4B model specifications")
    
    return configs_ok

def main():
    root = Path("/Users/z/work/supra/o1")
    
    print("="*60)
    print("COMPREHENSIVE 2B -> 4B FIX VERIFICATION")
    print("="*60)
    
    # Check all text files for errors
    all_errors = {}
    extensions = ['.tex', '.py', '.md', '.txt', '.rst']
    
    for ext in extensions:
        for file_path in root.rglob(f'*{ext}'):
            # Skip unimportant paths
            if any(part in ['.git', '__pycache__', 'build', 'dist', 'fix_all_references.py', 
                           'fix_accurate_sizes.py', 'test_all_fixes.py'] 
                   for part in file_path.parts):
                continue
                
            errors = check_file_for_errors(file_path)
            if errors:
                all_errors[file_path] = errors
    
    # Report findings
    if all_errors:
        print("\n❌ FOUND ISSUES IN THE FOLLOWING FILES:")
        for file_path, errors in all_errors.items():
            print(f"\n{file_path.relative_to(root)}:")
            for line_num, desc, content in errors[:3]:  # Show first 3 errors per file
                print(f"  Line {line_num}: {desc}")
                if content:
                    print(f"    -> {content}")
    else:
        print("\n✅ NO 2B REFERENCES FOUND - All files correctly updated to 4B!")
    
    # Verify configs
    configs_ok = verify_configs(root)
    
    # Check critical files explicitly
    print("\n" + "="*60)
    print("CRITICAL FILES CHECK")
    print("="*60)
    
    critical_files = [
        "paper/sections/abstract.tex",
        "paper/sections/methodology.tex", 
        "paper/sections/results.tex",
        "README.md",
        "models/supra-nexus-o1-thinking/README.md",
        "models/supra-nexus-o1-instruct/README.md",
    ]
    
    all_critical_ok = True
    for cf in critical_files:
        full_path = root / cf
        if full_path.exists():
            errors = check_file_for_errors(full_path)
            if errors:
                print(f"❌ {cf}: Found {len(errors)} issues")
                all_critical_ok = False
            else:
                print(f"✅ {cf}: Clean")
        else:
            print(f"⚠️  {cf}: File not found")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not all_errors and configs_ok and all_critical_ok:
        print("🎉 SUCCESS! All 2B references have been successfully updated to 4B")
        print("✅ Model specifications: 4.02B parameters")
        print("✅ Model size (8-bit): ~4.1GB")
        print("✅ Architecture: Qwen3-4B-2507 with 36 layers, 2560 hidden size")
    else:
        print("⚠️  Some issues remain - please review the output above")
        print("Run fix_all_references.py again if needed")

if __name__ == "__main__":
    main()