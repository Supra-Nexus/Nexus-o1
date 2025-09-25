#!/usr/bin/env python3
"""Create GGUF and MLX formats for Supra Nexus O1 models."""

import os
import subprocess
import sys
from pathlib import Path
import shutil

def run_command(cmd, cwd=None):
    """Execute command and handle errors."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Exception: {e}")
        return False

def setup_llama_cpp():
    """Setup llama.cpp for GGUF conversion."""
    llama_path = Path("llama.cpp")
    
    if not llama_path.exists():
        print("Cloning llama.cpp...")
        run_command(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"])
    
    print("Building llama.cpp...")
    run_command(["make", "-j8"], cwd=llama_path)
    return llama_path

def create_gguf_models(model_name, llama_path):
    """Create GGUF versions of a model."""
    print(f"\n=== Creating GGUF for {model_name} ===")
    
    # Download model if needed
    model_dir = Path(f"models/{model_name}")
    if not model_dir.exists():
        print(f"Downloading {model_name}...")
        run_command([
            "huggingface-cli", "download",
            f"Supra-Nexus/{model_name}",
            "--local-dir", str(model_dir)
        ])
    
    # Convert to GGUF F16
    print("Converting to GGUF F16...")
    gguf_dir = Path(f"models/{model_name}-gguf")
    gguf_dir.mkdir(parents=True, exist_ok=True)
    
    convert_script = llama_path / "convert_hf_to_gguf.py"
    f16_path = gguf_dir / f"{model_name}.gguf"
    
    run_command([
        sys.executable, str(convert_script),
        str(model_dir),
        "--outtype", "f16",
        "--outfile", str(f16_path)
    ])
    
    if not f16_path.exists():
        print("F16 conversion failed, trying legacy script...")
        # Try legacy convert.py
        convert_script = llama_path / "convert.py"
        run_command([
            sys.executable, str(convert_script),
            str(model_dir),
            "--outtype", "f16",
            "--outfile", str(f16_path)
        ])
    
    # Create quantized versions
    quantizations = ["Q4_K_M", "Q5_K_M", "Q8_0"]
    quantize_bin = llama_path / "llama-quantize"
    
    for quant in quantizations:
        print(f"Creating {quant} quantization...")
        quant_path = gguf_dir / f"{model_name}-{quant}.gguf"
        run_command([
            str(quantize_bin),
            str(f16_path),
            str(quant_path),
            quant
        ])
    
    return gguf_dir

def create_mlx_models(model_name):
    """Create MLX versions of a model."""
    print(f"\n=== Creating MLX for {model_name} ===")
    
    model_dir = Path(f"models/{model_name}")
    mlx_dir = Path(f"models/{model_name}-mlx")
    mlx_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to MLX format
    print("Converting to MLX format...")
    run_command([
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", str(model_dir),
        "--mlx-path", str(mlx_dir)
    ])
    
    # Create 4-bit quantized version
    print("Creating 4-bit quantization...")
    mlx_4bit_dir = Path(f"models/{model_name}-mlx-4bit")
    mlx_4bit_dir.mkdir(parents=True, exist_ok=True)
    
    run_command([
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", str(model_dir),
        "--mlx-path", str(mlx_4bit_dir),
        "-q"
    ])
    
    return mlx_dir, mlx_4bit_dir

def upload_to_huggingface(model_name, format_dir, format_type):
    """Upload model format to HuggingFace."""
    print(f"\n=== Uploading {format_type} for {model_name} ===")
    
    repo_id = f"Supra-Nexus/{model_name}-{format_type}"
    
    # Create repo if it doesn't exist
    run_command([
        "huggingface-cli", "repo", "create",
        repo_id,
        "--type", "model",
        "-y"
    ])
    
    # Upload files
    for file in format_dir.glob("*"):
        if file.is_file():
            print(f"Uploading {file.name}...")
            run_command([
                "huggingface-cli", "upload",
                repo_id,
                str(file),
                file.name
            ])

def main():
    """Main conversion and upload process."""
    models = [
        "supra-nexus-o1-instruct",
        "supra-nexus-o1-thinking"
    ]
    
    # Setup llama.cpp
    llama_path = setup_llama_cpp()
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Processing {model_name}")
        print('='*50)
        
        # Create GGUF versions
        gguf_dir = create_gguf_models(model_name, llama_path)
        if gguf_dir and gguf_dir.exists():
            upload_to_huggingface(model_name, gguf_dir, "gguf")
        
        # Create MLX versions
        mlx_dir, mlx_4bit_dir = create_mlx_models(model_name)
        if mlx_dir and mlx_dir.exists():
            upload_to_huggingface(model_name, mlx_dir, "mlx")
        if mlx_4bit_dir and mlx_4bit_dir.exists():
            upload_to_huggingface(model_name, mlx_4bit_dir, "mlx-4bit")
    
    print("\n✅ All formats created and uploaded!")
    print("\nAvailable models:")
    for model in models:
        print(f"\n{model}:")
        print(f"  - https://huggingface.co/Supra-Nexus/{model}")
        print(f"  - https://huggingface.co/Supra-Nexus/{model}-gguf")
        print(f"  - https://huggingface.co/Supra-Nexus/{model}-mlx")
        print(f"  - https://huggingface.co/Supra-Nexus/{model}-mlx-4bit")

if __name__ == "__main__":
    main()