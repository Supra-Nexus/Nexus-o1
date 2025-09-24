#!/usr/bin/env python3
"""Convert dataset to MLX expected format."""

import json
from pathlib import Path

def convert_to_mlx_format(input_file, output_file):
    """Convert chat format to MLX format."""
    data = []

    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            messages = entry.get("messages", [])

            # Convert to text format
            text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    text += f"User: {content}\n\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n\n"

            # Save as text field
            data.append({"text": text.strip()})

    # Write output
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    print(f"Converted {len(data)} examples from {input_file} to {output_file}")

if __name__ == "__main__":
    base_dir = Path("/Users/z/work/supra/o1")

    # Convert thinking datasets
    convert_to_mlx_format(
        base_dir / "training/supra_thinking_train.jsonl",
        base_dir / "training/mlx_thinking_train.jsonl"
    )

    convert_to_mlx_format(
        base_dir / "training/supra_thinking_valid.jsonl",
        base_dir / "training/mlx_thinking_valid.jsonl"
    )

    # Convert instruct datasets
    convert_to_mlx_format(
        base_dir / "training/supra_instruct_train.jsonl",
        base_dir / "training/mlx_instruct_train.jsonl"
    )

    convert_to_mlx_format(
        base_dir / "training/supra_instruct_valid.jsonl",
        base_dir / "training/mlx_instruct_valid.jsonl"
    )