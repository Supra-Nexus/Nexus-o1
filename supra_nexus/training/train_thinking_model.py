#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, '/Users/z/work/zen/zen-nano')

import mlx.core as mx
import mlx_lm

def main():
    print("🚀 Training Supra Nexus o1 Thinking Model")
    print("   Base: Qwen3-4B-Thinking-2507-MLX-8bit")
    print("   Training data: training/supra_thinking.jsonl")
    
    try:
        # Train the thinking model
        mlx_lm.lora.lora(
            model='base-models/Qwen3-4B-Thinking-2507-MLX-8bit',
            train='training/supra_thinking.jsonl',
            valid='training/supra_thinking.jsonl',
            adapter_path='models/supra-nexus-o1-thinking-adapters',
            iters=200,
            learning_rate=1e-5,
            batch_size=1,
            lora_layers=16,
            steps_per_report=10,
            steps_per_eval=50,
            save_every=100,
            test=False
        )
        print('✅ Supra o1 thinking model training completed!')
        print('   Adapter saved to: models/supra-nexus-o1-thinking-adapters')
        
        # Fuse the model
        print('🔗 Fusing Supra o1 thinking model...')
        mlx_lm.fuse.fuse(
            model='base-models/Qwen3-4B-Thinking-2507-MLX-8bit',
            adapter_path='models/supra-nexus-o1-thinking-adapters',
            save_path='models/supra-nexus-o1-thinking-fused',
            de_quantize=True
        )
        print('✅ Supra o1 thinking model fused!')
        print('   Fused model saved to: models/supra-nexus-o1-thinking-fused')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())