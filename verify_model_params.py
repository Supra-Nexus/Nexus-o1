#!/usr/bin/env python3
"""Verify and calculate actual model parameters for Qwen3-4B-2507."""

def calculate_qwen3_4b_params():
    """Calculate exact parameter count for Qwen3-4B-2507 model."""
    
    # Model configuration from config.json
    vocab_size = 151936
    hidden_size = 2560
    intermediate_size = 9728
    num_hidden_layers = 36
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    
    total_params = 0
    
    # Embedding layer
    embedding_params = vocab_size * hidden_size
    total_params += embedding_params
    print(f"Embedding: {embedding_params:,} params")
    
    # Each transformer layer
    layer_params = 0
    
    # Self-attention
    # Q, K, V projections (with GQA - grouped query attention)
    q_params = hidden_size * (num_attention_heads * head_dim)
    k_params = hidden_size * (num_key_value_heads * head_dim)
    v_params = hidden_size * (num_key_value_heads * head_dim)
    o_params = (num_attention_heads * head_dim) * hidden_size
    
    attention_params = q_params + k_params + v_params + o_params
    layer_params += attention_params
    
    # MLP/FFN
    up_proj = hidden_size * intermediate_size
    down_proj = intermediate_size * hidden_size
    gate_proj = hidden_size * intermediate_size  # For SwiGLU
    
    mlp_params = up_proj + down_proj + gate_proj
    layer_params += mlp_params
    
    # Layer norms (2 per layer)
    ln_params = 2 * hidden_size
    layer_params += ln_params
    
    print(f"Per layer: {layer_params:,} params")
    print(f"  - Attention: {attention_params:,}")
    print(f"  - MLP: {mlp_params:,}")
    print(f"  - LayerNorm: {ln_params:,}")
    
    # Total for all layers
    all_layers_params = layer_params * num_hidden_layers
    total_params += all_layers_params
    print(f"All {num_hidden_layers} layers: {all_layers_params:,} params")
    
    # Final layer norm
    final_ln_params = hidden_size
    total_params += final_ln_params
    print(f"Final LayerNorm: {final_ln_params:,} params")
    
    # Output projection (if not tied with embeddings)
    # Note: tie_word_embeddings = true, so no additional params
    
    print(f"\n{'='*50}")
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print(f"Approximately: {total_params/1e9:.2f}B parameters")
    
    # Model size estimates
    print(f"\n{'='*50}")
    print("Model Size Estimates:")
    
    # FP16/BF16 size
    fp16_size = total_params * 2 / (1024**3)
    print(f"FP16/BF16: ~{fp16_size:.1f} GB")
    
    # INT8 size
    int8_size = total_params * 1 / (1024**3)
    print(f"INT8: ~{int8_size:.1f} GB")
    
    # INT4 size
    int4_size = total_params * 0.5 / (1024**3)
    print(f"INT4: ~{int4_size:.1f} GB")
    
    # Typical quantized sizes with overhead
    print(f"\nWith quantization overhead:")
    print(f"8-bit quantized: ~{int8_size * 1.1:.1f} GB")
    print(f"4-bit quantized: ~{int4_size * 1.2:.1f} GB")

if __name__ == "__main__":
    calculate_qwen3_4b_params()