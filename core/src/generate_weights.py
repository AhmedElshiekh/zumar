import numpy as np
from safetensors.numpy import save_file
import os

def generate_zumar_weights_lite():
    d_model = 768
    vocab_size = 32000
    d_state = 16
    expand = 2
    d_inner = d_model * expand
    num_experts = 8

    print("⚡ Harmonizing Names: Matching Rust MoE naming convention...")

    weights = {
        "pre_norm.weight": np.random.randn(d_model, d_model).astype(np.float32) * 0.02,
        "pre_norm.bias": np.zeros(d_model).astype(np.float32),
        
        "mamba.in_proj.weight": np.random.randn(d_inner * 2, d_model).astype(np.float32) * 0.02,
        "mamba.in_proj.bias": np.zeros(d_inner * 2).astype(np.float32),
        
        "mamba.x_proj.weight": np.random.randn(d_state * 2 + d_inner, d_inner).astype(np.float32) * 0.02,
        "mamba.x_proj.bias": np.zeros(d_state * 2 + d_inner).astype(np.float32),
        
        "mamba.dt_proj.weight": np.random.randn(d_inner, d_inner).astype(np.float32) * 0.02,
        "mamba.dt_proj.bias": np.random.randn(d_inner).astype(np.float32) * 0.02,
        
        "mamba.out_proj.weight": np.random.randn(d_model, d_inner).astype(np.float32) * 0.02,
        "mamba.out_proj.bias": np.zeros(d_model).astype(np.float32),
        
        "mamba.a_log": np.random.randn(d_state, d_inner).astype(np.float32) * 0.02,
        "mamba.d": np.random.randn(d_inner).astype(np.float32) * 0.02,
        
        "moe.gate.weight": np.random.randn(num_experts, d_model).astype(np.float32) * 0.02,
        
        "post_norm.weight": np.random.randn(d_model, d_model).astype(np.float32) * 0.02,
        "post_norm.bias": np.zeros(d_model).astype(np.float32),
        "lm_head.weight": np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02,
        "lm_head.bias": np.zeros(vocab_size).astype(np.float32),
    }

    # حقن أوزان الخبراء بالتنسيق الذي طلبه Rust: expert_N.weight
    for i in range(num_experts):
        weights[f"moe.expert_{i}.weight"] = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        weights[f"moe.expert_{i}.bias"] = np.zeros(d_model).astype(np.float32)

    save_path = "core/models/zumar-v1/model.safetensors"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    save_file(weights, save_path)
    print(f"✅ Names Aligned! Weights created at: {save_path}")

if __name__ == "__main__":
    generate_zumar_weights_lite()
