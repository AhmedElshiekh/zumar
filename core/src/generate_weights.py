import numpy as np
from safetensors.numpy import save_file
import os

def generate_bitnet_weights():
    d_model = 768
    vocab_size = 32000
    d_state = 16
    expand = 2
    d_inner = d_model * expand
    num_experts = 8

    print("💎 Generating BitNet 1.58-bit Weights (-1, 0, 1)...")

    def quantize_1_58bit(shape):
        # توليد قيم عشوائية ثم تحويلها إلى {-1, 0, 1}
        random_matrix = np.random.randn(*shape)
        return np.sign(random_matrix).astype(np.float32)

    weights = {
        "pre_norm.weight": np.ones(d_model).astype(np.float32),
        "mamba.in_proj.weight": quantize_1_58bit((d_inner * 2, d_model)),
        "mamba.in_proj.bias": np.zeros(d_inner * 2).astype(np.float32),
        "mamba.x_proj.weight": quantize_1_58bit((d_state * 2 + d_inner, d_inner)),
        "mamba.dt_proj.weight": quantize_1_58bit((d_inner, d_inner)),
        "mamba.dt_proj.bias": np.zeros(d_inner).astype(np.float32),
        "mamba.out_proj.weight": quantize_1_58bit((d_model, d_inner)),
        "mamba.a_log": np.random.randn(d_state, d_inner).astype(np.float32),
        "mamba.d": np.ones(d_inner).astype(np.float32),
        "moe.gate.weight": quantize_1_58bit((num_experts, d_model)),
        "post_norm.weight": np.ones(d_model).astype(np.float32),
        "lm_head.weight": quantize_1_58bit((vocab_size, d_model)),
    }

    for i in range(num_experts):
        weights[f"moe.expert_{i}.weight"] = quantize_1_58bit((d_model, d_model))
        weights[f"moe.expert_{i}.bias"] = np.zeros(d_model).astype(np.float32)

    save_path = "core/models/zumar-v1/model.safetensors"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_file(weights, save_path)
    print(f"✅ BitNet Weights Created: {save_path}")

if __name__ == "__main__":
    generate_bitnet_weights()
