import numpy as np
from safetensors.numpy import save_file
import os

def generate_zumar_sovereign_weights():
    d_model = 1024       
    vocab_size = 50257   
    num_experts = 8     
    d_state = 16        
    expand = 2
    d_inner = d_model * expand 

    print(f"💎 Generating Sovereign Weights: d_model={d_model}, d_inner={d_inner}...")

    def quantize_1_58bit(shape):
        return np.sign(np.random.randn(*shape)).astype(np.float32)

    weights = {
        "model.embed_tokens.weight": quantize_1_58bit((vocab_size, d_model)),
        "model.norm.weight": np.ones(d_model).astype(np.float32),
        "model.norm.bias": np.zeros(d_model).astype(np.float32),
    }

    for i in [0, 29]:
        prefix = f"model.layers.{i}"
        
        # --- Mamba Block ---
        weights[f"{prefix}.self_attn.in_proj.weight"] = quantize_1_58bit((d_inner * 2, d_model))
        weights[f"{prefix}.self_attn.in_proj.bias"] = np.zeros(d_inner * 2).astype(np.float32)
        
        weights[f"{prefix}.self_attn.x_proj.weight"] = quantize_1_58bit((d_state * 2 + d_inner, d_inner))
        weights[f"{prefix}.self_attn.x_proj.bias"] = np.zeros(d_state * 2 + d_inner).astype(np.float32)
        
        weights[f"{prefix}.self_attn.dt_proj.weight"] = quantize_1_58bit((d_inner, d_inner))
        weights[f"{prefix}.self_attn.dt_proj.bias"] = np.zeros(d_inner).astype(np.float32)
        
        weights[f"{prefix}.self_attn.out_proj.weight"] = quantize_1_58bit((d_model, d_inner))
        weights[f"{prefix}.self_attn.out_proj.bias"] = np.zeros(d_model).astype(np.float32)

        # التصحيح النهائي: استخدام 'd' الصغيرة بدلاً من 'D'
        weights[f"{prefix}.self_attn.a_log"] = np.random.randn(d_state, d_inner).astype(np.float32)
        weights[f"{prefix}.self_attn.d"] = np.ones(d_inner).astype(np.float32)

        # --- MoE Block ---
        weights[f"{prefix}.mlp.gate.weight"] = quantize_1_58bit((num_experts, d_model))
        for e in range(num_experts):
            weights[f"{prefix}.mlp.expert_{e}.weight"] = quantize_1_58bit((d_model, d_model))
            weights[f"{prefix}.mlp.expert_{e}.bias"] = np.zeros(d_model).astype(np.float32)

        # --- Norms ---
        weights[f"{prefix}.input_layernorm.weight"] = np.ones(d_model).astype(np.float32)
        weights[f"{prefix}.input_layernorm.bias"] = np.zeros(d_model).astype(np.float32)
        weights[f"{prefix}.post_attention_layernorm.weight"] = np.ones(d_model).astype(np.float32)
        weights[f"{prefix}.post_attention_layernorm.bias"] = np.zeros(d_model).astype(np.float32)

    save_path = "core/models/zumar-v1/model.safetensors"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_file(weights, save_path)
    print(f"✅ Final Corrected Weights Created: {save_path}")

if __name__ == "__main__":
    generate_zumar_sovereign_weights()
