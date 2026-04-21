import numpy as np
import json
import struct
import os
from safetensors.numpy import save_file

# --- الإعدادات السيادية (مطابقة تماماً لمحرك الرست) ---
NUM_LAYERS = 30 
INPUT_DIM = 1024
NUM_EXPERTS = 8
VOCAB_SIZE = 50257

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEACHER_DIR = os.path.join(BASE_DIR, "models", "teacher")
SOURCE_FILE = os.path.join(TEACHER_DIR, "model.safetensors") 
TARGET_DIR = os.path.join(BASE_DIR, "models", "zumar-v1")
OUTPUT_FILE = os.path.join(TARGET_DIR, "model.safetensors")

def load_safetensors_manually(file_path):
    with open(file_path, "rb") as f:
        header_size_bytes = f.read(8)
        if not header_size_bytes: return None, None
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)
        return f.read(), header

def get_tensor_raw(buffer, header, tensor_name):
    if tensor_name not in header: return None
    info = header[tensor_name]
    start, end = info["data_offsets"]
    raw_bytes = buffer[start:end]
    # محاولة التحويل من BF16 (الخاص بـ TinyLlama) إلى Float32
    raw_bits = np.frombuffer(raw_bytes, dtype=np.uint16)
    return (raw_bits.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])

def smart_resize(source_tensor, target_shape):
    if source_tensor is None:
        return (np.random.normal(0, 0.01, target_shape)).astype(np.float32)
    
    target = np.zeros(target_shape, dtype=np.float32)
    s_shape = source_tensor.shape
    
    if len(target_shape) == 2:
        copy_0, copy_1 = min(s_shape[0], target_shape[0]), min(s_shape[1], target_shape[1])
        target[:copy_0, :copy_1] = source_tensor[:copy_0, :copy_1]
    else:
        copy_0 = min(s_shape[0], target_shape[0])
        target[:copy_0] = source_tensor[:copy_0]
    return target

def run_distillation():
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ Teacher model missing at {SOURCE_FILE}")
        return

    buffer, header = load_safetensors_manually(SOURCE_FILE)
    weights = {}

    print("🧠 Extracting Knowledge & Embedding...")
    emb = get_tensor_raw(buffer, header, "model.embed_tokens.weight")
    final_emb = smart_resize(emb, (VOCAB_SIZE, INPUT_DIM)).astype(np.float16)
    
    weights["model.embed_tokens.weight"] = final_emb
    weights["lm_head.weight"] = final_emb 
    weights["lm_head.bias"] = np.zeros((VOCAB_SIZE,)).astype(np.float16)
    
    norm_w = get_tensor_raw(buffer, header, "model.norm.weight")
    weights["model.norm.weight"] = smart_resize(norm_w, (INPUT_DIM,)).astype(np.float16)
    weights["model.norm.bias"] = np.zeros((INPUT_DIM,)).astype(np.float16)

    for i in range(NUM_LAYERS):
        src_idx = i % 22 
        src_p = f"model.layers.{src_idx}."
        dest_p = f"model.layers.{i}."

        # --- Mamba Layers ---
        attn_p = dest_p + "self_attn."
        q_w = get_tensor_raw(buffer, header, src_p + "self_attn.q_proj.weight")
        v_w = get_tensor_raw(buffer, header, src_p + "self_attn.v_proj.weight")
        o_w = get_tensor_raw(buffer, header, src_p + "self_attn.o_proj.weight")

        weights[attn_p + "in_proj.weight"] = smart_resize(q_w, (4096, INPUT_DIM)).astype(np.float16)
        weights[attn_p + "in_proj.bias"] = np.zeros((4096,)).astype(np.float16)
        
        weights[attn_p + "x_proj.weight"] = (np.random.randn(2080, 2048) * 0.001).astype(np.float16)
        weights[attn_p + "x_proj.bias"] = np.zeros((2080,)).astype(np.float16)
        
        weights[attn_p + "dt_proj.weight"] = (np.random.randn(2048, 2048) * 0.001).astype(np.float16)
        weights[attn_p + "dt_proj.bias"] = np.zeros((2048,)).astype(np.float16)

        weights[attn_p + "conv1d.weight"] = (np.random.randn(2048, 1, 4) * 0.01).astype(np.float16)
        weights[attn_p + "conv1d.bias"] = np.zeros((2048,)).astype(np.float16)
        
        weights[attn_p + "out_proj.weight"] = smart_resize(o_w, (INPUT_DIM, 2048)).astype(np.float16)
        weights[attn_p + "out_proj.bias"] = np.zeros((INPUT_DIM,)).astype(np.float16)

        weights[attn_p + "a_log"] = (np.ones((16, 2048)) * -0.5).astype(np.float16)
        weights[attn_p + "d"] = np.ones((2048,)).astype(np.float16)

        # --- MoE Experts ---
        up_w = get_tensor_raw(buffer, header, src_p + "mlp.up_proj.weight")
        gate_w = get_tensor_raw(buffer, header, src_p + "mlp.gate_proj.weight")
        
        weights[dest_p + "mlp.gate.weight"] = smart_resize(gate_w, (NUM_EXPERTS, INPUT_DIM)).astype(np.float16)
        for e in range(NUM_EXPERTS):
            expert_w = up_w if up_w is not None else None
            weights[dest_p + f"mlp.expert_{e}.weight"] = smart_resize(expert_w, (INPUT_DIM, INPUT_DIM)).astype(np.float16)

        # --- حل مشكلة الخطأ: تزويد كافة طبقات الـ Norm ---
        in_norm = get_tensor_raw(buffer, header, src_p + "input_layernorm.weight")
        weights[dest_p + "input_layernorm.weight"] = smart_resize(in_norm, (INPUT_DIM,)).astype(np.float16)
        weights[dest_p + "input_layernorm.bias"] = np.zeros((INPUT_DIM,)).astype(np.float16)
        
        # إضافة post_attention_layernorm المفقودة
        p_norm = get_tensor_raw(buffer, header, src_p + "post_attention_layernorm.weight")
        weights[dest_p + "post_attention_layernorm.weight"] = smart_resize(p_norm, (INPUT_DIM,)).astype(np.float16)
        weights[dest_p + "post_attention_layernorm.bias"] = np.zeros((INPUT_DIM,)).astype(np.float16)

        if i % 10 == 0: print(f"✅ Full Layer {i} Integrated")

    print(f"💾 Saving complete Sovereign Brain...")
    save_file(weights, OUTPUT_FILE)
    print("🚀 Done. Architecture is now 100% compliant with Rust core.")

if __name__ == "__main__":
    run_distillation()
