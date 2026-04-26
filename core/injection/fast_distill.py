#!/usr/bin/env python3
"""تقطير فوري من GPT-2 إلى Zumar"""
# يحمل هذا الملف في core/injection/fast_distill.py

import json
import struct
import numpy as np
from pathlib import Path
from safetensors.numpy import save_file

# ============================================================
# استخراج الأوزان من GPT-2 مباشرة
# ============================================================

def download_gpt2_weights():
    """يحمل أوزان GPT-2 الصغيرة من HuggingFace"""
    import urllib.request
    
    url = "https://huggingface.co/gpt2/resolve/main/model.safetensors"
    path = "models/teacher/model.safetensors"
    
    print("Downloading GPT-2 weights (548MB)...")
    urllib.request.urlretrieve(url, path)
    print("Done!")
    return path

def extract_knowledge(teacher_path):
    """يستخرج المعرفة من GPT-2 ويحولها لـ Zumar"""
    
    # قراءة GPT-2
    with open(teacher_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data = f.read()
    
    def get_tensor(name):
        if name not in header: return None
        info = header[name]
        start, end = info["data_offsets"]
        raw = data[start:end]
        return np.frombuffer(raw, dtype=np.float16).reshape(info["shape"])
    
    # GPT-2: 768d, 12L, vocab=50257
    # Zumar: 1024d, 12L, vocab=50257
    
    weights = {}
    
    # 1. Embedding: [50257, 768] -> [50257, 1024]
    print("Extracting embedding...")
    wte = get_tensor("wte.weight")
    emb = np.zeros((50257, 1024), dtype=np.float16)
    emb[:, :768] = wte  # أول 768 بعد كما هي
    emb[:, 768:] = np.random.randn(50257, 256).astype(np.float16) * 0.001  # الباقي صغير
    
    weights["model.embed_tokens.weight"] = emb
    weights["lm_head.weight"] = emb.copy()
    weights["lm_head.bias"] = np.zeros(50257, dtype=np.float16)
    
    # 2. Final Norm
    print("Extracting norm...")
    ln_f = get_tensor("ln_f.weight")
    norm = np.ones(1024, dtype=np.float16)
    norm[:768] = ln_f
    weights["model.norm.weight"] = norm
    weights["model.norm.bias"] = np.zeros(1024, dtype=np.float16)
    
    # 3. 12 طبقة
    for i in range(12):
        print(f"Layer {i+1}/12...")
        p = f"model.layers.{i}"
        h = f"h.{i}"
        
        # GPT-2: c_attn = Q+K+V مدمجين (768, 2304)
        c_attn = get_tensor(f"{h}.attn.c_attn.weight")  # [768, 2304]
        c_attn_b = get_tensor(f"{h}.attn.c_attn.bias")  # [2304]
        
        # تفكيك Q, K, V
        q_w = np.zeros((1024, 1024), dtype=np.float16)
        k_w = np.zeros((1024, 1024), dtype=np.float16)
        v_w = np.zeros((1024, 1024), dtype=np.float16)
        
        q_w[:768, :768] = c_attn[:768, :]
        k_w[:768, :768] = c_attn[768:1536, :]
        v_w[:768, :768] = c_attn[1536:, :]
        
        for name, w in [("q_proj", q_w), ("k_proj", k_w), ("v_proj", v_w)]:
            weights[f"{p}.self_attn.{name}.weight"] = w
            b = np.zeros(1024, dtype=np.float16)
            if name == "q_proj": b[:768] = c_attn_b[:768]
            elif name == "k_proj": b[:768] = c_attn_b[768:1536]
            else: b[:768] = c_attn_b[1536:]
            weights[f"{p}.self_attn.{name}.bias"] = b
        
        # O projection: [768, 768] -> [1024, 1024]
        c_proj = get_tensor(f"{h}.attn.c_proj.weight")
        c_proj_b = get_tensor(f"{h}.attn.c_proj.bias")
        o_w = np.zeros((1024, 1024), dtype=np.float16)
        o_w[:768, :768] = c_proj
        weights[f"{p}.self_attn.o_proj.weight"] = o_w
        o_b = np.zeros(1024, dtype=np.float16)
        o_b[:768] = c_proj_b
        weights[f"{p}.self_attn.o_proj.bias"] = o_b
        
        # MoE Gate: من MLP الأولى
        c_fc = get_tensor(f"{h}.mlp.c_fc.weight")  # [768, 3072]
        gate_w = np.zeros((8, 1024), dtype=np.float16)
        gate_w[:, :768] = c_fc[:8, :]  # أول 8 صفوف
        weights[f"{p}.mlp.gate.weight"] = gate_w
        weights[f"{p}.mlp.gate.bias"] = np.zeros(8, dtype=np.float16)
        
        # MoE Experts: كل خبير نسخة من MLP
        c_proj_mlp = get_tensor(f"{h}.mlp.c_proj.weight")  # [3072, 768]
        for e in range(8):
            exp_w = np.zeros((1024, 1024), dtype=np.float16)
            exp_w[:768, :768] = c_fc[:1024, :] if e == 0 else c_fc[:1024, :] + np.random.randn(1024, 768).astype(np.float16) * 0.001
            weights[f"{p}.mlp.expert_{e}.weight"] = exp_w
            weights[f"{p}.mlp.expert_{e}.bias"] = np.zeros(1024, dtype=np.float16)
        
        # LayerNorms
        for src, dst in [("ln_1", "input_layernorm"), ("ln_2", "post_attention_layernorm")]:
            ln_w = get_tensor(f"{h}.{src}.weight")
            w = np.ones(1024, dtype=np.float16)
            w[:768] = ln_w
            weights[f"{p}.{dst}.weight"] = w
            b = np.zeros(1024, dtype=np.float16)
            ln_b = get_tensor(f"{h}.{src}.bias")
            if ln_b is not None: b[:768] = ln_b
            weights[f"{p}.{dst}.bias"] = b
    
    return weights

# ============================================================
# تنفيذ
# ============================================================

base = Path(__file__).parent.parent.parent
teacher_path = base / "models" / "teacher" / "model.safetensors"

# حمل GPT-2 إذا لم يكن موجوداً
if not teacher_path.exists():
    download_gpt2_weights()

# استخرج المعرفة
print("Extracting knowledge from GPT-2...")
weights = extract_knowledge(str(teacher_path))

# حفظ
output = base / "models" / "zumar-v1" / "model.safetensors"
output.parent.mkdir(parents=True, exist_ok=True)
save_file(weights, str(output))

total = sum(v.nbytes for v in weights.values())
print(f"\n✅ Done! Model: {total/1e6:.0f}MB")
print(f"Run: cargo run -p core --release")