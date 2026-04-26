#!/usr/bin/env python3
"""diagnose_weights.py - فحص جودة الأوزان"""
import numpy as np
import json
import struct
import os

def load_safetensors(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
        data = f.read()
    return header, data

def get_tensor(header, data, name):
    if name not in header:
        return None
    info = header[name]
    start, end = info["data_offsets"]
    raw = data[start:end]
    dtype = info.get("dtype", "F32")
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).reshape(info["shape"])
    return np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])

path = "models/zumar-v1/model.safetensors"
if not os.path.exists(path):
    print(f"❌ {path} not found!")
    exit(1)

header, data = load_safetensors(path)
print(f"✅ Loaded {len(header)} tensors\n")

# فحص عينات من الأوزان
key_samples = [
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.expert_0.weight",
]

for key in key_samples:
    t = get_tensor(header, data, key)
    if t is None:
        print(f"❌ {key}: MISSING")
    else:
        print(f"📦 {key}:")
        print(f"   shape={t.shape}, dtype={t.dtype}")
        print(f"   mean={t.mean():.6f}, std={t.std():.6f}")
        print(f"   min={t.min():.6f}, max={t.max():.6f}")
        zeros = (np.abs(t) < 1e-8).mean()
        print(f"   zeros={zeros:.2%}")
        print()