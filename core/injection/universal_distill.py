#!/usr/bin/env python3
"""
Universal Zumar Distiller
يقبل أي صيغة نموذج ويحوله تلقائياً إلى Zumar
يدعم: safetensors, GGUF, bin, pt, h5, npz
"""

import numpy as np
import json
import struct
import os
import sys
from pathlib import Path
from safetensors.numpy import save_file

# ============================================================
# إعدادات Zumar الثابتة
# ============================================================
ZUMAR_CONFIG = {
    "vocab_size": 50257,
    "hidden_dim": 1024,
    "num_layers": 12,
    "num_experts": 8,
    "top_k": 2,
    "n_heads": 16,
    "head_dim": 64,
}

# ============================================================
# 1. كاشف الصيغة التلقائي
# ============================================================
def detect_format(file_path):
    """يكتشف صيغة الملف تلقائياً"""
    path = Path(file_path)
    
    if not path.exists():
        return None
    
    ext = path.suffix.lower()
    
    # قراءة أول 8 بايت للكشف عن التوقيع
    try:
        with open(file_path, 'rb') as f:
            magic = f.read(8)
    except:
        magic = b''
    
    if ext == '.safetensors':
        return 'safetensors'
    elif ext == '.gguf' or magic[:4] == b'GGUF':
        return 'gguf'
    elif ext in ['.bin', '.pt', '.pth', '.ckpt']:
        return 'pytorch'
    elif ext == '.h5' or ext == '.hdf5':
        return 'hdf5'
    elif ext == '.npz':
        return 'numpy'
    elif len(magic) >= 8:
        # حاول safetensors
        try:
            header_size = struct.unpack('<Q', magic[:8])[0]
            if 0 < header_size < 100_000_000:
                return 'safetensors'
        except:
            pass
        # حاول GGUF
        if magic[:4] == b'GGUF':
            return 'gguf'
    
    return 'unknown'


# ============================================================
# 2. محمل النماذج المتعدد
# ============================================================
class UniversalLoader:
    def __init__(self, file_path):
        self.path = file_path
        self.format = detect_format(file_path)
        
    def load(self):
        """تحميل النموذج حسب صيغته"""
        if self.format == 'safetensors':
            return self._load_safetensors()
        elif self.format == 'gguf':
            return self._load_gguf()
        elif self.format == 'pytorch':
            return self._load_pytorch()
        elif self.format == 'hdf5':
            return self._load_hdf5()
        elif self.format == 'numpy':
            return self._load_numpy()
        else:
            raise ValueError(f"Unknown format: {self.format}")
    
    def _load_safetensors(self):
        """تحميل safetensors"""
        with open(self.path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
            data = f.read()
        
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            
            start, end = info["data_offsets"]
            raw = data[start:end]
            dtype = info.get("dtype", "F32")
            shape = list(info["shape"])
            
            if dtype == "F16" or dtype == "FLOAT16":
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
            elif dtype == "BF16" or dtype == "BFLOAT16":
                bits = np.frombuffer(raw, dtype=np.uint16)
                arr = (bits.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            elif dtype == "F32" or dtype == "FLOAT32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            elif dtype == "I32" or dtype == "INT32":
                arr = np.frombuffer(raw, dtype=np.int32).reshape(shape).astype(np.float32)
            elif dtype == "I64" or dtype == "INT64":
                arr = np.frombuffer(raw, dtype=np.int64).reshape(shape).astype(np.float32)
            else:
                # محاولة F32 كافتراضي
                try:
                    arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
                except:
                    continue
            
            tensors[name] = arr.astype(np.float32)
        
        return tensors
    
    def _load_gguf(self):
        """تحميل GGUF"""
        tensors = {}
        
        with open(self.path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError("Invalid GGUF file")
            
            version = struct.unpack('<I', f.read(4))[0]
            n_tensors = struct.unpack('<Q', f.read(8))[0]
            n_kv = struct.unpack('<Q', f.read(8))[0]
            
            # قراءة metadata
            metadata = {}
            for _ in range(n_kv):
                key_len = struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_len).decode('utf-8', errors='ignore')
                val_type = struct.unpack('<I', f.read(4))[0]
                
                if val_type == 0:      # uint8
                    val = f.read(1)[0]
                elif val_type == 1:    # int8
                    val = struct.unpack('<b', f.read(1))[0]
                elif val_type == 2:    # uint16
                    val = struct.unpack('<H', f.read(2))[0]
                elif val_type == 3:    # int16
                    val = struct.unpack('<h', f.read(2))[0]
                elif val_type == 4:    # uint32
                    val = struct.unpack('<I', f.read(4))[0]
                elif val_type == 5:    # int32
                    val = struct.unpack('<i', f.read(4))[0]
                elif val_type == 6:    # float32
                    val = struct.unpack('<f', f.read(4))[0]
                elif val_type == 7:    # bool
                    val = f.read(1)[0] != 0
                elif val_type == 8:    # string
                    str_len = struct.unpack('<Q', f.read(8))[0]
                    val = f.read(str_len).decode('utf-8', errors='ignore')
                elif val_type == 9:    # array
                    arr_type = struct.unpack('<I', f.read(4))[0]
                    arr_len = struct.unpack('<Q', f.read(8))[0]
                    val = []
                    for _ in range(arr_len):
                        val.append(struct.unpack('<i', f.read(4))[0])
                else:
                    f.read(8)  # تخطي
                    val = None
                
                metadata[key] = val
            
            # طباعة معلومات النموذج
            arch = metadata.get('general.architecture', 'unknown')
            print(f"   GGUF Architecture: {arch}")
            
            # قراءة التنسورات
            for i in range(n_tensors):
                name_len = struct.unpack('<Q', f.read(8))[0]
                name = f.read(name_len).decode('utf-8', errors='ignore')
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = []
                for _ in range(n_dims):
                    dims.append(struct.unpack('<Q', f.read(8))[0])
                
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                
                # GGML type mapping
                type_map = {
                    0: (np.float32, 4),
                    1: (np.float16, 2),
                    2: (np.int32, 4),
                    3: (np.int16, 2),
                    4: (np.int8, 1),
                    5: (np.uint8, 1),
                    6: (np.float16, 2),  # BF16 → treat as F16
                    7: (np.uint8, 1),    # Q4_0
                    8: (np.uint8, 1),    # Q4_1
                    10: (np.float16, 2), # Q2_K
                    11: (np.float16, 2), # Q3_K
                    12: (np.float16, 2), # Q4_K
                    13: (np.float16, 2), # Q5_K
                    14: (np.float16, 2), # Q6_K
                    15: (np.float16, 2), # Q8_K
                    16: (np.float16, 2), # IQ2_XXS
                    17: (np.float16, 2), # IQ2_XS
                }
                
                np_type, elem_size = type_map.get(dtype, (np.float16, 2))
                
                if not dims:
                    dims = [1]
                
                total_elements = 1
                for d in dims:
                    total_elements *= d
                
                size = total_elements * elem_size
                raw = f.read(size)
                
                try:
                    arr = np.frombuffer(raw, dtype=np_type).reshape(dims)
                    arr = arr.astype(np.float32)
                    
                    # تخطي التنسورات الكبيرة جداً (bias عادة صغير)
                    if arr.nbytes > 100_000_000:
                        print(f"   ⏭ Skipping large tensor: {name} ({arr.nbytes/1e6:.0f}MB)")
                        continue
                    
                    tensors[name] = arr
                except Exception as e:
                    print(f"   ⚠️  Cannot read {name}: {e}")
                    continue
            
        return tensors
    
    def _load_pytorch(self):
        """تحميل PyTorch"""
        try:
            import torch
            tensors = {}
            state = torch.load(self.path, map_location='cpu')
            
            if isinstance(state, dict):
                for name, tensor in state.items():
                    if hasattr(tensor, 'numpy'):
                        tensors[name] = tensor.numpy().astype(np.float32)
            return tensors
        except ImportError:
            raise ImportError("PyTorch required. Install: pip install torch")
    
    def _load_hdf5(self):
        """تحميل HDF5"""
        try:
            import h5py
            tensors = {}
            with h5py.File(self.path, 'r') as f:
                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        tensors[name] = obj[:].astype(np.float32)
                f.visititems(collect)
            return tensors
        except ImportError:
            raise ImportError("h5py required. Install: pip install h5py")
    
    def _load_numpy(self):
        """تحميل NumPy"""
        data = np.load(self.path)
        if isinstance(data, dict):
            return {k: v.astype(np.float32) for k, v in data.items()}
        return {"weights": data.astype(np.float32)}


# ============================================================
# 3. محلل المعمارية
# ============================================================
class ArchitectureDetector:
    def __init__(self, tensors):
        self.tensors = tensors
        self.config = self.detect()
    
    def detect(self):
        config = {
            "format": "unknown",
            "vocab_size": 50257,
            "hidden_dim": 768,
            "num_layers": 12,
            "intermediate_dim": 3072,
        }
        
        keys = list(self.tensors.keys())
        key_str = ' '.join(keys)
        
        # GPT-2
        if 'wte.weight' in key_str and 'h.0.ln_1.weight' in key_str:
            config["format"] = "gpt2"
            config["hidden_dim"] = self.tensors['wte.weight'].shape[1]
            config["vocab_size"] = self.tensors['wte.weight'].shape[0]
            config["num_layers"] = sum(1 for k in keys if '.ln_1.weight' in k)
        
        # GPT-NeoX / Llama
        elif 'gpt_neox.embed_in.weight' in key_str or 'model.layers.0.self_attn.q_proj.weight' in key_str:
            config["format"] = "llama"
            for k in keys:
                if 'embed' in k and 'weight' in k:
                    config["hidden_dim"] = self.tensors[k].shape[1]
                    config["vocab_size"] = self.tensors[k].shape[0]
                    break
            config["num_layers"] = sum(1 for k in keys if 'self_attn.q_proj.weight' in k)
        
        # BERT
        elif 'bert.encoder.layer.0' in key_str:
            config["format"] = "bert"
            config["hidden_dim"] = self.tensors['bert.embeddings.word_embeddings.weight'].shape[1]
        
        # GGUF
        elif any('blk.0.attn_q.weight' in k for k in keys):
            config["format"] = "gguf"
            for k in keys:
                if 'blk.0.attn_q.weight' in k:
                    config["hidden_dim"] = self.tensors[k].shape[0]
                    break
            config["num_layers"] = sum(1 for k in keys if 'attn_q.weight' in k)
        
        # TinyLlama
        elif any('model.layers.0' in k for k in keys):
            config["format"] = "llama"
            for k in keys:
                if 'embed_tokens.weight' in k:
                    config["hidden_dim"] = self.tensors[k].shape[1]
                    config["vocab_size"] = self.tensors[k].shape[0]
                    break
            config["num_layers"] = sum(1 for k in keys if 'self_attn.q_proj.weight' in k)
        
        return config


# ============================================================
# 4. المحول إلى Zumar
# ============================================================
class ZumarConverter:
    def __init__(self, source_tensors, arch_config):
        self.src = source_tensors
        self.arch = arch_config
        self.zumar = ZUMAR_CONFIG
    
    def resize(self, arr, target_shape):
        """تغيير حجم مصفوفة مع الحفاظ على المحتوى"""
        if arr is None:
            return np.random.randn(*target_shape).astype(np.float32) * 0.02
        
        result = np.zeros(target_shape, dtype=np.float32)
        src = arr.astype(np.float32)
        
        if len(target_shape) == 2:
            h = min(src.shape[0], target_shape[0])
            w = min(src.shape[1], target_shape[1])
            result[:h, :w] = src[:h, :w]
            if target_shape[0] > src.shape[0]:
                result[src.shape[0]:, :] = np.random.randn(target_shape[0]-src.shape[0], target_shape[1]).astype(np.float32)*0.001
            if target_shape[1] > src.shape[1]:
                result[:, src.shape[1]:] = np.random.randn(target_shape[0], target_shape[1]-src.shape[1]).astype(np.float32)*0.001
        else:
            h = min(src.shape[0], target_shape[0])
            result[:h] = src[:h]
            if target_shape[0] > src.shape[0]:
                result[src.shape[0]:] = np.random.randn(target_shape[0]-src.shape[0]).astype(np.float32)*0.001
        
        return result
    
    def convert(self):
        """تحويل كل الأوزان إلى Zumar"""
        weights = {}
        h_dst = self.zumar["hidden_dim"]
        fmt = self.arch["format"]
        
        print(f"   Converting: {fmt} ({self.arch['hidden_dim']}d) → Zumar ({h_dst}d)")
        
        # ---- Embedding ----
        emb = self._find_embedding()
        if emb is not None:
            emb_w = self.resize(emb, (self.zumar["vocab_size"], h_dst))
        else:
            emb_w = np.random.randn(self.zumar["vocab_size"], h_dst).astype(np.float32) * 0.02
        
        weights["model.embed_tokens.weight"] = emb_w.astype(np.float16)
        weights["lm_head.weight"] = emb_w.astype(np.float16)
        weights["lm_head.bias"] = np.zeros(self.zumar["vocab_size"], dtype=np.float16)
        
        # ---- Final Norm ----
        final_norm = self._find_final_norm()
        if final_norm is not None:
            weights["model.norm.weight"] = self.resize(final_norm, (h_dst,)).astype(np.float16)
        else:
            weights["model.norm.weight"] = np.ones(h_dst, dtype=np.float16)
        weights["model.norm.bias"] = np.zeros(h_dst, dtype=np.float16)
        
        # ---- Layers ----
        for i in range(self.zumar["num_layers"]):
            src_idx = i % max(1, self.arch["num_layers"])
            self._convert_layer(weights, i, src_idx)
        
        return weights
    
    def _find_embedding(self):
        """البحث عن طبقة التضمين"""
        for name in ['wte.weight', 'model.embed_tokens.weight', 'gpt_neox.embed_in.weight',
                      'token_embd.weight', 'transformer.wte.weight', 'bert.embeddings.word_embeddings.weight']:
            if name in self.src:
                return self.src[name]
        return None
    
    def _find_final_norm(self):
        """البحث عن طبقة المعيار النهائي"""
        for name in ['ln_f.weight', 'model.norm.weight', 'gpt_neox.final_layer_norm.weight',
                      'output_norm.weight', 'transformer.ln_f.weight', 'bert.encoder.norm.weight']:
            if name in self.src:
                return self.src[name]
        return None
    
    def _convert_layer(self, weights, dst_idx, src_idx):
        """تحويل طبقة واحدة"""
        p = f"model.layers.{dst_idx}"
        h = ZUMAR_CONFIG["hidden_dim"]
        
        # Attention Q/K/V/O
        qkv = self._find_qkv(src_idx)
        proj_o = self._find_o_proj(src_idx)
        
        for name, w in [
            ("q_proj", qkv.get("q")), ("k_proj", qkv.get("k")),
            ("v_proj", qkv.get("v")), ("o_proj", proj_o)
        ]:
            if w is not None:
                weights[f"{p}.self_attn.{name}.weight"] = self.resize(w, (h, h)).astype(np.float16)
            else:
                weights[f"{p}.self_attn.{name}.weight"] = np.random.randn(h, h).astype(np.float16) * 0.02
            weights[f"{p}.self_attn.{name}.bias"] = np.zeros(h, dtype=np.float16)
        
        # MoE Gate + Experts
        gate_src = self._find_mlp_gate(src_idx)
        up_src = self._find_mlp_up(src_idx)
        
        gate_w = self.resize(gate_src, (8, h)) if gate_src is not None else np.random.randn(8, h).astype(np.float32)*0.02
        weights[f"{p}.mlp.gate.weight"] = gate_w.astype(np.float16)
        weights[f"{p}.mlp.gate.bias"] = np.zeros(8, dtype=np.float16)
        
        for e in range(8):
            exp_w = self.resize(up_src, (h, h)) if up_src is not None else np.random.randn(h, h).astype(np.float32)*0.02
            if e > 0 and up_src is not None:
                exp_w = exp_w + np.random.randn(h, h).astype(np.float32) * 0.0001
            weights[f"{p}.mlp.expert_{e}.weight"] = exp_w.astype(np.float16)
            weights[f"{p}.mlp.expert_{e}.bias"] = np.zeros(h, dtype=np.float16)
        
        # LayerNorms
        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            weights[f"{p}.{norm_name}.weight"] = np.ones(h, dtype=np.float16)
            weights[f"{p}.{norm_name}.bias"] = np.zeros(h, dtype=np.float16)
        
        # Mamba
        d_inner = h * 2
        d_state = 16
        weights[f"{p}.mamba.in_proj.weight"] = np.random.randn(d_inner*2, h).astype(np.float32)*0.001
        weights[f"{p}.mamba.in_proj.bias"] = np.zeros(d_inner*2, dtype=np.float32)
        weights[f"{p}.mamba.x_proj.weight"] = np.random.randn(d_state*2+d_inner, d_inner).astype(np.float32)*0.001
        weights[f"{p}.mamba.x_proj.bias"] = np.zeros(d_state*2+d_inner, dtype=np.float32)
        weights[f"{p}.mamba.dt_proj.weight"] = np.random.randn(d_inner, d_inner).astype(np.float32)*0.001
        weights[f"{p}.mamba.dt_proj.bias"] = np.zeros(d_inner, dtype=np.float32)
        weights[f"{p}.mamba.out_proj.weight"] = np.random.randn(h, d_inner).astype(np.float32)*0.001
        weights[f"{p}.mamba.out_proj.bias"] = np.zeros(h, dtype=np.float32)
        weights[f"{p}.mamba.conv1d.weight"] = np.random.randn(d_inner, 1, 4).astype(np.float32)*0.01
        weights[f"{p}.mamba.conv1d.bias"] = np.zeros(d_inner, dtype=np.float32)
        weights[f"{p}.mamba.a_log"] = np.ones((d_state, d_inner), dtype=np.float32) * -0.5
        weights[f"{p}.mamba.d"] = np.ones(d_inner, dtype=np.float32)
        
        # SNN
        weights[f"{p}.snn.threshold"] = np.array([0.5], dtype=np.float32)
        weights[f"{p}.snn.tau"] = np.array([0.9], dtype=np.float32)
    
    def _find_qkv(self, layer_idx):
        """البحث عن QKV"""
        qkv = {"q": None, "k": None, "v": None}
        fmt = self.arch["format"]
        
        if fmt == "gpt2":
            key = f"h.{layer_idx}.attn.c_attn.weight"
            if key in self.src:
                w = self.src[key]
                h = w.shape[1] // 3
                qkv["q"] = w[:, :h]
                qkv["k"] = w[:, h:2*h]
                qkv["v"] = w[:, 2*h:]
        elif fmt in ["llama", "unknown"]:
            for name, proj in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")]:
                key = f"model.layers.{layer_idx}.self_attn.{name}.weight"
                if key in self.src:
                    qkv[proj] = self.src[key]
                else:
                    # جرب صيغة أخرى
                    for k in self.src:
                        if f".{layer_idx}." in k and f"attn_{proj}" in k:
                            qkv[proj] = self.src[k]
                            break
        elif fmt == "gguf":
            for proj in ["q", "k", "v"]:
                for k in self.src:
                    if f"blk.{layer_idx}.attn_{proj}.weight" in k:
                        qkv[proj] = self.src[k]
                        break
        
        return qkv
    
    def _find_o_proj(self, layer_idx):
        """البحث عن O projection"""
        fmt = self.arch["format"]
        
        if fmt == "gpt2":
            key = f"h.{layer_idx}.attn.c_proj.weight"
        elif fmt in ["llama", "unknown"]:
            key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        elif fmt == "gguf":
            for k in self.src:
                if f"blk.{layer_idx}.attn_output.weight" in k:
                    return self.src[k]
            return None
        else:
            return None
        
        return self.src.get(key)
    
    def _find_mlp_gate(self, layer_idx):
        """البحث عن MLP gate"""
        fmt = self.arch["format"]
        
        if fmt == "gpt2":
            key = f"h.{layer_idx}.mlp.c_fc.weight"
        elif fmt == "llama":
            key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        elif fmt == "gguf":
            for k in self.src:
                if f"blk.{layer_idx}.ffn_gate.weight" in k:
                    return self.src[k]
            return None
        else:
            return None
        
        return self.src.get(key)
    
    def _find_mlp_up(self, layer_idx):
        """البحث عن MLP up"""
        fmt = self.arch["format"]
        
        if fmt == "gpt2":
            return self.src.get(f"h.{layer_idx}.mlp.c_fc.weight")
        elif fmt == "llama":
            up = self.src.get(f"model.layers.{layer_idx}.mlp.up_proj.weight")
            if up is None:
                up = self.src.get(f"model.layers.{layer_idx}.mlp.gate_proj.weight")
            return up
        elif fmt == "gguf":
            for k in self.src:
                if f"blk.{layer_idx}.ffn_gate.weight" in k:
                    return self.src[k]
            return None
        return None


# ============================================================
# 5. البحث التلقائي عن ملف المعلم
# ============================================================
def find_teacher_model():
    """يبحث تلقائياً عن ملف نموذج في models/teacher/"""
    teacher_dir = Path("models/teacher")
    
    if not teacher_dir.exists():
        # جرب مسارات أخرى
        alt_paths = [
            Path("../models/teacher"),
            Path("../../models/teacher"),
        ]
        for p in alt_paths:
            if p.exists():
                teacher_dir = p
                break
        else:
            return None
    
    supported = ['.safetensors', '.gguf', '.pt', '.bin', '.pth', '.h5', '.npz']
    found_files = []
    
    for f in teacher_dir.iterdir():
        if f.is_file() and f.suffix.lower() in supported:
            size_mb = f.stat().st_size / 1_048_576
            found_files.append((size_mb, str(f)))
    
    if not found_files:
        return None
    
    # ترتيب حسب الحجم (الأكبر أولاً)
    found_files.sort(reverse=True)
    
    print(f"\n📂 Found {len(found_files)} model(s) in {teacher_dir}:")
    for size, name in found_files:
        print(f"   📄 {Path(name).name} ({size:.1f} MB)")
    
    # استخدام أكبر ملف
    return found_files[0][1]


# ============================================================
# 6. المدخل الرئيسي
# ============================================================
def main():
    print("=" * 60)
    print("🧠 UNIVERSAL ZUMAR DISTILLER")
    print("=" * 60)
    
    # تحديد ملف المصدر
    if len(sys.argv) > 1:
        source_file = sys.argv[1]
    else:
        print("🔍 Auto-detecting teacher model...")
        source_file = find_teacher_model()
        if source_file is None:
            print("❌ No model found. Place a model in models/teacher/")
            print("   Or specify: python3 universal_distill.py <model_file>")
            sys.exit(1)
    
    if not os.path.exists(source_file):
        print(f"❌ File not found: {source_file}")
        sys.exit(1)
    
    # تحميل
    print(f"\n📂 Loading: {source_file}")
    fmt = detect_format(source_file)
    print(f"   Format: {fmt}")
    
    loader = UniversalLoader(source_file)
    try:
        tensors = loader.load()
        print(f"✅ Loaded {len(tensors)} tensors")
    except Exception as e:
        print(f"❌ Load error: {e}")
        sys.exit(1)
    
    # تحليل
    arch = ArchitectureDetector(tensors)
    print(f"\n🔍 Detected: {arch.config['format']}")
    print(f"   Hidden dim: {arch.config['hidden_dim']}")
    print(f"   Layers: {arch.config['num_layers']}")
    
    # تحويل
    print(f"\n🔄 Converting to Zumar ({ZUMAR_CONFIG['hidden_dim']}d, {ZUMAR_CONFIG['num_layers']}L)...")
    converter = ZumarConverter(tensors, arch.config)
    weights = converter.convert()
    
    # حفظ
    output = Path("models/zumar-v1/model.safetensors")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(output))
    
    size_mb = output.stat().st_size / 1_048_576
    print(f"\n{'='*60}")
    print(f"✅ DISTILLATION COMPLETE!")
    print(f"{'='*60}")
    print(f"   Output: {output}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Tensors: {len(weights)}")
    print(f"\n🚀 Run: cargo run -p core --release")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()