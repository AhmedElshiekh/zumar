#!/usr/bin/env python3
"""
Zumar Intelligent Distillation v5.0
يحول أي نموذج معلم إلى إعدادات Zumar الثابتة بذكاء.
يستخدم: SVD، PCA، interpolation، و copying الذكي.
"""

import numpy as np
import json
import struct
import os
from pathlib import Path
from safetensors.numpy import save_file
from scipy import linalg  # لـ SVD - تثبيت: pip install scipy

# ============================================================
# 🎯 إعدادات Zumar الثابتة (مطابقة لـ main.rs)
# ============================================================
ZUMAR_CONFIG = {
    "vocab_size": 50257,
    "hidden_dim": 1024,
    "num_layers": 12,
    "num_experts": 8,
    "top_k": 2,
    "n_heads": 16,
    "head_dim": 64,  # 1024 / 16
}

# ============================================================
# 🔍 نظام اكتشاف المعلم
# ============================================================

class TeacherAnalyzer:
    def __init__(self, teacher_path):
        self.teacher_path = teacher_path
        self.buffer, self.header = self.load()
        self.config = self.analyze()
    
    def load(self):
        with open(self.teacher_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode("utf-8"))
            data = f.read()
        return data, header
    
    def get_tensor_raw(self, name):
        if name not in self.header:
            return None
        info = self.header[name]
        start, end = info["data_offsets"]
        raw = self.buffer[start:end]
        dtype = info.get("dtype", "F32")
        shape = info["shape"]
        
        if dtype == "BF16":
            bits = np.frombuffer(raw, dtype=np.uint16)
            return (bits.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        elif dtype == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape)
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    
    def get_tensor_info(self, name):
        if name not in self.header:
            return None
        return self.header[name]
    
    def analyze(self):
        config = {}
        
        # اكتشاف embedding
        for emb_name in ["model.embed_tokens.weight", "transformer.wte.weight",
                         "bert.embeddings.word_embeddings.weight", "gpt_neox.embed_in.weight"]:
            info = self.get_tensor_info(emb_name)
            if info:
                config["teacher_vocab"] = info["shape"][0]
                config["teacher_dim"] = info["shape"][1]
                config["emb_name"] = emb_name
                break
        
        if "teacher_dim" not in config:
            raise ValueError("Cannot detect embedding layer!")
        
        # اكتشاف عدد الطبقات
        num_layers = 0
        layer_keys = set()
        for key in self.header.keys():
            for pattern in ["model.layers.", "transformer.h.", "bert.encoder.layer."]:
                if pattern in key:
                    parts = key.split(".")
                    try:
                        idx = parts.index(pattern.strip(".")) if pattern.strip(".") in parts else -1
                        if idx == -1:
                            # جرب النمط المباشر
                            for p in ["layers", "h", "layer"]:
                                if p in parts:
                                    idx = parts.index(p) + 1
                                    break
                        if idx >= 0 and idx < len(parts):
                            layer_num = int(parts[idx])
                            layer_keys.add(layer_num)
                    except:
                        pass
        
        config["teacher_layers"] = max(layer_keys) + 1 if layer_keys else 12
        config["layer_format"] = self.detect_layer_format()
        
        # اكتشاف n_heads
        n_heads = None
        for q_name in ["model.layers.0.self_attn.q_proj.weight",
                       "transformer.h.0.attn.q_proj.weight"]:
            info = self.get_tensor_info(q_name)
            if info:
                out_dim = info["shape"][0]
                # جرب قواسم شائعة
                for head_dim in [32, 64, 96, 128]:
                    if out_dim % head_dim == 0:
                        candidate = out_dim // head_dim
                        if n_heads is None or candidate < n_heads:
                            n_heads = candidate
                break
        
        config["teacher_heads"] = n_heads or (config["teacher_dim"] // 64)
        config["teacher_head_dim"] = config["teacher_dim"] // config["teacher_heads"]
        
        # اكتشاف intermediate_dim
        for mlp_name in ["model.layers.0.mlp.up_proj.weight",
                         "transformer.h.0.mlp.c_fc.weight",
                         "model.layers.0.mlp.gate_proj.weight"]:
            info = self.get_tensor_info(mlp_name)
            if info:
                config["teacher_intermediate"] = info["shape"][0]
                break
        
        if "teacher_intermediate" not in config:
            config["teacher_intermediate"] = config["teacher_dim"] * 4
        
        return config
    
    def detect_layer_format(self):
        """يكتشف صيغة أسماء الطبقات"""
        for key in self.header.keys():
            if "model.layers.0.self_attn" in key:
                return "llama"
            if "transformer.h.0.attn" in key:
                return "gpt2"
            if "bert.encoder.layer.0" in key:
                return "bert"
        return "unknown"
    
    def get_layer_prefix(self, layer_idx):
        """يرجع بادئة الطبقة حسب الصيغة"""
        fmt = self.config["layer_format"]
        if fmt == "llama":
            return f"model.layers.{layer_idx}"
        elif fmt == "gpt2":
            return f"transformer.h.{layer_idx}"
        elif fmt == "bert":
            return f"bert.encoder.layer.{layer_idx}"
        return f"model.layers.{layer_idx}"


# ============================================================
# 🧠 محول الأبعاد الذكي
# ============================================================

class SmartDimensionConverter:
    """يحول الأوزان بين الأبعاد المختلفة بذكاء"""
    
    @staticmethod
    def svd_project(weight, target_shape):
        """استخدام SVD للإسقاط مع الحفاظ على المعلومات الأساسية"""
        src_rows, src_cols = weight.shape
        tgt_rows, tgt_cols = target_shape
        
        # SVD: W = U @ S @ V^T
        try:
            U, S, Vt = linalg.svd(weight.astype(np.float64), full_matrices=False)
            
            # إعادة بناء بالمكونات الرئيسية
            k = min(tgt_rows, tgt_cols, len(S))
            U_k = U[:, :k]
            S_k = np.diag(np.sqrt(S[:k]))  # نأخذ الجذر التربيعي للتوزيع
            Vt_k = Vt[:k, :]
            
            # بناء مصفوفتين: A بحجم [tgt_rows, k] و B بحجم [k, tgt_cols]
            # بحيث A @ B ≈ W مع الأبعاد الجديدة
            A = U_k @ S_k
            B = S_k @ Vt_k
            
            # تغيير حجم A و B للأبعاد المطلوبة
            A_resized = SmartDimensionConverter._resize_2d(A, (tgt_rows, k))
            B_resized = SmartDimensionConverter._resize_2d(B, (k, tgt_cols))
            
            return (A_resized @ B_resized).astype(np.float32)
        except:
            return SmartDimensionConverter._resize_2d(weight, target_shape)
    
    @staticmethod
    def pca_align(source, target_shape):
        """محاذاة الأبعاد باستخدام PCA"""
        src_rows, src_cols = source.shape
        tgt_rows, tgt_cols = target_shape
        
        # توسيع أو تقليص الصفوف
        if tgt_rows != src_rows:
            # استخدام PCA لتقليص/توسيع البعد
            cov = source @ source.T
            eigenvalues, eigenvectors = linalg.eigh(cov)
            
            # أخذ المكونات الرئيسية
            if tgt_rows < src_rows:
                # تقليص: أخذ أكبر المكونات
                idx = np.argsort(eigenvalues)[-tgt_rows:]
                eigenvectors = eigenvectors[:, idx]
                result = eigenvectors.T @ source
            else:
                # توسيع: إضافة مكونات عشوائية صغيرة
                result = np.zeros((tgt_rows, src_cols), dtype=np.float32)
                result[:src_rows, :] = source
                result[src_rows:, :] = np.random.normal(0, 0.001, (tgt_rows - src_rows, src_cols))
                return result
        else:
            result = source.copy()
        
        # توسيع أو تقليص الأعمدة
        if tgt_cols != src_cols:
            result = SmartDimensionConverter._resize_2d(result, (tgt_rows, tgt_cols))
        
        return result
    
    @staticmethod
    def _resize_2d(source, target_shape):
        """تغيير حجم مصفوفة ثنائية الأبعاد"""
        target = np.zeros(target_shape, dtype=np.float32)
        copy_0 = min(source.shape[0], target_shape[0])
        copy_1 = min(source.shape[1], target_shape[1])
        target[:copy_0, :copy_1] = source[:copy_0, :copy_1]
        
        # ملء الفراغات بقيم صغيرة جداً
        if target_shape[0] > source.shape[0]:
            target[source.shape[0]:, :] = np.random.normal(0, 0.0001, 
                (target_shape[0] - source.shape[0], target_shape[1]))
        if target_shape[1] > source.shape[1]:
            target[:, source.shape[1]:] = np.random.normal(0, 0.0001,
                (target_shape[0], target_shape[1] - source.shape[1]))
        
        return target


# ============================================================
# 🚀 نظام التقطير الذكي
# ============================================================

class IntelligentDistiller:
    def __init__(self, teacher_path):
        self.teacher = TeacherAnalyzer(teacher_path)
        self.zumar = ZUMAR_CONFIG
        self.converter = SmartDimensionConverter()
        
        print(f"""
╔══════════════════════════════════════════════╗
║  🧬 ZUMAR INTELLIGENT DISTILLER v5.0       ║
╠══════════════════════════════════════════════╣
║  Teacher: {self.teacher.config.get('teacher_dim', '?'):>4}d, {self.teacher.config.get('teacher_layers', '?'):>2}L, {self.teacher.config.get('teacher_heads', '?'):>2}h
║  Zumar:   {self.zumar['hidden_dim']:>4}d, {self.zumar['num_layers']:>2}L, {self.zumar['n_heads']:>2}h
╚══════════════════════════════════════════════╝
""")
    
    def convert_embedding(self):
        """تحويل تضمين المفردات بذكاء"""
        emb = self.teacher.get_tensor_raw(self.teacher.config["emb_name"])
        if emb is None:
            raise ValueError("Cannot load teacher embedding")
        
        teacher_vocab, teacher_dim = emb.shape
        target_vocab, target_dim = self.zumar["vocab_size"], self.zumar["hidden_dim"]
        
        print(f"📥 Embedding: {teacher_vocab}x{teacher_dim} → {target_vocab}x{target_dim}")
        
        if teacher_dim != target_dim:
            # تحويل الأبعاد باستخدام SVD
            print("   🔄 Using SVD projection for dimension conversion...")
            result = self.converter.svd_project(emb.T, (target_dim, teacher_vocab)).T
        else:
            result = emb.copy()
        
        # تغيير حجم المفردات
        if target_vocab != teacher_vocab:
            final = np.zeros((target_vocab, target_dim), dtype=np.float32)
            copy_size = min(teacher_vocab, target_vocab)
            final[:copy_size, :] = result[:copy_size, :]
            
            # للمفردات الجديدة: استخدم متوسط التضمينات الموجودة
            if target_vocab > teacher_vocab:
                mean_emb = result.mean(axis=0)
                for i in range(teacher_vocab, target_vocab):
                    final[i, :] = mean_emb + np.random.normal(0, 0.001, target_dim)
            
            result = final
        
        print(f"   ✅ Done: {result.shape}")
        return result.astype(np.float16)
    
    def convert_attention_weights(self, src_layer, teacher_heads):
        """تحويل أوزان الانتباه مع تغيير عدد الرؤوس"""
        target_dim = self.zumar["hidden_dim"]
        target_heads = self.zumar["n_heads"]
        teacher_dim = self.teacher.config["teacher_dim"]
        
        weights = {}
        
        # جلب أوزان Q, K, V, O
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            fmt = self.teacher.config["layer_format"]
            
            # جلب الوزن من المعلم
            if fmt == "llama":
                w = self.teacher.get_tensor_raw(f"{src_layer}.self_attn.{proj}.weight")
            elif fmt == "gpt2":
                if proj in ["q_proj", "k_proj", "v_proj"]:
                    w_full = self.teacher.get_tensor_raw(f"{src_layer}.attn.c_attn.weight")
                    if w_full is not None:
                        h = w_full.shape[1] // 3
                        if proj == "q_proj":
                            w = w_full[:, :h]
                        elif proj == "k_proj":
                            w = w_full[:, h:2*h]
                        else:
                            w = w_full[:, 2*h:]
                    else:
                        w = None
                else:
                    w = self.teacher.get_tensor_raw(f"{src_layer}.attn.c_proj.weight")
            else:
                w = None
            
            if w is None:
                # تهيئة عشوائية ذكية
                print(f"   📋 Random init: {proj}")
                w = np.random.normal(0, 0.02, (teacher_dim, teacher_dim)).astype(np.float32)
            
            # تحويل الأبعاد
            if w.shape[0] != target_dim or w.shape[1] != target_dim:
                w = self.converter.svd_project(w, (target_dim, target_dim))
            
            weights[f"self_attn.{proj}.weight"] = w.astype(np.float16)
            weights[f"self_attn.{proj}.bias"] = np.zeros(target_dim, dtype=np.float16)
        
        return weights
    
    def convert_moe_weights(self, src_layer):
        """تحويل أوزان MoE بذكاء"""
        target_dim = self.zumar["hidden_dim"]
        num_experts = self.zumar["num_experts"]
        teacher_intermediate = self.teacher.config.get("teacher_intermediate", target_dim * 4)
        
        weights = {}
        fmt = self.teacher.config["layer_format"]
        
        # جلب أوزان MLP من المعلم
        if fmt == "llama":
            gate_w = self.teacher.get_tensor_raw(f"{src_layer}.mlp.gate_proj.weight")
            up_w = self.teacher.get_tensor_raw(f"{src_layer}.mlp.up_proj.weight")
            down_w = self.teacher.get_tensor_raw(f"{src_layer}.mlp.down_proj.weight")
        elif fmt == "gpt2":
            gate_w = self.teacher.get_tensor_raw(f"{src_layer}.mlp.c_fc.weight")
            up_w = gate_w  # GPT-2 يدمجهم
            down_w = self.teacher.get_tensor_raw(f"{src_layer}.mlp.c_proj.weight")
        else:
            gate_w = up_w = down_w = None
        
        # تحويل gate
        gate_target = (num_experts, target_dim)
        if gate_w is not None:
            gate_w = self.converter.svd_project(gate_w, gate_target)
        else:
            gate_w = np.random.normal(0, 0.02, gate_target).astype(np.float32)
        weights["mlp.gate.weight"] = gate_w.astype(np.float16)
        weights["mlp.gate.bias"] = np.zeros(num_experts, dtype=np.float16)
        
        # تحويل الخبراء
        expert_target = (target_dim, target_dim)
        if up_w is not None:
            # تحويل up_proj ليكون expert
            base_expert = self.converter.svd_project(up_w, expert_target)
        else:
            base_expert = np.random.normal(0, 0.02, expert_target).astype(np.float32)
        
        # لكل خبير، نسخة معدلة قليلاً
        for e in range(num_experts):
            noise = np.random.normal(0, 0.0001, expert_target)
            expert_w = (base_expert + noise).clip(-3, 3).astype(np.float16)
            weights[f"mlp.expert_{e}.weight"] = expert_w
            weights[f"mlp.expert_{e}.bias"] = np.zeros(target_dim, dtype=np.float16)
        
        return weights
    
    def convert_layernorm(self, src_layer):
        """تحويل LayerNorm"""
        target_dim = self.zumar["hidden_dim"]
        weights = {}
        fmt = self.teacher.config["layer_format"]
        
        norm_mapping = {
            "llama": [("input_layernorm", "input_layernorm"),
                      ("post_attention_layernorm", "post_attention_layernorm")],
            "gpt2": [("ln_1", "input_layernorm"),
                     ("ln_2", "post_attention_layernorm")],
        }
        
        mapping = norm_mapping.get(fmt, norm_mapping["llama"])
        
        for teacher_norm, zumar_norm in mapping:
            w = self.teacher.get_tensor_raw(f"{src_layer}.{teacher_norm}.weight")
            if w is not None:
                # LayerNorm بسيط: مجرد scale و bias
                result = np.ones(target_dim, dtype=np.float32)
                copy_size = min(len(w), target_dim)
                result[:copy_size] = w[:copy_size]
                weights[f"{zumar_norm}.weight"] = result.astype(np.float16)
                weights[f"{zumar_norm}.bias"] = np.zeros(target_dim, dtype=np.float16)
        
        return weights
    
    def distill(self):
        """عملية التقطير الكاملة"""
        weights = {}
        
        # 1. Embedding
        print("\n🧬 Phase 1: Converting Embedding...")
        emb = self.convert_embedding()
        weights["model.embed_tokens.weight"] = emb
        weights["lm_head.weight"] = emb.copy()
        weights["lm_head.bias"] = np.zeros(self.zumar["vocab_size"], dtype=np.float16)
        
        # 2. Final LayerNorm
        print("\n📏 Phase 2: Final LayerNorm...")
        weights["model.norm.weight"] = np.ones(self.zumar["hidden_dim"], dtype=np.float16)
        weights["model.norm.bias"] = np.zeros(self.zumar["hidden_dim"], dtype=np.float16)
        
        # 3. Layers
        teacher_layers = self.teacher.config["teacher_layers"]
        zumar_layers = self.zumar["num_layers"]
        teacher_heads = self.teacher.config["teacher_heads"]
        
        print(f"\n🧠 Phase 3: Converting {zumar_layers} layers...")
        for i in range(zumar_layers):
            # اختيار طبقة المعلم (مع تدوير إذا لزم)
            teacher_idx = i % teacher_layers
            src = self.teacher.get_layer_prefix(teacher_idx)
            dest = f"model.layers.{i}"
            
            # Attention
            attn_weights = self.convert_attention_weights(src, teacher_heads)
            for k, v in attn_weights.items():
                weights[f"{dest}.{k}"] = v
            
            # MoE
            moe_weights = self.convert_moe_weights(src)
            for k, v in moe_weights.items():
                weights[f"{dest}.{k}"] = v
            
            # LayerNorm
            norm_weights = self.convert_layernorm(src)
            for k, v in norm_weights.items():
                weights[f"{dest}.{k}"] = v
            
            if i % 5 == 0:
                print(f"   ✅ Layer {i}/{zumar_layers}")
        
        return weights
    
    def save(self, weights, output_path):
        """حفظ الأوزان"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving {len(weights)} tensors...")
        save_file(weights, str(output_path))
        
        total_params = sum(np.prod(v.shape) for v in weights.values())
        size_gb = total_params * 2 / 1e9
        print(f"\n📊 Total Parameters: {total_params:,}")
        print(f"📊 Estimated Size: {size_gb:.2f} GB (FP16)")
        print(f"\n🚀 Distillation Complete!")


# ============================================================
# 🚀 المدخل الرئيسي
# ============================================================

def main():
    base_dir = Path(__file__).parent.parent.parent
    teacher_path = base_dir / "models" / "teacher" / "model.safetensors"
    output_path = base_dir / "models" / "zumar-v1" / "model.safetensors"
    
    if not teacher_path.exists():
        print(f"❌ Teacher model not found: {teacher_path}")
        print("💡 Place your model at: models/teacher/model.safetensors")
        return 1
    
    distiller = IntelligentDistiller(str(teacher_path))
    weights = distiller.distill()
    distiller.save(weights, str(output_path))
    
    # حفظ إعدادات Zumar للمرجع
    config_path = output_path.parent / "zumar_config.json"
    with open(config_path, "w") as f:
        json.dump(ZUMAR_CONFIG, f, indent=2)
    print(f"📝 Config saved to: {config_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())