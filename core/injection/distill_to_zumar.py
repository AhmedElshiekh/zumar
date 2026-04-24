#!/usr/bin/env python3
"""
Zumar Distillation Script v3.0
يحول أوزان نموذج معلم (Teacher) إلى هيكل ZumarModel متعدد الطبقات.
يدعم: 30 طبقة، MoE بـ 8 خبراء، Mamba، LayerNorm، Embedding، LM Head.
"""

import numpy as np
import json
import struct
import os
from safetensors.numpy import save_file

# --- الإعدادات (مطابقة لـ main.rs) ---
NUM_LAYERS = 12          # عدد الطبقات
INPUT_DIM = 1024          # البعد الخفي
NUM_EXPERTS = 8           # عدد الخبراء في MoE
TOP_K = 2                 # عدد الخبراء النشطين
VOCAB_SIZE = 50257        # حجم المفردات
N_HEADS = 16              # عدد رؤوس الانتباه
HEAD_DIM = INPUT_DIM // N_HEADS  # 64

# --- مسارات الملفات ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEACHER_DIR = os.path.join(BASE_DIR, "models", "teacher")
SOURCE_FILE = os.path.join(TEACHER_DIR, "model.safetensors")
TARGET_DIR = os.path.join(BASE_DIR, "models", "zumar-v1")
OUTPUT_FILE = os.path.join(TARGET_DIR, "model.safetensors")


def load_safetensors_manually(file_path):
    """
    تحميل ملف safetensors يدوياً بدون مكتبة safetensors كاملة.
    يدعم تنسيق BF16 (الشائع في TinyLlama والنماذج الصغيرة).
    """
    with open(file_path, "rb") as f:
        # قراءة حجم الهيدر (8 بايت الأولى)
        header_size_bytes = f.read(8)
        if not header_size_bytes or len(header_size_bytes) < 8:
            return None, None
        
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        
        # قراءة الهيدر JSON
        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)
        
        # قراءة البيانات الخام
        data_buffer = f.read()
    
    return data_buffer, header


def get_tensor_raw(buffer, header, tensor_name):
    """
    استخراج تنسور من البيانات الخام.
    يدعم التحويل من BF16 إلى Float32.
    """
    if buffer is None or header is None:
        return None
    
    if tensor_name not in header:
        print(f"  ⚠️  Tensor '{tensor_name}' not found in teacher model")
        return None
    
    info = header[tensor_name]
    start, end = info["data_offsets"]
    raw_bytes = buffer[start:end]
    
    # تحديد نوع البيانات
    dtype = info.get("dtype", "F32")
    
    if dtype == "BF16":
        # تحويل BF16 (uint16) إلى Float32
        raw_bits = np.frombuffer(raw_bytes, dtype=np.uint16)
        # BF16 إلى F32: إزاحة 16 بت لليسار
        return (raw_bits.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
    elif dtype == "F16":
        return np.frombuffer(raw_bytes, dtype=np.float16).reshape(info["shape"])
    elif dtype == "F32":
        return np.frombuffer(raw_bytes, dtype=np.float32).reshape(info["shape"])
    else:
        print(f"  ⚠️  Unknown dtype '{dtype}' for '{tensor_name}', trying F32")
        return np.frombuffer(raw_bytes, dtype=np.float32).reshape(info["shape"])


def smart_resize(source_tensor, target_shape, tensor_name=""):
    """
    تغيير حجم تنسور معلم ليطابق أبعاد Zumar.
    إذا كان المصدر None، يُنشئ تهيئة عشوائية صغيرة.
    """
    if source_tensor is None:
        print(f"    📋 Initializing '{tensor_name}' randomly (shape: {target_shape})")
        return (np.random.normal(0, 0.01, target_shape)).astype(np.float32)
    
    target = np.zeros(target_shape, dtype=np.float32)
    s_shape = source_tensor.shape
    
    if len(target_shape) == 2:
        # مصفوفة ثنائية الأبعاد
        copy_0 = min(s_shape[0], target_shape[0])
        copy_1 = min(s_shape[1], target_shape[1])
        target[:copy_0, :copy_1] = source_tensor[:copy_0, :copy_1]
        
        # إذا كان هناك أعمدة إضافية، نُهيئها بقيم صغيرة
        if target_shape[1] > s_shape[1]:
            target[:, s_shape[1]:] = np.random.normal(0, 0.001, (target_shape[0], target_shape[1] - s_shape[1]))
        if target_shape[0] > s_shape[0]:
            target[s_shape[0]:, :] = np.random.normal(0, 0.001, (target_shape[0] - s_shape[0], target_shape[1]))
    else:
        # متجه أحادي البعد
        copy_0 = min(s_shape[0], target_shape[0])
        target[:copy_0] = source_tensor[:copy_0]
        if target_shape[0] > s_shape[0]:
            target[s_shape[0]:] = np.random.normal(0, 0.001, (target_shape[0] - s_shape[0],))
    
    return target


def create_mamba_weights(src_prefix, dest_prefix, buffer, header):
    """
    إنشاء أوزان طبقة Mamba من أوزان الانتباه للمعلم.
    
    هيكل Mamba في Zumar:
    - in_proj: [4096, 1024] (مضاعف بمقدار expand*2 = 4)
    - x_proj: [2080, 2048] (d_state*2 + d_inner = 16*2 + 2048)
    - dt_proj: [2048, 2048]
    - out_proj: [1024, 2048]
    - conv1d: [2048, 1, 4]
    - a_log: [16, 2048]
    - d: [2048]
    """
    weights = {}
    d_inner = INPUT_DIM * 2  # 2048
    d_state = 16
    
    # in_proj: دمج q_proj و k_proj من المعلم
    q_w = get_tensor_raw(buffer, header, f"{src_prefix}self_attn.q_proj.weight")
    k_w = get_tensor_raw(buffer, header, f"{src_prefix}self_attn.k_proj.weight")
    v_w = get_tensor_raw(buffer, header, f"{src_prefix}self_attn.v_proj.weight")
    
    # in_proj ينتج 4096 (2048 للمسار + 2048 للبوابة)
    in_proj_weight = np.zeros((d_inner * 2, INPUT_DIM), dtype=np.float32)
    if q_w is not None:
        in_proj_weight[:d_inner, :] = smart_resize(q_w, (d_inner, INPUT_DIM))
    if k_w is not None:
        in_proj_weight[d_inner:, :] = smart_resize(k_w, (d_inner, INPUT_DIM))
    
    weights[f"{dest_prefix}self_attn.in_proj.weight"] = in_proj_weight.astype(np.float16)
    weights[f"{dest_prefix}self_attn.in_proj.bias"] = np.zeros((d_inner * 2,), dtype=np.float16)
    
    # x_proj: [d_state*2 + d_inner, d_inner] = [32 + 2048, 2048] = [2080, 2048]
    weights[f"{dest_prefix}self_attn.x_proj.weight"] = (
        np.random.randn(d_state * 2 + d_inner, d_inner) * 0.001
    ).astype(np.float16)
    weights[f"{dest_prefix}self_attn.x_proj.bias"] = np.zeros((d_state * 2 + d_inner,), dtype=np.float16)
    
    # dt_proj: [d_inner, d_inner] = [2048, 2048]
    weights[f"{dest_prefix}self_attn.dt_proj.weight"] = (
        np.random.randn(d_inner, d_inner) * 0.001
    ).astype(np.float16)
    weights[f"{dest_prefix}self_attn.dt_proj.bias"] = np.zeros((d_inner,), dtype=np.float16)
    
    # out_proj: من o_proj للمعلم
    o_w = get_tensor_raw(buffer, header, f"{src_prefix}self_attn.o_proj.weight")
    weights[f"{dest_prefix}self_attn.out_proj.weight"] = smart_resize(
        o_w, (INPUT_DIM, d_inner), f"out_proj"
    ).astype(np.float16)
    weights[f"{dest_prefix}self_attn.out_proj.bias"] = np.zeros((INPUT_DIM,), dtype=np.float16)
    
    # conv1d: [d_inner, 1, d_conv] = [2048, 1, 4]
    weights[f"{dest_prefix}self_attn.conv1d.weight"] = (
        np.random.randn(d_inner, 1, 4) * 0.01
    ).astype(np.float16)
    weights[f"{dest_prefix}self_attn.conv1d.bias"] = np.zeros((d_inner,), dtype=np.float16)
    
    # a_log: [d_state, d_inner] = [16, 2048] - مهيأة بقيم سالبة صغيرة
    weights[f"{dest_prefix}self_attn.a_log"] = (
        np.ones((d_state, d_inner)) * -0.5
    ).astype(np.float16)
    
    # d: [d_inner] = [2048]
    weights[f"{dest_prefix}self_attn.d"] = np.ones((d_inner,), dtype=np.float16)
    
    return weights


def create_moe_weights(src_prefix, dest_prefix, buffer, header):
    """
    إنشاء أوزان MoE من طبقة MLP للمعلم.
    
    هيكل MoE في Zumar:
    - gate: [NUM_EXPERTS, INPUT_DIM] = [8, 1024]
    - expert_i: [INPUT_DIM, INPUT_DIM] = [1024, 1024] (لكل خبير)
    """
    weights = {}
    
    # gate: من gate_proj للمعلم
    gate_w = get_tensor_raw(buffer, header, f"{src_prefix}mlp.gate_proj.weight")
    weights[f"{dest_prefix}mlp.gate.weight"] = smart_resize(
        gate_w, (NUM_EXPERTS, INPUT_DIM), "moegate"
    ).astype(np.float16)
    weights[f"{dest_prefix}mlp.gate.bias"] = np.zeros((NUM_EXPERTS,), dtype=np.float16)
    
    # كل خبير يحصل على نسخة مشوشة قليلاً من أوزان المعلم
    up_w = get_tensor_raw(buffer, header, f"{src_prefix}mlp.up_proj.weight")
    down_w = get_tensor_raw(buffer, header, f"{src_prefix}mlp.down_proj.weight")
    
    # استخدام up_proj كأساس للخبراء
    base_expert = smart_resize(up_w, (INPUT_DIM, INPUT_DIM), "expert_base")
    
    for e in range(NUM_EXPERTS):
        # إضافة ضوضاء صغيرة لكل خبير لتمييزهم
        noise = np.random.normal(0, 0.0001, (INPUT_DIM, INPUT_DIM))
        expert_weight = (base_expert + noise).astype(np.float16)
        weights[f"{dest_prefix}mlp.expert_{e}.weight"] = expert_weight
        weights[f"{dest_prefix}mlp.expert_{e}.bias"] = np.zeros((INPUT_DIM,), dtype=np.float16)
    
    return weights


def run_distillation():
    """
    عملية التقطير الرئيسية:
    1. تحميل نموذج المعلم
    2. استخراج الأوزان وتحويلها لهيكل ZumarModel
    3. حفظ الملف النهائي
    """
    # إنشاء مجلد الهدف
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # التحقق من وجود المعلم
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ Teacher model missing at: {SOURCE_FILE}")
        print("💡 Please place your teacher model (model.safetensors) in models/teacher/")
        return False
    
    print("=" * 60)
    print("🧬 ZUMAR KNOWLEDGE DISTILLATION v3.0")
    print("=" * 60)
    print(f"📂 Teacher: {SOURCE_FILE}")
    print(f"📂 Target:  {OUTPUT_FILE}")
    print(f"📊 Config:  {NUM_LAYERS} layers, {INPUT_DIM} dim, {NUM_EXPERTS} experts")
    print("-" * 60)
    
    # تحميل المعلم
    buffer, header = load_safetensors_manually(SOURCE_FILE)
    if buffer is None:
        print("❌ Failed to load teacher model!")
        return False
    
    print(f"✅ Teacher loaded: {len(header)} tensors found")
    
    # اكتشاف عدد طبقات المعلم
    teacher_layers = 0
    for key in header.keys():
        if key.startswith("model.layers."):
            layer_num = int(key.split(".")[2])
            teacher_layers = max(teacher_layers, layer_num + 1)
    print(f"📊 Teacher has {teacher_layers} layers")
    
    # --- بناء أوزان Zumar ---
    weights = {}
    
    # 1. التضمين (Embedding)
    print("\n📥 Extracting Embeddings...")
    emb = get_tensor_raw(buffer, header, "model.embed_tokens.weight")
    final_emb = smart_resize(emb, (VOCAB_SIZE, INPUT_DIM), "embed_tokens").astype(np.float16)
    weights["model.embed_tokens.weight"] = final_emb
    
    # 2. رأس اللغة (LM Head) - مشارك مع التضمين
    weights["lm_head.weight"] = final_emb.copy()
    weights["lm_head.bias"] = np.zeros((VOCAB_SIZE,), dtype=np.float16)
    
    # 3. معيار الطبقة النهائية (Final LayerNorm)
    print("📥 Extracting Final Norm...")
    norm_w = get_tensor_raw(buffer, header, "model.norm.weight")
    weights["model.norm.weight"] = smart_resize(norm_w, (INPUT_DIM,), "final_norm").astype(np.float16)
    weights["model.norm.bias"] = np.zeros((INPUT_DIM,), dtype=np.float16)
    
    # 4. الطبقات (Layers)
    print(f"\n📥 Distilling {NUM_LAYERS} layers...")
    for i in range(NUM_LAYERS):
        # تدوير عبر طبقات المعلم إذا كان عدد طبقات Zumar أكبر
        src_idx = i % teacher_layers
        src_p = f"model.layers.{src_idx}."
        dest_p = f"model.layers.{i}."
        
        print(f"  🔄 Layer {i} (from teacher layer {src_idx})")
        
        # --- Mamba Weights ---
        mamba_weights = create_mamba_weights(src_p, dest_p, buffer, header)
        weights.update(mamba_weights)
        
        # --- MoE Weights ---
        moe_weights = create_moe_weights(src_p, dest_p, buffer, header)
        weights.update(moe_weights)
        
        # --- LayerNorm Weights ---
        # input_layernorm
        in_norm = get_tensor_raw(buffer, header, f"{src_p}input_layernorm.weight")
        weights[f"{dest_p}input_layernorm.weight"] = smart_resize(
            in_norm, (INPUT_DIM,), f"layer{i}_in_norm"
        ).astype(np.float16)
        weights[f"{dest_p}input_layernorm.bias"] = np.zeros((INPUT_DIM,), dtype=np.float16)
        
        # post_attention_layernorm
        post_norm = get_tensor_raw(buffer, header, f"{src_p}post_attention_layernorm.weight")
        weights[f"{dest_p}post_attention_layernorm.weight"] = smart_resize(
            post_norm, (INPUT_DIM,), f"layer{i}_post_norm"
        ).astype(np.float16)
        weights[f"{dest_p}post_attention_layernorm.bias"] = np.zeros((INPUT_DIM,), dtype=np.float16)
        
        if i % 5 == 0 or i == NUM_LAYERS - 1:
            print(f"  ✅ Layer {i} complete")
    
    # --- حفظ الملف ---
    print(f"\n💾 Saving {len(weights)} tensors to safetensors...")
    
    try:
        save_file(weights, OUTPUT_FILE)
        print(f"✅ Successfully saved to: {OUTPUT_FILE}")
        
        # طباعة ملخص
        total_params = 0
        for key, val in weights.items():
            params = int(np.prod(val.shape))
            total_params += params
            if "weight" in key and "layer" not in key:
                print(f"  📦 {key}: {val.shape} ({params:,} params)")
        
        print(f"\n📊 Total parameters: {total_params:,}")
        print("🚀 Zumar Brain is ready for inference!")
        return True
        
    except Exception as e:
        print(f"❌ Error saving safetensors: {e}")
        return False


if __name__ == "__main__":
    success = run_distillation()
    if not success:
        exit(1)