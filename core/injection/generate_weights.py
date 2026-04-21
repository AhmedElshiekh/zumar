import numpy as np
from safetensors.numpy import save_file
import os
import gc

# الإعدادات الإجبارية
NUM_LAYERS = 30
INPUT_DIM = 1024
NUM_EXPERTS = 8
EXPERT_INNER_DIM = 1024
VOCAB_SIZE = 50257

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET_DIR = os.path.join(BASE_DIR, "models", "zumar-v1")
OUTPUT_FILE = os.path.join(TARGET_DIR, "model.safetensors")

# تعديل المنطق ليدعم F16 ✅
def bitnet_logic_f16(w):
    scale = np.mean(np.abs(w)) + 1e-7
    # تحويل مباشر لـ f16 لتوفير الرام فوراً
    return np.clip(np.round(w / scale), -1, 1).astype(np.float16)

def generate_chunked_weights():
    if not os.path.exists(TARGET_DIR): os.makedirs(TARGET_DIR)
    
    final_weights = {}

    print("💎 Generating Embeddings & Head (F16)...")
    # تحويل كافة الأوزان الضخمة إلى f16 ✅
    final_weights["model.embed_tokens.weight"] = (np.random.normal(0, 0.01, (VOCAB_SIZE, INPUT_DIM))).astype(np.float16)
    final_weights["lm_head.weight"] = (np.random.normal(0, 0.01, (VOCAB_SIZE, INPUT_DIM))).astype(np.float16)
    final_weights["lm_head.bias"] = np.zeros((VOCAB_SIZE,)).astype(np.float16)
    final_weights["model.norm.weight"] = np.ones((INPUT_DIM,)).astype(np.float16)
    final_weights["model.norm.bias"] = np.zeros((INPUT_DIM,)).astype(np.float16)

    for i in range(NUM_LAYERS):
        print(f"🏗️ Processing Layer {i+1}/{NUM_LAYERS} [F16 Mode]...")
        
        p = f"model.layers.{i}.self_attn."
        # استخدام f16 في كل المصفوفات ✅
        final_weights[p + "in_proj.weight"] = bitnet_logic_f16(np.random.randn(4096, INPUT_DIM))
        final_weights[p + "in_proj.bias"] = np.zeros((4096,)).astype(np.float16)
        final_weights[p + "conv1d.weight"] = np.random.randn(2048, 1, 4).astype(np.float16)
        final_weights[p + "conv1d.bias"] = np.zeros((2048,)).astype(np.float16)
        final_weights[p + "x_proj.weight"] = bitnet_logic_f16(np.random.randn(2080, 2048))
        final_weights[p + "x_proj.bias"] = np.zeros((2080,)).astype(np.float16)
        final_weights[p + "dt_proj.weight"] = bitnet_logic_f16(np.random.randn(2048, 2048))
        final_weights[p + "dt_proj.bias"] = np.zeros((2048,)).astype(np.float16)
        final_weights[p + "a_log"] = np.log(np.random.uniform(1, 5, (16, 2048))).astype(np.float16)
        final_weights[p + "d"] = np.ones((2048,)).astype(np.float16)
        final_weights[p + "out_proj.weight"] = bitnet_logic_f16(np.random.randn(INPUT_DIM, 2048))
        final_weights[p + "out_proj.bias"] = np.zeros((INPUT_DIM,)).astype(np.float16)

        final_weights[f"model.layers.{i}.mlp.gate.weight"] = bitnet_logic_f16(np.random.randn(NUM_EXPERTS, INPUT_DIM))
        for e in range(NUM_EXPERTS):
            # توفير هائل هنا: المصفوفة أصبحت 2MB بدلاً من 4MB ✅
            final_weights[f"model.layers.{i}.mlp.expert_{e}.weight"] = bitnet_logic_f16(np.random.randn(EXPERT_INNER_DIM, INPUT_DIM))
        
        final_weights[f"model.layers.{i}.input_layernorm.weight"] = np.ones((INPUT_DIM,)).astype(np.float16)
        final_weights[f"model.layers.{i}.input_layernorm.bias"] = np.zeros((INPUT_DIM,)).astype(np.float16)
        final_weights[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones((INPUT_DIM,)).astype(np.float16)
        final_weights[f"model.layers.{i}.post_attention_layernorm.bias"] = np.zeros((INPUT_DIM,)).astype(np.float16)

        if i % 3 == 0:
            gc.collect()

    print(f"💾 Saving F16 weights to {OUTPUT_FILE}...")
    save_file(final_weights, OUTPUT_FILE)
    print(f"✅ Success! File size should be ~50% smaller.")

if __name__ == "__main__":
    generate_chunked_weights()
