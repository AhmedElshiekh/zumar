import numpy as np
from safetensors.numpy import save_file
import os

# المسار الصحيح للوصول من core/injection إلى جذر المشروع
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET_DIR = os.path.join(BASE_DIR, "models", "zumar-v1")

def quantize_1_58_bit(w):
    scale = np.mean(np.abs(w)) + 1e-7
    return np.clip(np.round(w / scale), -1, 1).astype(np.int8)

def run_injection():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📁 Created directory: {TARGET_DIR}")

    zumar_weights = {}
    hidden_size = 1024

    # الحقن لـ 100 طبقة (تطابقاً مع main.rs)
    for i in range(100):
        # محاكاة وزن "معالج" (Logic)
        w = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        zumar_weights[f"layers.{i}.weight"] = quantize_1_58_bit(w)

    # أوزان الـ Embedding والـ Head (يجب أن تظل Float32)
    zumar_weights["embed.weight"] = (np.random.randn(50257, hidden_size) * 0.02).astype(np.float32)
    zumar_weights["project_head.weight"] = (np.random.randn(50257, hidden_size) * 0.02).astype(np.float32)

    output_file = os.path.join(TARGET_DIR, "model.safetensors")
    save_file(zumar_weights, output_file)
    print(f"✅ Sovereign Intelligence Injected into: {output_file}")

if __name__ == "__main__":
    run_injection()
