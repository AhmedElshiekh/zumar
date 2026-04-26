#!/usr/bin/env python3
"""
Distill from llama.cpp server to Zumar
يستخدم خادم llama.cpp لتجنب إعادة تحميل النموذج كل مرة
"""

import subprocess
import numpy as np
import os
import time
import json
import urllib.request
from pathlib import Path
from safetensors.numpy import save_file

# ============================================================
# إعدادات
# ============================================================
ZUMAR_CONFIG = {
    "vocab_size": 50257,
    "hidden_dim": 1024,
    "num_layers": 12,
    "num_experts": 8,
    "top_k": 2,
    "n_heads": 16,
}

LLAMA_SERVER = "/root/llama.cpp/build/bin/llama-server"
LLAMA_CLI = "/root/llama.cpp/build/bin/llama-cli"
MODEL_PATH = "models/teacher/Bonsai-1.7B.gguf"
SERVER_URL = "http://localhost:8080"

# ============================================================
# 1. تشغيل خادم llama.cpp
# ============================================================
class LlamaServer:
    def __init__(self, server_path, model_path):
        self.server_path = server_path
        self.model_path = model_path
        self.process = None
    
    def start(self):
        """تشغيل الخادم في الخلفية"""
        print("🔄 Starting llama.cpp server...")
        
        cmd = [
            self.server_path,
            "-m", self.model_path,
            "--host", "0.0.0.0",
            "--port", "8080",
            "-t", "1",
            "-ngl", "0",
            "-c", "512",
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # انتظر حتى يبدأ الخادم
            for i in range(30):
                time.sleep(2)
                try:
                    req = urllib.request.Request(f"{SERVER_URL}/health")
                    urllib.request.urlopen(req, timeout=2)
                    print("✅ Server ready!")
                    return True
                except:
                    print(f"   Waiting... ({i+1}/30)")
            
            print("❌ Server failed to start")
            return False
            
        except Exception as e:
            print(f"❌ Cannot start server: {e}")
            return False
    
    def stop(self):
        """إيقاف الخادم"""
        if self.process:
            print("🛑 Stopping server...")
            self.process.terminate()
            self.process.wait()
            print("✅ Server stopped")
    
    def generate(self, prompt, max_tokens=3):
        """استدعاء الخادم عبر API"""
        data = json.dumps({
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.7,
        }).encode('utf-8')
        
        try:
            req = urllib.request.Request(
                f"{SERVER_URL}/completion",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            response = urllib.request.urlopen(req, timeout=30)
            result = json.loads(response.read())
            return result.get("content", "").strip()
        except Exception as e:
            print(f"      ⚠️  Server error: {e}")
            return ""

# ============================================================
# 2. LlamaTeacher (يستخدم الخادم)
# ============================================================
class LlamaTeacher:
    def __init__(self, server):
        self.server = server
    
    def get_logits(self, prompt):
        """محاكاة logits من النص المولد"""
        output = self.server.generate(prompt, max_tokens=1)
        
        logits = np.zeros(ZUMAR_CONFIG["vocab_size"], dtype=np.float32)
        
        for char in output:
            token_id = ord(char) % ZUMAR_CONFIG["vocab_size"]
            logits[token_id] += 1.0
        
        if logits.sum() == 0:
            logits[0] = 1.0
        else:
            logits = logits + 0.1
        
        return logits / logits.sum()

# ============================================================
# 3. نموذج Zumar
# ============================================================
class ZumarStudent:
    def __init__(self):
        v = ZUMAR_CONFIG["vocab_size"]
        h = ZUMAR_CONFIG["hidden_dim"]
        n = ZUMAR_CONFIG["num_layers"]
        
        print(f"   Creating model: {v} vocab, {h} dim, {n} layers...")
        
        self.emb = np.random.randn(v, h).astype(np.float32) * 0.02
        self.layers = []
        
        for i in range(n):
            layer = {
                'w': np.random.randn(h, h).astype(np.float32) * 0.02,
            }
            self.layers.append(layer)
        
        self.head = np.random.randn(h, v).astype(np.float32) * 0.02
        print(f"   Model created.")
    
    def forward(self, token_id):
        x = self.emb[token_id].copy()
        for layer in self.layers:
            x = x @ layer['w']
            x = np.maximum(x, 0)
        return x @ self.head
    
    def train_step(self, token_id, teacher_logits, lr=0.001):
        student_logits = self.forward(token_id)
        
        student_max = student_logits.max()
        student_probs = np.exp(student_logits - student_max)
        student_probs = student_probs / (student_probs.sum() + 1e-9)
        
        teacher_probs = teacher_logits / (teacher_logits.sum() + 1e-9)
        
        eps = 1e-9
        loss = np.sum(teacher_probs * np.log((teacher_probs + eps) / (student_probs + eps)))
        
        grad = student_probs - teacher_probs
        self.head -= lr * np.outer(self.emb[token_id], grad)
        self.emb[token_id] -= lr * (grad @ self.head.T)
        
        return loss

# ============================================================
# 4. حفظ الأوزان
# ============================================================
def save_weights(student):
    weights = {}
    h = ZUMAR_CONFIG["hidden_dim"]
    v = ZUMAR_CONFIG["vocab_size"]
    
    weights["model.embed_tokens.weight"] = student.emb.astype(np.float16)
    weights["lm_head.weight"] = student.head.T.astype(np.float16)
    weights["lm_head.bias"] = np.zeros(v, dtype=np.float16)
    weights["model.norm.weight"] = np.ones(h, dtype=np.float16)
    weights["model.norm.bias"] = np.zeros(h, dtype=np.float16)
    
    for i, layer in enumerate(student.layers):
        p = f"model.layers.{i}"
        w = layer['w'].astype(np.float16)
        
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weights[f"{p}.self_attn.{proj}.weight"] = w
            weights[f"{p}.self_attn.{proj}.bias"] = np.zeros(h, dtype=np.float16)
        
        weights[f"{p}.mlp.gate.weight"] = np.zeros((8, h), dtype=np.float16)
        weights[f"{p}.mlp.gate.bias"] = np.zeros(8, dtype=np.float16)
        for e in range(8):
            weights[f"{p}.mlp.expert_{e}.weight"] = w
            weights[f"{p}.mlp.expert_{e}.bias"] = np.zeros(h, dtype=np.float16)
        
        weights[f"{p}.input_layernorm.weight"] = np.ones(h, dtype=np.float16)
        weights[f"{p}.input_layernorm.bias"] = np.zeros(h, dtype=np.float16)
        weights[f"{p}.post_attention_layernorm.weight"] = np.ones(h, dtype=np.float16)
        weights[f"{p}.post_attention_layernorm.bias"] = np.zeros(h, dtype=np.float16)
    
    output = Path("models/zumar-v1/model.safetensors")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(output))
    
    size_mb = output.stat().st_size / 1_048_576
    print(f"✅ Saved ({size_mb:.1f} MB)")
    return output

# ============================================================
# 5. التقطير
# ============================================================
def distill():
    print("=" * 60)
    print("🧠 DISTILLING FROM LLAMA.CPP SERVER TO ZUMAR")
    print("=" * 60)
    
    # تشغيل الخادم
    server = LlamaServer(LLAMA_SERVER, MODEL_PATH)
    
    if not server.start():
        print("❌ Cannot start server. Trying CLI mode...")
        return distill_cli_mode()
    
    try:
        teacher = LlamaTeacher(server)
        student = ZumarStudent()
        
        prompts = [
            "Hello", "How are you?", "The weather is",
            "I think", "What is", "The answer is",
            "My name", "Today I", "I love",
            "Please", "Thank you", "Good morning",
        ]
        
        print(f"\n🎓 Starting distillation...")
        print(f"   Prompts: {len(prompts)}")
        
        step_count = 0
        
        for epoch in range(10):
            total_loss = 0
            
            for prompt in prompts:
                print(f"   [{epoch+1}/10] '{prompt}'")
                
                try:
                    teacher_logits = teacher.get_logits(prompt)
                except:
                    continue
                
                for char in prompt:
                    token_id = ord(char) % ZUMAR_CONFIG["vocab_size"]
                    
                    try:
                        loss = student.train_step(token_id, teacher_logits, lr=0.001)
                        total_loss += loss
                        step_count += 1
                        
                        if step_count % 50 == 0:
                            print(f"      Step {step_count}, Loss: {loss:.4f}")
                    except:
                        continue
            
            avg_loss = total_loss / max(1, step_count)
            print(f"   ✅ Epoch {epoch+1}: loss={avg_loss:.4f}\n")
        
        save_weights(student)
        
    finally:
        server.stop()
    
    print(f"\n🚀 Run: cargo run -p core --release")

def distill_cli_mode():
    """وضع احتياطي: llama-cli مع مهلة أطول"""
    print("\n⚠️  Server mode failed. Trying CLI with 60s timeout...")
    
    teacher_cmd = LlamaTeacherCLI(LLAMA_CLI, MODEL_PATH)
    student = ZumarStudent()
    
    prompts = [
        "Hello", "How are you?", "The weather is",
        "I think", "What is", "The answer is",
        "My name", "Today I", "I love",
        "Please", "Thank you", "Good morning",
    ]
    
    step_count = 0
    
    for epoch in range(5):
        total_loss = 0
        
        for prompt in prompts:
            print(f"   [{epoch+1}/5] '{prompt}'")
            
            try:
                teacher_logits = teacher_cmd.get_logits(prompt)
            except:
                continue
            
            for char in prompt:
                token_id = ord(char) % ZUMAR_CONFIG["vocab_size"]
                try:
                    loss = student.train_step(token_id, teacher_logits, lr=0.001)
                    total_loss += loss
                    step_count += 1
                except:
                    continue
        
        avg_loss = total_loss / max(1, step_count)
        print(f"   ✅ Epoch {epoch+1}: loss={avg_loss:.4f}\n")
    
    save_weights(student)
    print(f"\n🚀 Run: cargo run -p core --release")

class LlamaTeacherCLI:
    def __init__(self, cli_path, model_path):
        self.cli_path = cli_path
        self.model_path = model_path
    
    def generate(self, prompt, max_tokens=3):
        cmd = [
            self.cli_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", "1",
            "--no-display-prompt",
            "--temp", "0.7",
            "--repeat-penalty", "1.0",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            output = result.stdout.strip()
            if prompt in output:
                output = output.split(prompt)[-1].strip()
            return output[:10] if output else "a"
        except:
            return "a"
    
    def get_logits(self, prompt):
        output = self.generate(prompt, max_tokens=1)
        logits = np.zeros(ZUMAR_CONFIG["vocab_size"], dtype=np.float32)
        for char in output:
            token_id = ord(char) % ZUMAR_CONFIG["vocab_size"]
            logits[token_id] += 1.0
        if logits.sum() == 0:
            logits[0] = 1.0
        else:
            logits = logits + 0.1
        return logits / logits.sum()

# ============================================================
# تشغيل
# ============================================================
if __name__ == "__main__":
    distill()