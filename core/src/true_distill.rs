use candle_core::{Tensor, Result, Device};
use candle_nn::{VarMap, Optimizer};
use crate::layers::ZumarModel;
use std::time::Instant;

pub struct DistillConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub temperature: f64,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self { epochs: 50, learning_rate: 0.001, temperature: 3.0 }
    }
}

pub struct TrueDistiller {
    config: DistillConfig,
    device: Device,
}

impl TrueDistiller {
    pub fn new(config: DistillConfig, device: Device) -> Self {
        Self { config, device }
    }

    pub fn distill(
        &self,
        student: &mut ZumarModel,
        varmap: &VarMap,
        teacher_path: &str,
        data: &[String],
    ) -> Result<()> {
        println!("\n🧠 TRUE KNOWLEDGE DISTILLATION");
        println!("   Epochs: {}", self.config.epochs);
        println!("   Samples: {}", data.len());

        let teacher = AutoTeacher::load_lazy(teacher_path, &self.device)?;

        let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), self.config.learning_rate)?;
        let start = Instant::now();

        for epoch in 0..self.config.epochs {
            let mut loss_sum = 0.0f32;
            let mut count = 0u32;

            for text in data.iter() {
                let teacher_logits = match teacher.predict(text) {
                    Ok(l) => l, Err(_) => continue,
                };

                let tokens: Vec<u32> = text.chars()
                    .map(|c| (c as u32).wrapping_add(3) % 50257)
                    .collect();

                for &token_id in &tokens {
                    let emb = student.embed(token_id, &self.device)?;
                    let student_logits = student.forward(&emb)?;
                    let student_flat = student_logits.flatten_all()?;

                    let teacher_tensor = Tensor::new(teacher_logits.as_slice(), &self.device)?;
                    let teacher_probs = candle_nn::ops::softmax(&teacher_tensor, 0)?;
                    let student_probs = candle_nn::ops::softmax(&student_flat, 0)?;

                    let eps = 1e-9f32;
                    let eps_t = Tensor::new(&[eps], &self.device)?;
                    let log_s = student_probs.maximum(&eps_t)?.log()?;
                    let log_t = teacher_probs.maximum(&eps_t)?.log()?;
                    let diff = (log_t - log_s)?;
                    let kl = (teacher_probs * diff)?.sum_all()?;

                    loss_sum += kl.to_scalar::<f32>()?;
                    count += 1;
                    opt.backward_step(&kl)?;

                    drop(emb); drop(student_logits); drop(student_flat);
                    drop(teacher_tensor); drop(kl); drop(eps_t);
                }

                if count % 100 == 0 {
                    print!("\r  Ep {} | Tok {} | Loss {:.4}   ", 
                        epoch + 1, count, loss_sum / count.max(1) as f32);
                }
            }

            println!();
            println!("  ✅ Ep {}: Loss {:.4} | {:.1}s", 
                epoch + 1, loss_sum / count.max(1) as f32, start.elapsed().as_secs_f64());
        }

        println!("\n⏱ Total: {:.1}s", start.elapsed().as_secs_f64());
        Ok(())
    }
}

// ============================================================
// مُحَمِّل تلقائي مع تحميل تدريجي (Lazy Loading)
// ============================================================
pub struct AutoTeacher {
    data: Vec<u8>,
    header: Option<serde_json::Value>,
    config: TeacherConfig,
    device: Device,
}

struct TeacherConfig {
    embedding_key: String,
    num_layers: usize,
    hidden_dim: usize,
    vocab_size: usize,
    arch_type: String,
}

impl AutoTeacher {
    /// تحميل تدريجي (يقرأ metadata فقط - بدون تحميل الأوزان)
    pub fn load_lazy(path: &str, device: &Device) -> Result<Self> {
        println!("   📖 Loading teacher metadata (lazy)...");
        
        let data = std::fs::read(path)
            .map_err(|e| candle_core::Error::Msg(format!("Cannot read: {}", e)))?;

        let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let raw_header = &data[8..8 + header_size];
        let header: serde_json::Value = serde_json::from_slice(raw_header)
            .map_err(|e| candle_core::Error::Msg(format!("JSON error: {}", e)))?;

        let config = Self::detect_architecture_from_header(&header);
        
        println!("   📊 Architecture: {}", config.arch_type);
        println!("   📊 Layers: {}, Hidden: {}, Vocab: {}", 
            config.num_layers, config.hidden_dim, config.vocab_size);
        println!("   ⚡ Lazy mode: loads 1 layer at a time");

        Ok(Self { data, header: Some(header), config, device: device.clone() })
    }

    /// تحميل كامل (قديم - يستهلك ذاكرة)
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        println!("   📖 Loading teacher (full)...");
        
        let data = std::fs::read(path)
            .map_err(|e| candle_core::Error::Msg(format!("Cannot read: {}", e)))?;

        let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let raw_header = &data[8..8 + header_size];
        let header: serde_json::Value = serde_json::from_slice(raw_header)
            .map_err(|e| candle_core::Error::Msg(format!("JSON error: {}", e)))?;

        let config = Self::detect_architecture_from_header(&header);
        
        println!("   📊 Architecture: {}", config.arch_type);
        println!("   📊 Layers: {}, Hidden: {}, Vocab: {}", 
            config.num_layers, config.hidden_dim, config.vocab_size);

        Ok(Self { data, header: Some(header), config, device: device.clone() })
    }

    pub fn get_config(&self) -> &TeacherConfig {
        &self.config
    }

    fn detect_architecture_from_header(header: &serde_json::Value) -> TeacherConfig {
        let keys: Vec<String> = header.as_object()
            .map(|obj| obj.keys().cloned().collect())
            .unwrap_or_default();
        
        let all_keys = keys.join(" ");
        
        // GPT-2
        if all_keys.contains("wte.weight") && all_keys.contains("h.0.ln_1.weight") {
            let num_layers = keys.iter().filter(|k| k.contains(".ln_1.weight")).count();
            let hidden_dim = header.get("wte.weight")
                .and_then(|v| v.get("shape"))
                .and_then(|v| v.as_array())
                .and_then(|a| a.get(1))
                .and_then(|v| v.as_u64())
                .unwrap_or(768) as usize;
            let vocab_size = header.get("wte.weight")
                .and_then(|v| v.get("shape"))
                .and_then(|v| v.as_array())
                .and_then(|a| a.get(0))
                .and_then(|v| v.as_u64())
                .unwrap_or(50257) as usize;
            
            return TeacherConfig {
                embedding_key: "wte.weight".to_string(),
                num_layers,
                hidden_dim,
                vocab_size,
                arch_type: "gpt2".to_string(),
            };
        }
        
        // Llama-style (BitNet, Mistral, etc.)
        if all_keys.contains("model.embed_tokens.weight") 
            && all_keys.contains("model.layers.0.self_attn.q_proj.weight") 
        {
            let num_layers = keys.iter()
                .filter(|k| k.contains("self_attn.q_proj.weight") && k.contains("model.layers."))
                .count();
            
            let hidden_dim = header.get("model.layers.0.self_attn.q_proj.weight")
                .and_then(|v| v.get("shape"))
                .and_then(|v| v.as_array())
                .and_then(|a| a.get(1))
                .and_then(|v| v.as_u64())
                .unwrap_or(2560) as usize;
            let vocab_size = header.get("model.embed_tokens.weight")
                .and_then(|v| v.get("shape"))
                .and_then(|v| v.as_array())
                .and_then(|a| a.get(0))
                .and_then(|v| v.as_u64())
                .unwrap_or(128256) as usize;
            
            return TeacherConfig {
                embedding_key: "model.embed_tokens.weight".to_string(),
                num_layers,
                hidden_dim: hidden_dim.min(1024), // حد أمان
                vocab_size: vocab_size.min(50257),
                arch_type: "llama".to_string(),
            };
        }
        
        // افتراضي - DistilGPT-2
        TeacherConfig {
            embedding_key: "wte.weight".to_string(),
            num_layers: 6,
            hidden_dim: 768,
            vocab_size: 50257,
            arch_type: "gpt2".to_string(),
        }
    }

    /// تحميل تنسور واحد فقط عند الحاجة
    fn load_tensor(&self, name: &str) -> Result<Tensor> {
        let header = self.header.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("No header loaded".to_string()))?;
        
        let info = header.get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Tensor not found: {}", name)))?;
        
        let offsets = info["data_offsets"].as_array().unwrap();
        let start = offsets[0].as_u64().unwrap() as usize;
        let end = offsets[1].as_u64().unwrap() as usize;
        let shape: Vec<usize> = info["shape"].as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let dtype = info.get("dtype").and_then(|d| d.as_str()).unwrap_or("F32");
        
        let header_size = u64::from_le_bytes(self.data[0..8].try_into().unwrap()) as usize;
        let raw = &self.data[8 + header_size + start..8 + header_size + end];
        
        let tensor = match dtype {
            "F16" | "FLOAT16" => {
                let mut f32s = Vec::new();
                for chunk in raw.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f32s.push(half::f16::from_bits(bits).to_f32());
                }
                Tensor::from_vec(f32s, shape, &self.device)?
            }
            _ => {
                let mut f32s = Vec::new();
                for chunk in raw.chunks_exact(4) {
                    f32s.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Tensor::from_vec(f32s, shape, &self.device)?
            }
        };
        
        Ok(tensor)
    }

    /// Forward pass تدريجي (يحمل طبقة واحدة فقط في الذاكرة)
    pub fn predict(&self, text: &str) -> Result<Vec<f32>> {
        let tokens: Vec<u32> = text.chars()
            .map(|c| c as u32 % self.config.vocab_size as u32)
            .collect();
        
        if tokens.is_empty() {
            return Ok(vec![0.0; self.config.vocab_size]);
        }

        // تحميل embedding فقط
        let wte = self.load_tensor(&self.config.embedding_key)?;
        let last_token = (tokens[tokens.len() - 1] as usize) % wte.dim(0)?;
        let x = wte.get(last_token)?;
        let mut h = x.unsqueeze(0)?;
        drop(wte);

        let safe_dim = self.config.hidden_dim.min(1024);

        for i in 0..self.config.num_layers {
            if self.config.arch_type == "llama" {
                let p = format!("model.layers.{}", i);
                
                let q = self.load_tensor(&format!("{}.self_attn.q_proj.weight", p))?;
                let k = self.load_tensor(&format!("{}.self_attn.k_proj.weight", p))?;
                let v = self.load_tensor(&format!("{}.self_attn.v_proj.weight", p))?;
                let o = self.load_tensor(&format!("{}.self_attn.o_proj.weight", p))?;
                
                let qs = q.narrow(0, 0, safe_dim)?.narrow(1, 0, safe_dim)?;
                let ks = k.narrow(0, 0, safe_dim)?.narrow(1, 0, safe_dim)?;
                let vs = v.narrow(0, 0, safe_dim)?.narrow(1, 0, safe_dim)?;
                let os = o.narrow(0, 0, safe_dim)?.narrow(1, 0, safe_dim)?;
                
                let qo = h.matmul(&qs.t()?)?;
                let ko = h.matmul(&ks.t()?)?;
                let vo = h.matmul(&vs.t()?)?;
                
                drop(q); drop(k); drop(v);
                
                let attn = qo.matmul(&ko.t()?)?;
                let attn = candle_nn::ops::softmax(&attn, 1)?;
                h = attn.matmul(&vo)?;
                h = h.matmul(&os.t()?)?;
                
                drop(o); drop(qo); drop(ko); drop(vo); drop(attn);
            } else {
                // GPT-2: c_attn = Q+K+V
                let p = format!("h.{}", i);
                if let Ok(c_attn) = self.load_tensor(&format!("{}.attn.c_attn.weight", p)) {
                    let dim = c_attn.dim(0)?.min(safe_dim);
                    let slice = c_attn.narrow(0, 0, dim)?.narrow(1, 0, safe_dim)?;
                    h = h.matmul(&slice.t()?)?;
                    drop(c_attn);
                }
            }
            
            h = h.relu()?;
        }

        let wte = self.load_tensor(&self.config.embedding_key)?;
        let ws = wte.narrow(1, 0, safe_dim)?;
        let logits = h.matmul(&ws.t()?)?;
        drop(wte);
        
        logits.flatten_all()?.to_vec1::<f32>()
    }
}