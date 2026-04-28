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

        let teacher = AutoTeacher::load(teacher_path, &self.device)?;

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
// مُحَمِّل تلقائي مع Forward Pass ذكي
// ============================================================
pub struct AutoTeacher {
    weights: std::collections::HashMap<String, Tensor>,
    config: TeacherConfig,
    device: Device,
}

struct TeacherConfig {
    embedding_key: String,
    num_layers: usize,
    hidden_dim: usize,
    vocab_size: usize,
    arch_type: String,
    layer_patterns: LayerPatterns,
}

struct LayerPatterns {
    ln1_weight: Option<String>,
    ln1_bias: Option<String>,
    ln2_weight: Option<String>,
    ln2_bias: Option<String>,
    attn_q: Option<String>,
    attn_k: Option<String>,
    attn_v: Option<String>,
    attn_qkv: Option<String>,
    attn_o: Option<String>,
    mlp_gate: Option<String>,
    mlp_up: Option<String>,
    mlp_down: Option<String>,
    mlp_fc: Option<String>,
    mlp_proj: Option<String>,
    final_ln_weight: Option<String>,
    final_ln_bias: Option<String>,
}

impl LayerPatterns {
    fn new() -> Self {
        Self {
            ln1_weight: None, ln1_bias: None,
            ln2_weight: None, ln2_bias: None,
            attn_q: None, attn_k: None, attn_v: None,
            attn_qkv: None, attn_o: None,
            mlp_gate: None, mlp_up: None, mlp_down: None,
            mlp_fc: None, mlp_proj: None,
            final_ln_weight: None, final_ln_bias: None,
        }
    }
}

impl AutoTeacher {
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| candle_core::Error::Msg(format!("Cannot read: {}", e)))?;

        let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let raw_header = &data[8..8 + header_size];
        let header: serde_json::Value = serde_json::from_slice(raw_header)
            .map_err(|e| candle_core::Error::Msg(format!("JSON error: {}", e)))?;

        let mut weights = std::collections::HashMap::new();

        if let serde_json::Value::Object(obj) = &header {
            for (name, info) in obj {
                if name == "__metadata__" { continue; }

                let offsets = info["data_offsets"].as_array().unwrap();
                let start = offsets[0].as_u64().unwrap() as usize;
                let end = offsets[1].as_u64().unwrap() as usize;
                let shape: Vec<usize> = info["shape"].as_array().unwrap()
                    .iter().map(|v: &serde_json::Value| v.as_u64().unwrap() as usize).collect();
                let dtype = info.get("dtype")
                    .and_then(|d: &serde_json::Value| d.as_str()).unwrap_or("F32");
                let raw = &data[8 + header_size + start..8 + header_size + end];

                let tensor = match dtype {
                    "F16" | "FLOAT16" => {
                        let mut f32s = Vec::new();
                        for chunk in raw.chunks_exact(2) {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            f32s.push(half::f16::from_bits(bits).to_f32());
                        }
                        Tensor::from_vec(f32s, shape, device)?
                    }
                    _ => {
                        let mut f32s = Vec::new();
                        for chunk in raw.chunks_exact(4) {
                            f32s.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                        }
                        Tensor::from_vec(f32s, shape, device)?
                    }
                };
                weights.insert(name.clone(), tensor);
            }
        }

        let config = Self::detect_architecture(&weights);
        println!("   📊 Architecture: {}", config.arch_type);
        println!("   📊 Layers: {}, Hidden: {}, Vocab: {}", 
            config.num_layers, config.hidden_dim, config.vocab_size);
        println!("   ✅ {} tensors loaded", weights.len());

        Ok(Self { weights, config, device: device.clone() })
    }

    fn detect_architecture(weights: &std::collections::HashMap<String, Tensor>) -> TeacherConfig {
        let keys: Vec<&str> = weights.keys().map(|k| k.as_str()).collect();
        let all_keys = keys.join(" ");

        // اكتشف صيغة الطبقة بذكاء
        let mut patterns = LayerPatterns::new();
        
        // البحث عن صيغ LayerNorm
        for key in &keys {
            if key.ends_with(".ln_1.weight") || key.ends_with(".input_layernorm.weight") {
                patterns.ln1_weight = Some(key.to_string());
                patterns.ln1_bias = Some(key.replace("weight", "bias"));
                patterns.ln2_weight = Some(key.replace("ln_1", "ln_2").replace("input", "post_attention"));
                patterns.ln2_bias = Some(patterns.ln2_weight.as_ref().unwrap().replace("weight", "bias"));
            }
        }

        // GPT-2
        if all_keys.contains("wte.weight") && all_keys.contains("h.0.ln_1.weight") {
            let num_layers = keys.iter().filter(|k| k.contains(".ln_1.weight")).count();
            let hidden_dim = weights.get("wte.weight").map(|t| t.dim(1).unwrap_or(768)).unwrap_or(768);
            let vocab_size = weights.get("wte.weight").map(|t| t.dim(0).unwrap_or(50257)).unwrap_or(50257);
            
            patterns.attn_qkv = Some("h.{i}.attn.c_attn.weight".to_string());
            patterns.attn_o = Some("h.{i}.attn.c_proj.weight".to_string());
            patterns.mlp_fc = Some("h.{i}.mlp.c_fc.weight".to_string());
            patterns.mlp_proj = Some("h.{i}.mlp.c_proj.weight".to_string());
            patterns.final_ln_weight = Some("ln_f.weight".to_string());
            patterns.final_ln_bias = Some("ln_f.bias".to_string());
            
            return TeacherConfig {
                embedding_key: "wte.weight".to_string(),
                num_layers, hidden_dim, vocab_size,
                arch_type: "gpt2".to_string(),
                layer_patterns: patterns,
            };
        }

        // Llama-style
        if all_keys.contains("model.embed_tokens.weight") 
            && all_keys.contains("model.layers.0.self_attn.q_proj.weight") 
        {
            let num_layers = keys.iter()
                .filter(|k| k.contains("self_attn.q_proj.weight") && k.contains("model.layers."))
                .count();
            let emb = weights.get("model.embed_tokens.weight").unwrap();
            let hidden_dim = emb.dim(1).unwrap_or(4096);
            let vocab_size = emb.dim(0).unwrap_or(32000);
            
            patterns.attn_q = Some("model.layers.{i}.self_attn.q_proj.weight".to_string());
            patterns.attn_k = Some("model.layers.{i}.self_attn.k_proj.weight".to_string());
            patterns.attn_v = Some("model.layers.{i}.self_attn.v_proj.weight".to_string());
            patterns.attn_o = Some("model.layers.{i}.self_attn.o_proj.weight".to_string());
            patterns.mlp_gate = Some("model.layers.{i}.mlp.gate_proj.weight".to_string());
            patterns.mlp_up = Some("model.layers.{i}.mlp.up_proj.weight".to_string());
            patterns.mlp_down = Some("model.layers.{i}.mlp.down_proj.weight".to_string());
            patterns.final_ln_weight = Some("model.norm.weight".to_string());
            patterns.final_ln_bias = Some("model.norm.bias".to_string());
            
            return TeacherConfig {
                embedding_key: "model.embed_tokens.weight".to_string(),
                num_layers, hidden_dim, vocab_size,
                arch_type: "llama".to_string(),
                layer_patterns: patterns,
            };
        }

        // افتراضي - اكتشف كل شيء
        let emb_key: String = keys.iter()
            .find(|k| k.contains("embed") && k.contains("weight"))
            .map(|&k| k.to_string())
            .unwrap_or_else(|| "wte.weight".to_string());
        let emb = weights.get(&emb_key);
        let hidden_dim = emb.map(|t| t.dim(1).unwrap_or(768)).unwrap_or(768);
        let vocab_size = emb.map(|t| t.dim(0).unwrap_or(50257)).unwrap_or(50257);
        
        // اكتشف كل الصيغ الممكنة من الطبقة 0
        for key in &keys {
            let key_str = key.to_string();
            if key_str.contains(".0.") || key_str.contains(".0_") {
                let generic = key_str
                    .replace(".0.", ".{i}.")
                    .replace("_0_", "_{i}_")
                    .replace("h.0", "h.{i}")
                    .replace("layer.0", "layer.{i}")
                    .replace("layers.0", "layers.{i}")
                    .replace("blk.0.", "blk.{i}.");
                
                if key_str.contains("attn") || key_str.contains("attention") {
                    if key_str.contains("q_proj") || key_str.contains("query") {
                        if patterns.attn_q.is_none() { patterns.attn_q = Some(generic.clone()); }
                    }
                    if key_str.contains("k_proj") || key_str.contains("key") {
                        if patterns.attn_k.is_none() { patterns.attn_k = Some(generic.clone()); }
                    }
                    if key_str.contains("v_proj") || key_str.contains("value") {
                        if patterns.attn_v.is_none() { patterns.attn_v = Some(generic.clone()); }
                    }
                    if key_str.contains("o_proj") || key_str.contains("output") || key_str.contains("c_proj") {
                        if patterns.attn_o.is_none() { patterns.attn_o = Some(generic.clone()); }
                    }
                    if key_str.contains("qkv") || key_str.contains("c_attn") {
                        if patterns.attn_qkv.is_none() { patterns.attn_qkv = Some(generic.clone()); }
                    }
                }
                if key_str.contains("mlp") || key_str.contains("ffn") || key_str.contains("feed") {
                    if key_str.contains("gate") || key_str.contains("c_fc") {
                        if patterns.mlp_fc.is_none() { patterns.mlp_fc = Some(generic.clone()); }
                    }
                    if key_str.contains("up") || key_str.contains("c_fc") {
                        if patterns.mlp_up.is_none() { patterns.mlp_up = Some(generic.clone()); }
                    }
                    if key_str.contains("down") || key_str.contains("c_proj") || key_str.contains("proj") {
                        if patterns.mlp_proj.is_none() { patterns.mlp_proj = Some(generic.clone()); }
                    }
                }
            }
        }

        TeacherConfig {
            embedding_key: emb_key,
            num_layers: 6,
            hidden_dim,
            vocab_size,
            arch_type: "auto".to_string(),
            layer_patterns: patterns,
        }
    }

    /// Forward pass ذكي يتكيف مع أي معمارية
    pub fn predict(&self, text: &str) -> Result<Vec<f32>> {
        let tokens: Vec<u32> = text.chars().map(|c| c as u32 % self.config.vocab_size as u32).collect();
        if tokens.is_empty() {
            return Ok(vec![0.0; self.config.vocab_size]);
        }

        let wte = self.weights.get(&self.config.embedding_key)
            .ok_or_else(|| candle_core::Error::Msg(format!("No embedding: {}", self.config.embedding_key)))?;

        let last_token = (tokens[tokens.len() - 1] as usize) % wte.dim(0)?;
        let x = wte.get(last_token)?;
        let mut h = x.unsqueeze(0)?;

        let p = &self.config.layer_patterns;

        for i in 0..self.config.num_layers {
            let residual = h.clone();

            // LayerNorm 1 (إذا وجد)
            if let (Some(ln_w), Some(ln_b)) = (
                p.ln1_weight.as_ref().and_then(|k| self.get_layer(k, i, &["weight", "w"])),
                p.ln1_bias.as_ref().and_then(|k| self.get_layer(k, i, &["bias", "b"])),
            ) {
                let mean = h.mean_keepdim(1)?;
                let var = h.var_keepdim(1)?;
                h = ((&h - &mean)? / (var + 1e-5)?.sqrt()?)?;
                h = (h * ln_w)?;
                h = (h + ln_b)?;
            }

            // Attention
            if let Some(qkv_key) = &p.attn_qkv {
                if let Some(qkv) = self.get_layer(qkv_key, i, &["weight", "w"]) {
                    let bias = self.get_layer(&qkv_key.replace("weight", "bias"), i, &["bias", "b"]);
                    let mut attn = h.matmul(&qkv.t()?)?;
                    if let Some(b) = bias { attn = attn.broadcast_add(&b)?; }
                    let split = attn.dim(1)? / 3;
                    h = attn.narrow(1, 0, split)?;
                }
            } else if let (Some(q_key), Some(k_key), Some(v_key)) = (
                &p.attn_q, &p.attn_k, &p.attn_v
            ) {
                if let (Some(q), Some(k), Some(v)) = (
                    self.get_layer(q_key, i, &["weight", "w"]),
                    self.get_layer(k_key, i, &["weight", "w"]),
                    self.get_layer(v_key, i, &["weight", "w"]),
                ) {
                    let qo = h.matmul(&q.t()?)?;
                    let ko = h.matmul(&k.t()?)?;
                    let vo = h.matmul(&v.t()?)?;
                    let scores = qo.matmul(&ko.t()?)?;
                    let attn = candle_nn::ops::softmax(&scores, 1)?;
                    h = attn.matmul(&vo)?;
                }
            }

            // Output projection
            if let Some(o_key) = &p.attn_o {
                if let Some(o) = self.get_layer(o_key, i, &["weight", "w"]) {
                    h = h.matmul(&o.t()?)?;
                    if let Some(b) = self.get_layer(&o_key.replace("weight", "bias"), i, &["bias", "b"]) {
                        h = h.broadcast_add(&b)?;
                    }
                }
            }

            // Residual 1
            h = (&residual + &h)?;

            // LayerNorm 2
            let residual2 = h.clone();
            if let (Some(ln2_w), Some(ln2_b)) = (
                p.ln2_weight.as_ref().and_then(|k| self.get_layer(k, i, &["weight", "w"])),
                p.ln2_bias.as_ref().and_then(|k| self.get_layer(k, i, &["bias", "b"])),
            ) {
                let mean = h.mean_keepdim(1)?;
                let var = h.var_keepdim(1)?;
                h = ((&h - &mean)? / (var + 1e-5)?.sqrt()?)?;
                //h = (h * ln2_w)? + ln2_b?;
                h = (h * ln2_w)?;
                h = (h + ln2_b)?;
            }

            // MLP
            if let Some(gate_key) = &p.mlp_gate {
                // Llama-style: Gate + Up + Down
                if let (Some(gate), Some(up), Some(down)) = (
                    self.get_layer(gate_key, i, &["weight", "w"]),
                    p.mlp_up.as_ref().and_then(|k| self.get_layer(k, i, &["weight", "w"])),
                    p.mlp_down.as_ref().and_then(|k| self.get_layer(k, i, &["weight", "w"])),
                ) {
                    let g = candle_nn::ops::silu(&h.matmul(&gate.t()?)?)?;
                    let u = h.matmul(&up.t()?)?;
                    h = (g * u)?.matmul(&down.t()?)?;
                }
            } else if let Some(fc_key) = &p.mlp_fc {
                // GPT-2 style: FC + GELU + Proj
                if let Some(fc) = self.get_layer(fc_key, i, &["weight", "w"]) {
                    let mut mlp = h.matmul(&fc.t()?)?;
                    // GELU تقريبي
                    mlp = mlp.gelu()?;
                    
                    if let Some(proj_key) = &p.mlp_proj {
                        if let Some(proj) = self.get_layer(proj_key, i, &["weight", "w"]) {
                            mlp = mlp.matmul(&proj.t()?)?;
                        }
                    }
                    h = mlp;
                }
            } else if let Some(up_key) = &p.mlp_up {
                // بسيط: Up + ReLU
                if let Some(up) = self.get_layer(up_key, i, &["weight", "w"]) {
                    h = h.matmul(&up.t()?)?;
                    h = h.relu()?;
                    if let Some(down_key) = &p.mlp_down {
                        if let Some(down) = self.get_layer(down_key, i, &["weight", "w"]) {
                            h = h.matmul(&down.t()?)?;
                        }
                    }
                }
            }

            // Residual 2
            h = (&residual2 + &h)?;
        }

        // Final LayerNorm
        if let (Some(ln_w), Some(ln_b)) = (
            p.final_ln_weight.as_ref().and_then(|k| self.weights.get(k.as_str())),
            p.final_ln_bias.as_ref().and_then(|k| self.weights.get(k.as_str())),
        ) {
            let mean = h.mean_keepdim(1)?;
            let var = h.var_keepdim(1)?;
            h = ((&h - &mean)? / (var + 1e-5)?.sqrt()?)?;
            h = (h * ln_w)?;
            h = (h + ln_b)?;
        }

        // LM Head
        let logits = h.matmul(&wte.t()?)?;
        logits.flatten_all()?.to_vec1::<f32>()
    }

    /// البحث عن وزن طبقة مع تعويض {i}
    fn get_layer(&self, key_pattern: &str, layer_idx: usize, suffixes: &[&str]) -> Option<Tensor> {
        let key = key_pattern.replace("{i}", &layer_idx.to_string());
        
        // جرب المفتاح مباشرة
        if let Some(t) = self.weights.get(&key) {
            return Some(t.clone());
        }
        
        // جرب مع لاحقات مختلفة
        for suffix in suffixes {
            let k = if key.ends_with(suffix) { key.clone() } else { format!("{}.{}", key, suffix) };
            if let Some(t) = self.weights.get(&k) {
                return Some(t.clone());
            }
        }
        
        None
    }
}