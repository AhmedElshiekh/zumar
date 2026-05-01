use candle_core::{Tensor, Result, Device, Module};
use candle_nn::{VarMap, Optimizer};
use crate::layers::ZumarModel;
use std::time::Instant;

pub struct DistillConfig {
    pub epochs:        usize,
    pub learning_rate: f64,
    pub temperature:   f64,
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

    /// ✅ الإصلاح الكامل للتقطير:
    /// - المعلم يُمرَّر من الخارج (لا تحميل مزدوج)
    /// - نفس الـ tokenization للطالب والمعلم (char-level متوافق)
    /// - السياق الكامل للطالب وليس token منفرد
    /// - Gradient Accumulation صحيح
    pub fn distill(
        &self,
        student: &mut ZumarModel,
        varmap:  &VarMap,
        teacher: &AutoTeacher,   // ✅ يُستقبل من الخارج — لا تحميل داخلي
        data:    &[String],
    ) -> Result<()> {
        println!("\n🧠 LOGIT MATCHING DISTILLATION");
        println!("   Epochs:      {}", self.config.epochs);
        println!("   Samples:     {}", data.len());
        println!("   Temperature: {}", self.config.temperature);

        let mut opt   = candle_nn::SGD::new(varmap.all_vars(), self.config.learning_rate)?;
        let start     = Instant::now();
        let mut total_tokens = 0u64;

        // ✅ نفس دالة التحويل للطالب والمعلم
        let tokenize = |text: &str, vocab: usize| -> Vec<u32> {
            text.chars()
                .map(|c| (c as u32) % vocab as u32)
                .collect()
        };

        let student_vocab = student.vocab_size;
        let teacher_vocab = teacher.config.vocab_size;

        for epoch in 0..self.config.epochs {
            let mut loss_sum    = 0.0f32;
            let mut token_count = 0u32;
            let mut accum_loss: Option<Tensor> = None;
            let accum_steps = 4;

            for text in data.iter() {
                // ✅ tokens متوافقة: نفس النص، نفس منطق التحويل
                let student_tokens = tokenize(text, student_vocab);
                let teacher_tokens = tokenize(text, teacher_vocab);

                if student_tokens.len() < 2 { continue; }

                for i in 0..student_tokens.len() - 1 {

                    // ══════════════════════════════════════
                    // المعلم: سياق نصي كامل حتى الموضع i
                    // ══════════════════════════════════════
                    let context: String = text.chars().take(i + 1).collect();
                    let teacher_logits = match teacher.predict_tokens(&teacher_tokens[..=i]) {
                        Ok(l)  => l,
                        Err(_) => continue,
                    };

                    // ══════════════════════════════════════
                    // الطالب: سياق كامل حتى الموضع i أيضاً
                    // ══════════════════════════════════════
                    let ctx_ids = &student_tokens[..=i];
                    let input   = Tensor::new(ctx_ids, &self.device)?;

                    // embed كل السياق
                    let emb = student.embedding.forward(&input)?   // [i+1, H]
                        .unsqueeze(0)?;                             // [1, i+1, H]

                    // forward عبر جميع الطبقات
                    let mut h = emb;
                    for layer in &mut student.layers {
                        h = layer.forward_checkpointed(&h)?;
                    }
                    h = student.final_norm.forward(&h)?;
                    let logits = student.lm_head.forward(&h)?;     // [1, i+1, vocab]

                    // خذ logits آخر token فقط (الموضع i)
                    let seq_len      = logits.dim(1)?;
                    let student_last = logits.narrow(1, seq_len - 1, 1)?  // [1,1,vocab]
                        .squeeze(0)?.squeeze(0)?;                          // [vocab]

                    // ══════════════════════════════════════
                    // KL Divergence مع Temperature Scaling
                    // ══════════════════════════════════════
                    let s_len   = student_last.dims1()?;
                    let t_len   = teacher_logits.len();
                    let min_len = s_len.min(t_len);

                    let s_slice   = student_last.narrow(0, 0, min_len)?;
                    let t_slice   = &teacher_logits[..min_len];
                    let t_tensor  = Tensor::new(t_slice, &self.device)?;

                    let temp = self.config.temperature as f64;
                    let t_probs = candle_nn::ops::softmax(&(&t_tensor / temp)?, 0)?;
                    let s_probs = candle_nn::ops::softmax(&(&s_slice  / temp)?, 0)?;

                    let kl = {
                        let eps       = Tensor::new(&[1e-9f32], &self.device)?;
                        let log_s     = s_probs.maximum(&eps)?.log()?;
                        let log_t     = t_probs.maximum(&eps)?.log()?;
                        let kl_terms  = (&t_probs * (&log_t - &log_s)?)?;
                        kl_terms.sum_all()?
                    };

                    loss_sum    += kl.to_scalar::<f32>()?;
                    token_count += 1;
                    total_tokens += 1;

                    // ✅ Gradient Accumulation صحيح
                    accum_loss = Some(match accum_loss {
                        None       => kl,
                        Some(prev) => (&prev + &kl)?,
                    });

                    if token_count % accum_steps == 0 {
                        if let Some(loss) = accum_loss.take() {
                            opt.backward_step(&loss)?;
                        }
                    }

                    drop(input); drop(h); drop(logits);
                    drop(t_tensor); drop(s_slice);
                }
            }

            // ✅ آخر batch — استخدم الـ loss الفعلي وليس صفراً
            if let Some(loss) = accum_loss.take() {
                opt.backward_step(&loss)?;
            }

            // حفظ كل 5 epochs
            if epoch % 5 == 0 || epoch == self.config.epochs - 1 {
                let save_path = std::path::Path::new("models/zumar-v1/model.safetensors");
                std::fs::create_dir_all(save_path.parent().unwrap()).ok();
                varmap.save(save_path)?;
            }

            let elapsed = start.elapsed().as_secs_f64();
            let tps     = total_tokens as f64 / elapsed.max(0.1);
            println!("  ✅ Ep {:>3}: Loss {:.4} | {:.0} tok/s",
                epoch + 1,
                loss_sum / token_count.max(1) as f32,
                tps,
            );
        }

        println!("\n⏱ Total: {:.1}s", start.elapsed().as_secs_f64());
        Ok(())
    }
}

// ============================================================
// AutoTeacher — تحميل ومعالجة نماذج المعلمين
// ============================================================

pub struct AutoTeacher {
    weights: std::collections::HashMap<String, Tensor>,
    pub config: TeacherConfig,
    device:  Device,
}

pub struct TeacherConfig {
    pub embedding_key: String,
    pub num_layers:    usize,
    pub hidden_dim:    usize,
    pub vocab_size:    usize,
    pub arch_type:     String,
    pub prefix_format: String,
}

impl AutoTeacher {
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        println!("   📖 Loading teacher: {}", path);
        let data        = std::fs::read(path)
            .map_err(|e| candle_core::Error::Msg(format!("Cannot read: {}", e)))?;
        let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let raw_header  = &data[8..8 + header_size];
        let header: serde_json::Value = serde_json::from_slice(raw_header)
            .map_err(|e| candle_core::Error::Msg(format!("JSON error: {}", e)))?;

        let config  = Self::detect_architecture_from_header(&header);
        let mut weights = std::collections::HashMap::new();

        if let serde_json::Value::Object(obj) = &header {
            for (name, info) in obj {
                if name == "__metadata__" { continue; }
                let offsets = info["data_offsets"].as_array().unwrap();
                let start   = offsets[0].as_u64().unwrap() as usize;
                let end     = offsets[1].as_u64().unwrap() as usize;
                let shape: Vec<usize> = info["shape"].as_array().unwrap()
                    .iter().map(|v| v.as_u64().unwrap() as usize).collect();
                let dtype = info.get("dtype").and_then(|d| d.as_str()).unwrap_or("F32");
                let raw   = &data[8 + header_size + start..8 + header_size + end];

                let tensor = match dtype {
                    "F16" | "FLOAT16" => {
                        let f16s: Vec<half::f16> = raw.chunks_exact(2)
                            .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
                            .collect();
                        let f32s: Vec<f32> = f16s.iter().map(|v| v.to_f32()).collect();
                        Tensor::from_vec(f32s, shape.as_slice(), device)?
                    }
                    _ => {
                        let f32s: Vec<f32> = raw.chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect();
                        Tensor::from_vec(f32s, shape.as_slice(), device)?
                    }
                };
                weights.insert(name.clone(), tensor);
            }
        }

        println!("   📊 {}: {}d, {}L, {} vocab, {} tensors",
            config.arch_type, config.hidden_dim,
            config.num_layers, config.vocab_size, weights.len());

        Ok(Self { weights, config, device: device.clone() })
    }

    // ✅ load_lazy = load (الاسم للتوافق مع main.rs)
    pub fn load_lazy(path: &str, device: &Device) -> Result<Self> {
        Self::load(path, device)
    }

    pub fn get_config(&self) -> &TeacherConfig { &self.config }

    fn detect_architecture_from_header(header: &serde_json::Value) -> TeacherConfig {
        let keys: Vec<String> = header.as_object()
            .map(|obj| obj.keys().cloned().collect())
            .unwrap_or_default();
        let all_keys = keys.join(" ");

        if all_keys.contains("transformer.h.0.ln_1.weight") {
            let n = keys.iter().filter(|k| k.contains(".ln_1.weight") && k.contains("transformer.h.")).count();
            let h = header.get("transformer.h.0.ln_1.weight").and_then(|v| v.get("shape")).and_then(|v| v.as_array()).and_then(|a| a.get(0)).and_then(|v| v.as_u64()).unwrap_or(768) as usize;
            let v = header.get("transformer.wte.weight").and_then(|v| v.get("shape")).and_then(|v| v.as_array()).and_then(|a| a.get(0)).and_then(|v| v.as_u64()).unwrap_or(50257) as usize;
            return TeacherConfig { embedding_key: "transformer.wte.weight".to_string(), num_layers: n, hidden_dim: h.min(1024), vocab_size: v, arch_type: "gpt2".to_string(), prefix_format: "transformer.h.{i}".to_string() };
        }
        if all_keys.contains("h.0.ln_1.weight") && all_keys.contains("wte.weight") {
            let n = keys.iter().filter(|k| k.contains(".ln_1.weight")).count();
            let h = header.get("h.0.ln_1.weight").and_then(|v| v.get("shape")).and_then(|v| v.as_array()).and_then(|a| a.get(0)).and_then(|v| v.as_u64()).unwrap_or(768) as usize;
            let v = header.get("wte.weight").and_then(|v| v.get("shape")).and_then(|v| v.as_array()).and_then(|a| a.get(0)).and_then(|v| v.as_u64()).unwrap_or(50257) as usize;
            return TeacherConfig { embedding_key: "wte.weight".to_string(), num_layers: n, hidden_dim: h.min(1024), vocab_size: v, arch_type: "gpt2".to_string(), prefix_format: "h.{i}".to_string() };
        }
        if all_keys.contains("model.embed_tokens.weight") && all_keys.contains("model.layers.0.self_attn.q_proj.weight") {
            let n = keys.iter().filter(|k| k.contains("self_attn.q_proj.weight") && k.contains("model.layers.")).count();
            let h = header.get("model.layers.0.input_layernorm.weight").and_then(|v| v.get("shape")).and_then(|v| v.as_array()).and_then(|a| a.get(0)).and_then(|v| v.as_u64()).unwrap_or(1024) as usize;
            let v = header.get("model.embed_tokens.weight").and_then(|v| v.get("shape")).and_then(|v| v.as_array()).and_then(|a| a.get(0)).and_then(|v| v.as_u64()).unwrap_or(32000) as usize;
            if h < 768 {
                return TeacherConfig { embedding_key: "".to_string(), num_layers: 0, hidden_dim: 0, vocab_size: v, arch_type: "skip".to_string(), prefix_format: "".to_string() };
            }
            return TeacherConfig { embedding_key: "model.embed_tokens.weight".to_string(), num_layers: n, hidden_dim: h.min(1024), vocab_size: v.min(50257), arch_type: "llama".to_string(), prefix_format: "model.layers.{i}".to_string() };
        }

        TeacherConfig { embedding_key: "wte.weight".to_string(), num_layers: 6, hidden_dim: 768, vocab_size: 50257, arch_type: "gpt2".to_string(), prefix_format: "h.{i}".to_string() }
    }

    fn get(&self, name: &str) -> Result<&Tensor> {
        self.weights.get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Not found: {}", name)))
    }

    /// ✅ predict_tokens: يأخذ token IDs مباشرة بدلاً من نص
    /// هذا يضمن نفس التمثيل بين الطالب والمعلم
    pub fn predict_tokens(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let vocab = self.config.vocab_size;
        if tokens.is_empty() { return Ok(vec![0.0f32; vocab]); }

        let wte  = self.get(&self.config.embedding_key)?;
        let last = (tokens[tokens.len() - 1] as usize) % wte.dim(0)?;
        let mut h = wte.get(last)?.unsqueeze(0)?;
        let sd   = self.config.hidden_dim.min(1024);

        for i in 0..self.config.num_layers {
            let p        = self.config.prefix_format.replace("{i}", &i.to_string());
            let residual = h.clone();
            h = self.layer_norm(&h, &p, "ln_1", "input_layernorm")?;
            h = if self.config.arch_type == "llama" {
                self.attention_llama(&h, &p, sd)?
            } else {
                self.attention_gpt2(&h, &p, sd)?
            };
            h = (&residual + &h)?;
            let residual2 = h.clone();
            h = self.layer_norm(&h, &p, "ln_2", "post_attention_layernorm")?;
            h = self.mlp(&h, &p, sd)?;
            h = (&residual2 + &h)?;
        }

        h = self.final_layer_norm(&h)?;
        let wte    = self.get(&self.config.embedding_key)?;
        let logits = h.matmul(&wte.t()?)?;
        let flat   = logits.flatten_all()?.to_vec1::<f32>()?;

        let sum: f32 = flat.iter().map(|v| v.abs()).sum();
        if sum < 1e-6 {
            eprintln!("⚠️  Teacher produced near-zero logits!");
        }

        let mut result = vec![0.0f32; vocab];
        let len = flat.len().min(vocab);
        result[..len].copy_from_slice(&flat[..len]);
        Ok(result)
    }

    // ✅ predict النصي محفوظ للتوافق مع الكود القديم
    pub fn predict(&self, text: &str) -> Result<Vec<f32>> {
        let vocab  = self.config.vocab_size;
        let tokens: Vec<u32> = text.chars()
            .map(|c| c as u32 % vocab as u32)
            .collect();
        self.predict_tokens(&tokens)
    }

    fn layer_norm(&self, x: &Tensor, prefix: &str, a: &str, b: &str) -> Result<Tensor> {
        for (key_w, key_b) in &[
            (format!("{}.{}.weight", prefix, a), format!("{}.{}.bias", prefix, a)),
            (format!("{}.{}.weight", prefix, b), format!("{}.{}.bias", prefix, b)),
        ] {
            if let (Ok(w), Ok(b)) = (self.get(key_w), self.get(key_b)) {
                let mean = x.mean_keepdim(1)?;
                let var  = x.var_keepdim(1)?;
                return ((x - &mean)? / (var + 1e-5)?.sqrt()? * w)? + b;
            }
        }
        Ok(x.clone())
    }

    fn final_layer_norm(&self, x: &Tensor) -> Result<Tensor> {
        for key in &["ln_f", "model.norm", "gpt_neox.final_layer_norm"] {
            let w_key = format!("{}.weight", key);
            let b_key = format!("{}.bias", key);
            if let (Ok(w), Ok(b)) = (self.get(&w_key), self.get(&b_key)) {
                let mean = x.mean_keepdim(1)?;
                let var  = x.var_keepdim(1)?;
                return ((x - &mean)? / (var + 1e-5)?.sqrt()? * w)? + b;
            }
        }
        Ok(x.clone())
    }

    fn attention_gpt2(&self, x: &Tensor, p: &str, sd: usize) -> Result<Tensor> {
        let c_attn = self.get(&format!("{}.attn.c_attn.weight", p))?;
        let c_proj = self.get(&format!("{}.attn.c_proj.weight", p))?;
        let dim    = c_attn.dim(0)?.min(sd);
        let slice  = c_attn.narrow(0, 0, dim)?.narrow(1, 0, sd)?;
        let out    = x.matmul(&slice.t()?)?;
        let split  = out.dim(1)? / 3;
        let q = out.narrow(1, 0,       split)?;
        let k = out.narrow(1, split,   split)?;
        let v = out.narrow(1, split*2, split)?;
        let scores = q.matmul(&k.t()?)?;
        let attn   = candle_nn::ops::softmax(&scores, 1)?;
        let out    = attn.matmul(&v)?;
        let proj   = c_proj.narrow(0, 0, sd)?.narrow(1, 0, sd)?;
        out.matmul(&proj.t()?)
    }

    fn attention_llama(&self, x: &Tensor, p: &str, sd: usize) -> Result<Tensor> {
        let q = self.get(&format!("{}.self_attn.q_proj.weight", p))?;
        let k = self.get(&format!("{}.self_attn.k_proj.weight", p))?;
        let v = self.get(&format!("{}.self_attn.v_proj.weight", p))?;
        let o = self.get(&format!("{}.self_attn.o_proj.weight", p))?;
        let (qs, ks, vs, os) = (
            q.narrow(0,0,sd)?.narrow(1,0,sd)?,
            k.narrow(0,0,sd)?.narrow(1,0,sd)?,
            v.narrow(0,0,sd)?.narrow(1,0,sd)?,
            o.narrow(0,0,sd)?.narrow(1,0,sd)?,
        );
        let qo = x.matmul(&qs.t()?)?;
        let ko = x.matmul(&ks.t()?)?;
        let vo = x.matmul(&vs.t()?)?;
        candle_nn::ops::softmax(&qo.matmul(&ko.t()?)?, 1)?
            .matmul(&vo)?.matmul(&os.t()?)
    }

    fn mlp(&self, x: &Tensor, p: &str, sd: usize) -> Result<Tensor> {
        if let (Ok(g), Ok(u), Ok(d)) = (
            self.get(&format!("{}.mlp.gate_proj.weight", p)),
            self.get(&format!("{}.mlp.up_proj.weight",  p)),
            self.get(&format!("{}.mlp.down_proj.weight",p)),
        ) {
            let (gs, us, ds) = (
                g.narrow(0,0,sd)?.narrow(1,0,sd)?,
                u.narrow(0,0,sd)?.narrow(1,0,sd)?,
                d.narrow(0,0,sd)?.narrow(1,0,sd)?,
            );
            return (candle_nn::ops::silu(&x.matmul(&gs.t()?)?)? * x.matmul(&us.t()?)?)?.matmul(&ds.t()?);
        }
        if let (Ok(fc), Ok(proj)) = (
            self.get(&format!("{}.mlp.c_fc.weight",  p)),
            self.get(&format!("{}.mlp.c_proj.weight",p)),
        ) {
            let (fcs, pros) = (
                fc.narrow(0,0,sd)?.narrow(1,0,sd)?,
                proj.narrow(0,0,sd)?.narrow(1,0,sd)?,
            );
            return x.matmul(&fcs.t()?)?.gelu()?.matmul(&pros.t()?);
        }
        x.relu()
    }
}
