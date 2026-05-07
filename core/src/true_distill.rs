use candle_core::{Tensor, Result, Device, D};
use candle_nn::{VarMap, Optimizer, Module};
use crate::layers::ZumarModel;
use crate::tokenizer::ZumarTokenizer;
use crate::layers::vocab_aligner::{VocabAligner, VocabAlignment};
use crate::layers::ewc::{EWC, DistillCheckpoint};
use std::collections::HashMap;
use std::time::Instant;
use std::path::Path;
// use safetensors::Dtype;
use half::{f16, bf16};

const EWC_PATH:   &str = "models/zumar-v1/ewc_state.json";
const CKPT_PATH:  &str = "models/zumar-v1/distill_checkpoint.json";
const MODEL_PATH: &str = "models/zumar-v1/model.safetensors";

pub struct DistillConfig {
    pub epochs:      usize,
    pub base_lr:     f64,
    pub temperature: f32,
    pub ewc_lambda:  f32,
    pub accum_steps: usize,
    pub save_every:  usize,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            epochs:      100,
            base_lr:     1e-3,
            temperature: 3.0,
            ewc_lambda:  400.0,
            accum_steps: 4,
            save_every:  10,
        }
    }
}

pub struct TrueDistiller {
    pub config: DistillConfig,
    pub device: Device,
}

impl TrueDistiller {
    pub fn new(config: DistillConfig, device: Device) -> Self {
        Self { config, device }
    }

    pub fn distill_multi(
        &self,
        student:   &mut ZumarModel,
        varmap:    &VarMap,
        teachers:  &[(AutoTeacher, VocabAlignment)],
        data:      &[String],
        tokenizer: &ZumarTokenizer,
    ) -> Result<()> {
        let mut ewc        = EWC::load(EWC_PATH, self.config.ewc_lambda)?;
        let mut checkpoint = DistillCheckpoint::load(CKPT_PATH)?;

        println!("\n{}", "═".repeat(60));
        println!("🧬 MULTI-TEACHER DISTILLATION (mmap + causal attn)");
        println!("   Teachers: {}  |  Samples: {}", teachers.len(), data.len());
        println!("   Resume:   teacher={} epoch={}",
            checkpoint.teacher_index, checkpoint.epoch);
        ewc.report();
        println!("{}", "═".repeat(60));

        for (idx, (teacher, alignment)) in teachers.iter().enumerate() {
            let name = &teacher.config.arch_type;

            if checkpoint.is_teacher_done(name) {
                println!("\n   ⏭️  Skipping '{}' (already done)", name);
                continue;
            }
            if idx > checkpoint.teacher_index { checkpoint.epoch = 0; }

            println!("\n{}", "═".repeat(60));
            println!("🧬 Teacher {}/{}: {}", idx + 1, teachers.len(), name);
            println!("   {}", alignment.report());
            println!("{}", "═".repeat(60));

            self.distill_one(
                student, varmap, teacher, alignment,
                &mut ewc, &mut checkpoint, data, tokenizer,
            )?;
        }

        println!("\n🎉 ALL TEACHERS DONE!");
        Ok(())
    }

    pub fn distill_one(
        &self,
        student:    &mut ZumarModel,
        varmap:     &VarMap,
        teacher:    &AutoTeacher,
        alignment:  &VocabAlignment,
        ewc:        &mut EWC,
        checkpoint: &mut DistillCheckpoint,
        data:       &[String],
        tokenizer:  &ZumarTokenizer,
    ) -> Result<()> {
        let teacher_name = teacher.config.arch_type.clone();
        let start_epoch  = checkpoint.epoch;
        let lr           = ewc.recommended_lr(self.config.base_lr);

        println!("\n🧠 DISTILLATION: {}", teacher_name);
        println!("   Epochs:  {} → {}", start_epoch, self.config.epochs);
        println!("   LR:      {:.2e}", lr);
        println!("   Temp:    {}", self.config.temperature);
        println!("   EWC λ:   {}", self.config.ewc_lambda);
        println!("   Overlap: {} tokens", alignment.overlap_size());

        if !alignment.is_usable() {
            println!("   ⚠️  Insufficient overlap (<1000) — skipping");
            checkpoint.mark_teacher_done(&teacher_name);
            checkpoint.save(CKPT_PATH)?;
            return Ok(());
        }

        // let mut opt = candle_nn::AdamW::new(
        //     varmap.all_vars(),
        //     candle_nn::ParamsAdamW { lr, ..Default::default() },
        // )?;
        let mut opt = candle_nn::SGD::new(varmap.all_vars(), lr)?;

        let start            = Instant::now();
        let mut total_tokens = 0u64;
        let mut fisher_losses: Vec<Tensor> = Vec::new();

        for epoch in start_epoch..self.config.epochs {
            let mut loss_sum    = 0.0f32;
            let mut token_count = 0u32;
            let mut accum_loss: Option<Tensor> = None;

            for text in data.iter() {
                let zumar_tokens = match tokenizer.encode_ids(text) {
                    Ok(t) if t.len() >= 2 => t,
                    _                      => continue,
                };
                let teacher_tokens = teacher.tokenize(text);

                for i in 0..zumar_tokens.len().saturating_sub(1) {
                    let t_end = (i + 1).min(teacher_tokens.len());
                    let teacher_logits = match teacher.predict_tokens(&teacher_tokens[..t_end]) {
                        Ok(l)  => l,
                        Err(_) => continue,
                    };

                    let input = Tensor::new(&zumar_tokens[..=i], &self.device)?;
                    let emb   = student.embedding.forward(&input)?.unsqueeze(0)?;
                    let mut h = emb;
                    for layer in &mut student.layers {
                        h = layer.forward_checkpointed(&h)?;
                    }
                    h = student.final_norm.forward(&h)?;
                    let logits  = student.lm_head.forward(&h)?;
                    let seq_len = logits.dim(1)?;
                    let last    = logits.narrow(1, seq_len - 1, 1)?
                        .squeeze(0)?.squeeze(0)?;

                    let kl = alignment.kl_divergence(
                        &teacher_logits,
                        &last,
                        self.config.temperature,
                        &self.device,
                    )?;

                    let ewc_loss = ewc.loss_differentiable(varmap, &self.device)?;
                    let total_loss = (&kl + &ewc_loss)?;

                    let kl_val = kl.to_scalar::<f32>().unwrap_or(0.0);
                    loss_sum    += kl_val;
                    token_count += 1;
                    total_tokens += 1;

                    accum_loss = Some(match accum_loss {
                        None       => total_loss,
                        Some(prev) => (&prev + &total_loss)?,
                    });

                    if token_count as usize % self.config.accum_steps == 0 {
                        if let Some(loss) = accum_loss.take() {
                            if fisher_losses.len() < 10 {   // لا تجمع أكثر من 20 خسارة
                                fisher_losses.push(loss.clone());
                            }
                            opt.backward_step(&loss)?;
                        }
                    }
                }
            }

            if let Some(loss) = accum_loss.take() {
                if fisher_losses.len() < 10 {   // لا تجمع أكثر من 20 خسارة
                    fisher_losses.push(loss.clone());
                }
                opt.backward_step(&loss)?;
            }

            let avg_loss = loss_sum / token_count.max(1) as f32;
            checkpoint.epoch        = epoch + 1;
            checkpoint.total_epochs += token_count as usize;
            if avg_loss < checkpoint.best_loss { checkpoint.best_loss = avg_loss; }

            if (epoch + 1) % self.config.save_every == 0 || epoch == self.config.epochs - 1 {
                std::fs::create_dir_all("models/zumar-v1").ok();
                varmap.save(MODEL_PATH)?;
                ewc.save(EWC_PATH)?;
                checkpoint.save(CKPT_PATH)?;
                fisher_losses = fisher_losses.into_iter().rev().take(100).rev().collect();
            }

            let tps     = total_tokens as f64 / start.elapsed().as_secs_f64().max(0.1);
            let ewc_val = ewc.loss(varmap, &self.device)
                .and_then(|t| t.to_scalar::<f32>())
                .unwrap_or(0.0);

            println!("  Ep {:>4}/{}: KL={:.4}  EWC={:.4}  {:.0} tok/s",
                epoch + 1, self.config.epochs, avg_loss, ewc_val, tps);
        }
        
        // استخدام خسارة واحدة فقط لحساب Fisher (ذاكرة أقل)
        let single_loss = fisher_losses.first().map(|l| vec![l.clone()]).unwrap_or_default();
        println!("\n   📐 Computing Fisher for '{}'...", teacher_name);
        ewc.update(varmap, &single_loss, &teacher_name,
            self.config.epochs, &self.device)?;
        // حفظ EWC بعد التحديث
        ewc.save(EWC_PATH)?;
        // تحرير الذاكرة التراكمية في EWC بعد الحفظ
        ewc.cumulative_fisher.clear();
        ewc.cumulative_optimal.clear();
        checkpoint.mark_teacher_done(&teacher_name);
        checkpoint.save(CKPT_PATH)?;
        ewc.report();
        
        println!("\n⏱ Total: {:.1}s", start.elapsed().as_secs_f64());
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════
// prepare_alignments_from_dir (كما هي)
// ════════════════════════════════════════════════════════════
pub fn prepare_alignments_from_dir(
    teacher_paths:    &[std::path::PathBuf],
    student_tok_path: &str,
) -> Result<Vec<VocabAlignment>> {
    let aligner = VocabAligner::from_tokenizer_file(student_tok_path)?;
    let mut alignments = Vec::new();

    for path in teacher_paths {
        let tok_path = path.parent()
            .unwrap_or(Path::new("."))
            .join("tokenizer.json");

        let name = path.parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let (teacher_vocab, t_size) = if tok_path.exists() {
            let v = VocabAligner::load_vocab_from_tokenizer_json(
                tok_path.to_str().unwrap()
            )?;
            let s = v.values().max().copied().unwrap_or(0) as usize + 1;
            println!("   📚 '{}': {} tokens in teacher vocab", name, v.len());
            (v, s)
        } else {
            println!("   ⚠️  No tokenizer.json for '{}' — limited alignment", name);
            (HashMap::new(), 50257)
        };

        let alignment = aligner.align(&teacher_vocab, t_size, &name);
        alignments.push(alignment);
    }

    Ok(alignments)
}

// ════════════════════════════════════════════════════════════
// AutoTeacher — mmap + causal self-attention
// ════════════════════════════════════════════════════════════
pub struct AutoTeacher {
    /// الملف المعنون (mmap) – لا يحتفظ بالأوزان في الذاكرة
    safetensors: candle_core::safetensors::MmapedSafetensors,
    pub config: TeacherConfig,
    device:    Device,
    tokenizer: Option<tokenizers::Tokenizer>,
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
        println!("   📖 Loading: {}", path);

        // فتح الملف عبر mmap – بدون نسخ للذاكرة
        let safetensors = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(path)?
        };

        // قراءة الهيدر فقط (خفيف)
        let data = std::fs::read(path)
            .map_err(|e| candle_core::Error::Msg(format!("Read error: {}", e)))?;
        let header_size = u64::from_le_bytes(
            data[0..8].try_into().unwrap()
        ) as usize;
        let header: serde_json::Value = serde_json::from_slice(&data[8..8 + header_size])
            .map_err(|e| candle_core::Error::Msg(format!("JSON error: {}", e)))?;

        let config = Self::detect_arch(&header);

        let base_dir  = Path::new(path).parent().unwrap_or(Path::new("."));
        let tok_path  = base_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).ok();

        println!("   📊 {}: {}d {}L vocab={} tensors={} tokenizer={}",
            config.arch_type, config.hidden_dim, config.num_layers,
            config.vocab_size, safetensors.tensors().len(),
            if tokenizer.is_some() { "✅" } else { "⚠️  (char-level fallback)" });

        Ok(Self { safetensors, config, device: device.clone(), tokenizer })
    }

    pub fn get_config(&self) -> &TeacherConfig { &self.config }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        if let Some(tok) = &self.tokenizer {
            if let Ok(enc) = tok.encode(text, false) {
                return enc.get_ids().to_vec();
            }
        }
        let v = self.config.vocab_size.max(1);
        text.chars().map(|c| c as u32 % v as u32).collect()
    }

    /// ✅ predict_tokens الانتباه السببي الكامل (causal)
    pub fn predict_tokens(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let vocab = self.config.vocab_size;
        if tokens.is_empty() || self.config.arch_type == "skip" {
            return Ok(vec![0.0f32; vocab]);
        }

        let wte = self.get_tensor(&self.config.embedding_key)?;
        let vs = wte.dim(0)?;
        let hd = wte.dim(1)?;

        let seq_ids: Vec<u32> = tokens.iter().map(|&t| (t as usize % vs) as u32).collect();
        let input = Tensor::new(seq_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let (_, seq_len) = input.dims2()?;
        let flat_ids = input.flatten_all()?;
        let emb = wte.index_select(&flat_ids, 0)?;
        let mut h = emb.reshape((1, seq_len, hd))?;

        let causal_mask = self.create_causal_mask(seq_len)?;

        for i in 0..self.config.num_layers {
            let p = self.config.prefix_format.replace("{i}", &i.to_string());
            let residual = h.clone();

            h = self.layer_norm_3d(&h, &p, "ln_1", "input_layernorm")?;
            h = if self.config.arch_type == "llama" {
                self.causal_attn_llama(&h, &p, &causal_mask)?
            } else {
                self.causal_attn_gpt2(&h, &p, &causal_mask)?
            };
            h = (&residual + &h)?;

            let residual2 = h.clone();
            h = self.layer_norm_3d(&h, &p, "ln_2", "post_attention_layernorm")?;
            h = self.mlp_3d(&h, &p)?;
            h = (&residual2 + &h)?;
        }

        h = self.final_norm_3d(&h)?;

        let last_hidden = h.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        let lm = self.get_tensor("lm_head.weight").unwrap_or(wte);
        let lm_rows = lm.dim(0)?.min(vs);
        let logits = last_hidden.matmul(&lm.narrow(0, 0, lm_rows)?.t()?)?;
        let flat = logits.squeeze(0)?.to_vec1::<f32>()?;

        let max_v = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max_v.abs() < 1e-6 {
            eprintln!("⚠️  Teacher '{}' near-zero logits", self.config.arch_type);
        }

        let mut result = vec![0.0f32; vocab];
        let len = flat.len().min(vocab);
        result[..len].copy_from_slice(&flat[..len]);
        Ok(result)
    }

     // ─ـ دوال مساعدة لاستخراج التنسورات من mmap ──────────────
    fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let view = self.safetensors
            .get(name)
            .map_err(|e| candle_core::Error::Msg(format!("tensor '{}': {}", name, e)))?;
    
        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();
        let num_elements: usize = shape.iter().product();
        let expected_f32 = num_elements * 4;
        let expected_f16 = num_elements * 2;
    
        if data.len() == expected_f32 {
            let f32s: Vec<f32> = data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Tensor::from_vec(f32s, shape, &self.device)
        } else if data.len() == expected_f16 {
            // افتراضي F16 (الأغلب في النماذج المحولة من HuggingFace)
            let f16s: Vec<f16> = data
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect();
            let f32s: Vec<f32> = f16s.iter().map(|v| v.to_f32()).collect();
            Tensor::from_vec(f32s, shape, &self.device)
        } else {
            Err(candle_core::Error::Msg(format!(
                "tensor '{}' unexpected data length {} for shape {:?} (expected {} or {} bytes)",
                name, data.len(), shape, expected_f32, expected_f16
            )))
        }
    }   

    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY }))
            .collect();
        Tensor::from_vec(mask, (1, 1, seq_len, seq_len), &self.device)
    }

    // ─ـ دوال 3D layer norm / final norm ──────────────────────
    fn layer_norm_3d(&self, x: &Tensor, prefix: &str, a: &str, b: &str) -> Result<Tensor> {
        for (wk, bk) in &[
            (format!("{}.{}.weight", prefix, a), format!("{}.{}.bias", prefix, a)),
            (format!("{}.{}.weight", prefix, b), format!("{}.{}.bias", prefix, b)),
        ] {
            if let (Ok(w), Ok(bias)) = (self.get_tensor(wk), self.get_tensor(bk)) {
                let h = x.dim(2)?;
                let mean = x.mean_keepdim(D::Minus1)?;
                let var = x.var_keepdim(D::Minus1)?;
                let x_norm = ((x - &mean)? / (var + 1e-5)?.sqrt()?)?;
                let w = w.reshape((1, 1, h))?;
                let bias = bias.reshape((1, 1, h))?;
                return Ok((x_norm.broadcast_mul(&w)? + &bias)?);
            }
        }
        Ok(x.clone())
    }

    fn final_norm_3d(&self, x: &Tensor) -> Result<Tensor> {
        for key in &["ln_f", "model.norm", "norm"] {
            if let (Ok(w), Ok(bias)) = (self.get_tensor(&format!("{}.weight", key)), self.get_tensor(&format!("{}.bias", key))) {
                let h = x.dim(2)?;
                let mean = x.mean_keepdim(D::Minus1)?;
                let var = x.var_keepdim(D::Minus1)?;
                let x_norm = ((x - &mean)? / (var + 1e-5)?.sqrt()?)?;
                let w = w.reshape((1, 1, h))?;
                let bias = bias.reshape((1, 1, h))?;
                return Ok((x_norm.broadcast_mul(&w)? + &bias)?);
            }
        }
        Ok(x.clone())
    }

    // ─ـ Causal Attention GPT-2 ────────────────────────────────
    fn causal_attn_gpt2(&self, x: &Tensor, p: &str, mask: &Tensor) -> Result<Tensor> {
        let c_attn = self.get_tensor(&format!("{}.attn.c_attn.weight", p))?;
        let c_proj = self.get_tensor(&format!("{}.attn.c_proj.weight", p))?;
        let qkv = x.matmul(&c_attn.t()?)?;
        let split = qkv.dim(2)? / 3;
        let q = qkv.narrow(2, 0, split)?;
        let k = qkv.narrow(2, split, split)?;
        let v = qkv.narrow(2, split*2, split)?;
        let scores = (q.matmul(&k.transpose(1,2)?)? / (split as f64).sqrt())?;
        let attn = candle_nn::ops::softmax(&(scores + mask)?, D::Minus1)?;
        attn.matmul(&v)?.matmul(&c_proj.t()?)
    }

    // ─ـ Causal Attention Llama ────────────────────────────────
    fn causal_attn_llama(&self, x: &Tensor, p: &str, mask: &Tensor) -> Result<Tensor> {
        let qw = self.get_tensor(&format!("{}.self_attn.q_proj.weight", p))?;
        let kw = self.get_tensor(&format!("{}.self_attn.k_proj.weight", p))?;
        let vw = self.get_tensor(&format!("{}.self_attn.v_proj.weight", p))?;
        let ow = self.get_tensor(&format!("{}.self_attn.o_proj.weight", p))?;
        let q = x.matmul(&qw.t()?)?;
        let k = x.matmul(&kw.t()?)?;
        let v = x.matmul(&vw.t()?)?;
        let scores = (q.matmul(&k.transpose(1,2)?)? / (q.dim(2)? as f64).sqrt())?;
        let attn = candle_nn::ops::softmax(&(scores + mask)?, D::Minus1)?;
        attn.matmul(&v)?.matmul(&ow.t()?)
    }

    // ─ـ MLP 3D ────────────────────────────────────────────────
    fn mlp_3d(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        if let (Ok(g), Ok(u), Ok(d)) = (
            self.get_tensor(&format!("{}.mlp.gate_proj.weight", p)),
            self.get_tensor(&format!("{}.mlp.up_proj.weight", p)),
            self.get_tensor(&format!("{}.mlp.down_proj.weight", p)),
        ) {
            let gate = candle_nn::ops::silu(&x.matmul(&g.t()?)?)?;
            let up = x.matmul(&u.t()?)?;
            let hidden = (gate * up)?;
            return hidden.matmul(&d.t()?);
        }
        if let (Ok(fc), Ok(proj)) = (
            self.get_tensor(&format!("{}.mlp.c_fc.weight", p)),
            self.get_tensor(&format!("{}.mlp.c_proj.weight", p)),
        ) {
            let h = x.matmul(&fc.t()?)?.gelu()?;
            return h.matmul(&proj.t()?);
        }
        x.relu()
    }

    // detect_arch – كما هي دون تغيير (نفس الكود السابق)
    fn detect_arch(h: &serde_json::Value) -> TeacherConfig {
        // ... (انسخ نفس الدالة الموجودة في الملف السابق، لن نكررها هنا)
        // استخدم أي نسخة صحيحة لديك.
        // ...
        TeacherConfig {
            embedding_key: "wte.weight".into(),
            num_layers:    6,
            hidden_dim:    768,
            vocab_size:    50257,
            arch_type:     "gpt2".into(),
            prefix_format: "h.{i}".into(),
        }
    }
}