use candle_core::{Tensor, Result, Device, D};
use candle_nn::{VarMap, Optimizer, Module};
use crate::layers::ZumarModel;
use crate::tokenizer::ZumarTokenizer;
use crate::layers::vocab_aligner::{VocabAligner, VocabAlignment};
use crate::layers::ewc::{EWC, DistillCheckpoint};
use std::collections::HashMap;
use std::time::Instant;
use std::path::Path;
use std::cell::RefCell;
use half::f16;

use crate::layers::config::ModelConfig;

// const EWC_PATH:   &str = "models/zumar-v1/ewc_state.json";
// const CKPT_PATH:  &str = "models/zumar-v1/distill_checkpoint.json";
// const MODEL_PATH: &str = "models/zumar-v1/model.safetensors";
// إزالة الثوابت الثابتة واستبدالها بدوال

pub fn get_ewc_path(output_dir: &str) -> String {
    format!("{}/ewc_state.json", output_dir)
}

pub fn get_ckpt_path(output_dir: &str) -> String {
    format!("{}/distill_checkpoint.json", output_dir)
}

pub fn get_model_path(output_dir: &str) -> String {
    format!("{}/model.safetensors", output_dir)
}

// ──────────────────────────────────────────────────────────────
// TeacherLogitsCache (كما هي)
// ──────────────────────────────────────────────────────────────
thread_local! {
    static LOGITS_CACHE: RefCell<TeacherLogitsCache> = RefCell::new(TeacherLogitsCache::new());
}

pub struct TeacherLogitsCache {
    cache: HashMap<u64, Vec<f32>>,
    hits: usize,
    misses: usize,
    causal_successes: usize,
    causal_failures: usize,
}

impl TeacherLogitsCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            hits: 0,
            misses: 0,
            causal_successes: 0,
            causal_failures: 0,
        }
    }
    
    fn hash_tokens(tokens: &[u32]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &t in tokens {
            hash ^= t as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
    
    pub fn get_or_compute(
        &mut self,
        tokens: &[u32],
        teacher: &AutoTeacher,
    ) -> Result<Vec<f32>> {
        let key = Self::hash_tokens(tokens);
        if let Some(cached) = self.cache.get(&key) {
            self.hits += 1;
            return Ok(cached.clone());
        }
        self.misses += 1;
        let logits = teacher.predict_with_embeddings(tokens)?; // تجنب causal attention
        self.cache.insert(key, logits.clone());
        Ok(logits)
    }
    
    pub fn stats(&self) -> (usize, usize, usize, usize) {
        (self.hits, self.misses, 0, 0)
    }
}

// ──────────────────────────────────────────────────────────────
// DistillConfig
// ──────────────────────────────────────────────────────────────
pub struct DistillConfig {
    pub epochs:      usize,
    pub base_lr:     f64,
    pub temperature: f32,
    pub ewc_lambda:  f32,
    pub accum_steps: usize,
    pub save_every:  usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
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
            lora_rank: 8,
            lora_alpha: 16.0,
        }
    }
}

// ──────────────────────────────────────────────────────────────
// TrueDistiller
// ──────────────────────────────────────────────────────────────
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
        student: &mut ZumarModel,
        varmap: &VarMap,
        teachers: &[(String, AutoTeacher, VocabAlignment)],
        data: &[String],
        tokenizer: &ZumarTokenizer,
        output_dir: &str,  // معامل جديد
    ) -> Result<()> {
        let ewc_path = get_ewc_path(output_dir);
        let ckpt_path = get_ckpt_path(output_dir);
        let model_path = get_model_path(output_dir);
        
        let mut ewc = EWC::load(&ewc_path, self.config.ewc_lambda)?;
        let mut checkpoint = DistillCheckpoint::load(&ckpt_path)?;

        println!("\n{}", "═".repeat(60));
        println!("🧬 MULTI-TEACHER DISTILLATION (Optimized)");
        println!("   Teachers: {}  |  Samples: {}", teachers.len(), data.len());
        println!("   Resume:   teacher={} epoch={}", checkpoint.teacher_index, checkpoint.epoch);
        ewc.report();
        println!("{}", "═".repeat(60));
    
        for (idx, (name, teacher, alignment)) in teachers.iter().enumerate() {
            if checkpoint.is_teacher_done(name) && checkpoint.epoch >= self.config.epochs {
                println!("\n   ⏭️  Skipping '{}'", name);
                continue;
            }
            if idx > checkpoint.teacher_index { checkpoint.epoch = 0; }
    
            println!("\n🧬 Teacher {}/{}: {}", idx + 1, teachers.len(), name);
            println!("   {}", alignment.report());
    
            // self.distill_one(student, varmap, name, teacher, alignment, &mut ewc, &mut checkpoint, data, tokenizer)?;

            self.distill_one(
                student, varmap, name, teacher, alignment,
                &mut ewc, &mut checkpoint, data, tokenizer,
                output_dir, &ewc_path, &ckpt_path, &model_path,
            )?;
        }
        Ok(())
    }

    pub fn distill_one(
        &self,
        student: &mut ZumarModel,
        varmap: &VarMap,
        teacher_name: &str,
        teacher: &AutoTeacher,
        alignment: &VocabAlignment,
        ewc: &mut EWC,
        checkpoint: &mut DistillCheckpoint,
        data: &[String],
        tokenizer: &ZumarTokenizer,
        output_dir: &str,
        ewc_path: &str,
        ckpt_path: &str,
        model_path: &str,
    ) -> Result<()> {
        let start_epoch = checkpoint.epoch;
        let lr = ewc.recommended_lr(self.config.base_lr);
    
        println!("   Epochs: {}/{} | LR: {:.2e} | Temp: {}", start_epoch, self.config.epochs, lr, self.config.temperature);
    
        if !alignment.is_usable() {
            println!("   ⚠️  Skipping (overlap <1000)");
            checkpoint.mark_teacher_done(teacher_name);
            checkpoint.save(&get_ckpt_path(output_dir))?;
            return Ok(());
        }
    
        let mut opt = candle_nn::SGD::new(varmap.all_vars(), lr)?;
        let start = Instant::now();
        let mut total_tokens = 0u64;
        let mut fisher_losses: Vec<Tensor> = Vec::new();
    
        // تحميل offline cache
        // let zlog_path = format!("models/zlog/{}.zlog", teacher_name);
        // let offline_cache = if Path::new(&zlog_path).exists() {
        //     println!("   📂 Offline mode using {}", zlog_path);
        //     match load_zlog(&zlog_path, teacher.config.vocab_size) {
        //         Ok(c) => {
        //             println!("   ✅ Loaded {} entries (vocab={})", c.len(), teacher.config.vocab_size);
        //             Some(c)
        //         }
        //         Err(e) => {
        //             println!("   ⚠️  Zlog error: {} -> online", e);
        //             None
        //         }
        //     }
        // } else { None };
        // ── تحميل Zlog offline ──
        let zlog_path = format!("models/zlog/{}.zlog", teacher_name);
        let offline_cache: Option<Vec<Vec<f32>>> = if Path::new(&zlog_path).exists() {
            println!("   📂 Offline mode using {}", zlog_path);
            match load_zlog(&zlog_path, teacher.config.vocab_size) {
                Ok(c) => {
                    println!("   ✅ Loaded {} entries", c.len());
                    Some(c)
                }
                Err(e) => {
                    println!("   ⚠️  Zlog error: {} -> online", e);
                    None
                }
            }
        } else { None };
    
        for epoch in start_epoch..self.config.epochs {
            let mut loss_sum = 0.0f32;
            let mut step_count = 0usize;
            let mut accum_kl: Option<Tensor> = None;

            for (idx, text) in data.iter().enumerate() {
                // 1. ترميز الطالب
                let zumar_tokens = match tokenizer.encode_ids(text) {
                    Ok(t) if t.len() >= 2 => t,
                    _ => continue,
                };
                let seq_len = zumar_tokens.len();
            
                // 2. الحصول على logits المعلم بالترتيب
                let teacher_logits = if let Some(ref cache) = offline_cache {
                    if idx >= cache.len() { continue; }
                    cache[idx].clone()
                // } else {
                // let teacher_logits = if let Some(ref logits_vec) = offline_logits {
                //     if idx < logits_vec.len() && !logits_vec[idx].is_empty() {
                //         logits_vec[idx].clone()
                //     } else {
                //         continue
                //     }
                } else {
                    let teacher_tokens = teacher.tokenize(text);
                    LOGITS_CACHE.with(|c| c.borrow_mut().get_or_compute(&teacher_tokens, teacher))?
                };
            
                // ... باقي الكود كما هو ...
            // }
            // for text in data.iter() {
            //     // 1. ترميز الطالب
            //     let zumar_tokens = match tokenizer.encode_ids(text) {
            //         Ok(t) if t.len() >= 2 => t,
            //         _ => continue,
            //     };
            //     let seq_len = zumar_tokens.len();
    
            //     // 2. الحصول على logits المعلم (مرة واحدة لكل نص)
            //     let teacher_logits = if let Some(ref cache) = offline_cache {
            //         let clean_text = text.trim();
            //         let hash = {
            //             let mut h = 0xcbf29ce484222325u64;
            //             for &b in clean_text.as_bytes() {
            //                 h ^= b as u64;
            //                 h = h.wrapping_mul(0x100000001b3);
            //             }
            //             h
            //         };
            //         match cache.get(&hash) {
            //             Some(l) => l.clone(),
            //             None => continue,
            //         }
            //     } else {
            //         let teacher_tokens = teacher.tokenize(text);
            //         LOGITS_CACHE.with(|c| c.borrow_mut().get_or_compute(&teacher_tokens, teacher))?
            //     };
    
                // 3. forward الطالب على التسلسل كاملاً (مرة واحدة)
                let input_ids = Tensor::new(zumar_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                let student_logits_all = student.forward_sequence(&input_ids)?; // (1, seq_len, vocab)
    
                // 4. استخدام آخر رمز فقط لحساب KL (تسريع هائل)
                let last_student_logit = student_logits_all
                    .narrow(1, seq_len - 1, 1)?
                    .squeeze(0)?
                    .squeeze(0)?;
    
                let kl = alignment.kl_divergence(
                    &teacher_logits,
                    &last_student_logit,
                    self.config.temperature,
                    &self.device,
                )?;
                // let kl = alignment.cross_entropy_loss(
                //     &teacher_logits,
                //     &last_student_logit,
                //     self.config.temperature,
                //     &self.device,
                // )?;
                // let kl = alignment.mse_loss(
                //     &teacher_logits,
                //     &last_student_logit,
                //     self.config.temperature,
                //     &self.device,
                // )?;
                
                let kl_val = kl.to_scalar::<f32>()?;
                loss_sum += kl_val;
                total_tokens += (seq_len - 1) as u64; // تقديري
                step_count += 1;
    
                accum_kl = Some(match accum_kl {
                    None => kl,
                    Some(prev) => (&prev + &kl)?,
                });
    
                // كل accum_steps نقوم بـ backward (مع حساب ewc_loss مرة واحدة فقط)
                if step_count % self.config.accum_steps == 0 {
                    if let Some(kl_loss) = accum_kl.take() {
                        let ewc_loss = ewc.loss_differentiable(varmap, &self.device)?.reshape(&[])?;
                        let total_loss = (&kl_loss + &ewc_loss)?;
                        opt.backward_step(&total_loss)?;
                        if fisher_losses.len() < 10 {
                            fisher_losses.push(total_loss);
                        }
                    }
                }
            }
    
            // باقي الخسارة بعد الحلقة
            if let Some(kl_loss) = accum_kl.take() {
                let ewc_loss = ewc.loss_differentiable(varmap, &self.device)?.reshape(&[])?;
                let total_loss = (&kl_loss + &ewc_loss)?;
                opt.backward_step(&total_loss)?;
                if fisher_losses.len() < 10 {
                    fisher_losses.push(total_loss);
                }
            }
    
            let avg_loss = loss_sum / step_count.max(1) as f32;
            checkpoint.epoch = epoch + 1;
            checkpoint.total_epochs += step_count;
            if avg_loss < checkpoint.best_loss {
                checkpoint.best_loss = avg_loss;
            }
    
            if (epoch + 1) % self.config.save_every == 0 || epoch == self.config.epochs - 1 {
                std::fs::create_dir_all("models/zumar-v1").ok();
                varmap.save(&get_model_path(output_dir))?;
                ewc.save(&get_ewc_path(output_dir))?;
                checkpoint.save(&get_ckpt_path(output_dir))?;
                fisher_losses.truncate(10);
            }
    
            let elapsed = start.elapsed().as_secs_f64();
            let tps = if elapsed > 0.0 {
                total_tokens as f64 / elapsed
            } else {
                0.0
            };
            println!(
                "  Ep {:>4}/{}: KL={:.8}  {:.1} tok/s",
                epoch + 1,
                self.config.epochs,
                avg_loss,
                tps
            );
        }
    
        // تحديث Fisher بعد الانتهاء
        println!("\n   📐 Computing Fisher for '{}'...", teacher_name);
        if let Some(loss) = fisher_losses.first() {
            ewc.update(varmap, &[loss.clone()], teacher_name, self.config.epochs, &self.device)?;
        }
        ewc.save(&get_ewc_path(output_dir))?;
        checkpoint.mark_teacher_done(teacher_name);
        checkpoint.save(&get_ckpt_path(output_dir))?;
    
        println!("   ⏱ Total: {:.1}s", start.elapsed().as_secs_f64());
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════
// باقي الدوال (prepare_alignments_from_dir, AutoTeacher, load_zlog, إلخ)
// يجب أن تبقى كما هي، لكن أضف forward_sequence في ZumarModel (خارج هذا الملف)
// ════════════════════════════════════════════════════════════

pub fn prepare_alignments_from_dir(
    teacher_paths:    &[std::path::PathBuf],
    student_tok_path: &str,
) -> Result<Vec<VocabAlignment>> {
    let aligner = VocabAligner::from_tokenizer_file(student_tok_path)?;
    let mut alignments = Vec::new();
    for path in teacher_paths {
        let tok_path = path.parent().unwrap_or(Path::new(".")).join("tokenizer.json");
        let name = path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
        let (teacher_vocab, t_size) = if tok_path.exists() {
            let v = VocabAligner::load_vocab_from_tokenizer_json(tok_path.to_str().unwrap())?;
            let s = v.values().max().copied().unwrap_or(0) as usize + 1;
            println!("   📚 '{}': {} tokens in teacher vocab", name, v.len());
            (v, s)
        } else {
            println!("   ⚠️  No tokenizer.json for '{}'", name);
            (HashMap::new(), 50257)
        };
        // alignments.push(aligner.align(&teacher_vocab, t_size, &name));
        let mut alignment = aligner.align(&teacher_vocab, t_size, &name);
        alignment.truncate(1000);   // أضف هذا السطر
        alignments.push(alignment);
    }
    Ok(alignments)
}


// ════════════════════════════════════════════════════════════
// AutoTeacher — mmap + Hybrid Predict
// ════════════════════════════════════════════════════════════
pub struct AutoTeacher {
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
        let safetensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
        let data = std::fs::read(path).map_err(|e| candle_core::Error::Msg(format!("Read error: {}", e)))?;
        let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&data[8..8 + header_size])
            .map_err(|e| candle_core::Error::Msg(format!("JSON error: {}", e)))?;
        let config = Self::detect_arch(&header);
        let base_dir = Path::new(path).parent().unwrap_or(Path::new("."));
        let tok_path = base_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).ok();
        println!("   📊 {}: {}d {}L vocab={} tensors={} tokenizer={}",
            config.arch_type, config.hidden_dim, config.num_layers,
            config.vocab_size, safetensors.tensors().len(),
            if tokenizer.is_some() { "✅" } else { "⚠️" });
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

    fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let view = self.safetensors.get(name)
            .map_err(|e| candle_core::Error::Msg(format!("tensor '{}': {}", name, e)))?;
        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();
        let num_elements: usize = shape.iter().product();
        let expected_f32 = num_elements * 4;
        let expected_f16 = num_elements * 2;
        if data.len() == expected_f32 {
            let f32s: Vec<f32> = data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            Tensor::from_vec(f32s, shape, &self.device)
        } else if data.len() == expected_f16 {
            let f16s: Vec<f16> = data.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]])).collect();
            let f32s: Vec<f32> = f16s.iter().map(|v| v.to_f32()).collect();
            Tensor::from_vec(f32s, shape, &self.device)
        } else {
            Err(candle_core::Error::Msg(format!("tensor '{}' unexpected size", name)))
        }
    }

    // ════════════════════════════════════════════════════════
    // طريقة 1: Embedding-only (تعمل دائماً)
    // ════════════════════════════════════════════════════════
    pub fn predict_with_embeddings(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let vocab = self.config.vocab_size;
        let wte = self.get_tensor(&self.config.embedding_key)?;
        let vs = wte.dim(0)?;
        let hd = wte.dim(1)?;
        let wte_data = wte.to_vec2::<f32>()?;
        
        let mut sum = vec![0.0f32; hd];
        let mut total_weight = 0.0f32;
        for (pos, &t) in tokens.iter().enumerate() {
            let idx = (t as usize) % vs;
            let weight = 1.0 / (1.0 + (tokens.len() - 1 - pos) as f32 * 0.1);
            for (s, &e) in sum.iter_mut().zip(wte_data[idx].iter()) { *s += e * weight; }
            total_weight += weight;
        }
        for s in &mut sum { *s /= total_weight; }
        
        let mut logits = vec![0.0f32; vocab.min(vs)];
        for (i, l) in logits.iter_mut().enumerate() {
            *l = sum.iter().zip(wte_data[i].iter()).map(|(a, b)| a * b).sum();
        }
        Ok(logits)
    }

    // ════════════════════════════════════════════════════════
    // طريقة 2: Causal Attention (للمعلمين المتوافقين)
    // ════════════════════════════════════════════════════════
    pub fn predict_with_causal_attention(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let vocab = self.config.vocab_size;
        if tokens.is_empty() || self.config.arch_type == "skip" {
            return Err(candle_core::Error::Msg("skip".into()));
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
            h = self.causal_attention_safe(&h, &p, &causal_mask)?;
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
        Ok(flat[..vocab.min(vs)].to_vec())
    }

    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY }))
            .collect();
        Tensor::from_vec(mask, (1, seq_len, seq_len), &self.device)
    }

    // ════════════════════════════════════════════════════════
     // دوال مساعدة للـ causal attention
    // ════════════════════════════════════════════════════════
    fn layer_norm_3d(&self, x: &Tensor, prefix: &str, a: &str, b: &str) -> Result<Tensor> {
        for (wk, bk) in &[
            (format!("{}.{}.weight", prefix, a), format!("{}.{}.bias", prefix, a)),
            (format!("{}.{}.weight", prefix, b), format!("{}.{}.bias", prefix, b)),
        ] {
            if let (Ok(w), Ok(bias)) = (self.get_tensor(wk), self.get_tensor(bk)) {
                let h = x.dim(2)?;
                let seq = x.dim(1)?;
                let mean = x.mean_keepdim(D::Minus1)?;
                let var = x.var_keepdim(D::Minus1)?;
                let mean_exp = mean.broadcast_as((1, seq, h))?;
                let var_exp = var.broadcast_as((1, seq, h))?;
                let x_norm = ((x - &mean_exp)? / (var_exp + 1e-5)?.sqrt()?)?;
                let w = w.reshape((1, 1, h))?;
                let bias = bias.reshape((1, 1, h))?;
                return Ok((x_norm.broadcast_mul(&w)? + &bias)?);
            }
        }
        Ok(x.clone())
    }

    fn final_norm_3d(&self, x: &Tensor) -> Result<Tensor> {
        for key in &["ln_f", "model.norm", "norm"] {
            let wk = format!("{}.weight", key);
            let bk = format!("{}.bias", key);
            if let (Ok(w), Ok(bias)) = (self.get_tensor(&wk), self.get_tensor(&bk)) {
                let h = x.dim(2)?;
                let seq = x.dim(1)?;
                let mean = x.mean_keepdim(D::Minus1)?;
                let var = x.var_keepdim(D::Minus1)?;
                let mean_exp = mean.broadcast_as((1, seq, h))?;
                let var_exp = var.broadcast_as((1, seq, h))?;
                let x_norm = ((x - &mean_exp)? / (var_exp + 1e-5)?.sqrt()?)?;
                let w = w.reshape((1, 1, h))?;
                let bias = bias.reshape((1, 1, h))?;
                return Ok((x_norm.broadcast_mul(&w)? + &bias)?);
            }
        }
        Ok(x.clone())
    }
    
    fn causal_attention_safe(&self, x: &Tensor, p: &str, mask: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = x.dims3()?;
        
        // محاولة نمط GPT-2: c_attn + c_proj
        if let (Ok(c_attn), Ok(c_proj)) = (
            self.get_tensor(&format!("{}.attn.c_attn.weight", p)),
            self.get_tensor(&format!("{}.attn.c_proj.weight", p)),
        ) {
            let attn_hidden = c_attn.dim(0)?;
            let qkv = x.reshape((batch * seq_len, hidden_dim))?
                        .matmul(&c_attn.t()?)?
                        .reshape((batch, seq_len, attn_hidden))?;
            
            let head_dim = attn_hidden / 3;
            let q = qkv.narrow(2, 0, head_dim)?;
            let k = qkv.narrow(2, head_dim, head_dim)?;
            let v = qkv.narrow(2, head_dim * 2, head_dim)?;
            
            let scale = (head_dim as f64).sqrt();
            let scores = (q.matmul(&k.transpose(1, 2)?)? / scale)?;
            let scores = (scores + mask)?;
            let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
            
            let attn_output = attn_weights.matmul(&v)?;
            let attn_2d = attn_output.reshape((batch * seq_len, head_dim))?;
            return attn_2d.matmul(&c_proj.t()?)?.reshape((batch, seq_len, c_proj.dim(0)?));
        }
        
        // محاولة نمط LLaMA: q_proj, k_proj, v_proj, o_proj
        if let (Ok(qw), Ok(kw), Ok(vw), Ok(ow)) = (
            self.get_tensor(&format!("{}.self_attn.q_proj.weight", p)),
            self.get_tensor(&format!("{}.self_attn.k_proj.weight", p)),
            self.get_tensor(&format!("{}.self_attn.v_proj.weight", p)),
            self.get_tensor(&format!("{}.self_attn.o_proj.weight", p)),
        ) {
            let x_2d = x.reshape((batch * seq_len, hidden_dim))?;
            
            let q = x_2d.matmul(&qw.t()?)?.reshape((batch, seq_len, qw.dim(0)?))?;
            let k = x_2d.matmul(&kw.t()?)?.reshape((batch, seq_len, kw.dim(0)?))?;
            let v = x_2d.matmul(&vw.t()?)?.reshape((batch, seq_len, vw.dim(0)?))?;
            
            let scale = (q.dim(2)? as f64).sqrt();
            let scores = (q.matmul(&k.transpose(1, 2)?)? / scale)?;
            let scores = (scores + mask)?;
            let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
            
            let attn_output = attn_weights.matmul(&v)?;
            let attn_2d = attn_output.reshape((batch * seq_len, attn_output.dim(2)?))?;
            return attn_2d.matmul(&ow.t()?)?.reshape((batch, seq_len, ow.dim(0)?));
        }
        
        Err(candle_core::Error::Msg(format!("No attention weights found for {}", p)))
    }
        
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

    fn detect_arch(h: &serde_json::Value) -> TeacherConfig {
        let keys: Vec<String> = h.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
        let all = keys.join(" ");
        let get_dim = |k: &str| -> usize {
            h.get(k).and_then(|v| v.get("shape")).and_then(|s| s.as_array()).and_then(|a| a.get(0)).and_then(|v| v.as_u64()).unwrap_or(768) as usize
        };
        let get_vocab = |k: &str| -> usize {
            h.get(k).and_then(|v| v.get("shape")).and_then(|s| s.as_array()).and_then(|a| a.get(0)).and_then(|v| v.as_u64()).unwrap_or(50257) as usize
        };
        let count = |pat: &str| keys.iter().filter(|k| k.contains(pat)).count();

        if all.contains("transformer.h.0") {
            return TeacherConfig { embedding_key: "transformer.wte.weight".into(), num_layers: count(".ln_1.weight"), hidden_dim: get_dim("transformer.h.0.ln_1.weight").min(1024), vocab_size: get_vocab("transformer.wte.weight"), arch_type: "gpt2".into(), prefix_format: "transformer.h.{i}".into() };
        }
        if all.contains("h.0.ln_1") && all.contains("wte.weight") {
            return TeacherConfig { embedding_key: "wte.weight".into(), num_layers: count(".ln_1.weight"), hidden_dim: get_dim("h.0.ln_1.weight").min(1024), vocab_size: get_vocab("wte.weight"), arch_type: "distilgpt2".into(), prefix_format: "h.{i}".into() };
        }
        if all.contains("model.embed_tokens") && all.contains("model.layers.0") {
            let hd = get_dim("model.layers.0.input_layernorm.weight").min(1024);
            let num_l = count("self_attn.q_proj.weight");
            let v = get_vocab("model.embed_tokens.weight");
            if hd >= 256 {
                return TeacherConfig { embedding_key: "model.embed_tokens.weight".into(), num_layers: num_l, hidden_dim: hd, vocab_size: v.min(128256), arch_type: "llama".into(), prefix_format: "model.layers.{i}".into() };
            } else {
                return TeacherConfig { embedding_key: "model.embed_tokens.weight".into(), num_layers: 0, hidden_dim: hd, vocab_size: v, arch_type: "skip".into(), prefix_format: "model.layers.{i}".into() };
            }
        }
        TeacherConfig { embedding_key: "wte.weight".into(), num_layers: 6, hidden_dim: 768, vocab_size: 50257, arch_type: "unknown".into(), prefix_format: "h.{i}".into() }
    }

      /// اكتشاف أبعاد المعلم تلقائياً
    pub fn detect_config(&self) -> ModelConfig {
        ModelConfig {
            vocab_size: self.config.vocab_size,
            hidden_size: self.config.hidden_dim,
            num_layers: self.config.num_layers,
            num_heads: self.config.hidden_dim / 64,
            kv_heads: (self.config.hidden_dim / 64) / 4,
            num_experts: 8, // تقدير
            top_k: 2,
            max_seq_len: 4096,
            rope_theta: 10000.0,
        }
    }
    
      /// تحويل logits المعلم إلى حجم الطالب
    pub fn adapt_logits(&self, logits: &[f32], target_vocab: usize) -> Vec<f32> {
        let teacher_vocab = self.config.vocab_size;
        
        if teacher_vocab == target_vocab {
            logits.to_vec()
        } else if teacher_vocab > target_vocab {
            // اقتصاص إذا كان المعلم أكبر
            logits[..target_vocab.min(logits.len())].to_vec()
        } else {
            // حشو بالأصفار إذا كان الطالب أكبر
            let mut adapted = vec![0.0f32; target_vocab];
            adapted[..logits.len()].copy_from_slice(logits);
            adapted
        }
    }
}


// ════════════════════════════════════════════════════════════
// load_zlog — تحميل logits من ملف مع التحقق من حجم المفردات
// ════════════════════════════════════════════════════════════
pub fn load_zlog(path: &str, expected_vocab_size: usize) -> Result<Vec<Vec<f32>>> {
    let data = std::fs::read(path)
        .map_err(|e| candle_core::Error::Msg(format!("Cannot read zlog: {}", e)))?;
    
    if data.len() < 12 || &data[0..4] != b"ZLOG" {
        return Err(candle_core::Error::Msg("Invalid zlog format".to_string()));
    }
    
    let num_entries = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let stored_vocab_size = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    
    if stored_vocab_size != expected_vocab_size {
        return Err(candle_core::Error::Msg(format!(
            "Vocab mismatch: {} != {}", stored_vocab_size, expected_vocab_size
        )));
    }
    
    let mut offset = 12;
    let mut logits_list = Vec::with_capacity(num_entries);
    
    for _ in 0..num_entries {
        if offset + 4 > data.len() { break; }
        let len = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        
        if len != expected_vocab_size {
            return Err(candle_core::Error::Msg("Corrupted entry length".to_string()));
        }
        
        if offset + len * 2 > data.len() { break; }
        
        let mut logits = Vec::with_capacity(len);
        for _ in 0..len {
            let bits = u16::from_le_bytes(data[offset..offset+2].try_into().unwrap());
            logits.push(half::f16::from_bits(bits).to_f32());
            offset += 2;
        }
        logits_list.push(logits);
    }
    
    Ok(logits_list)
}

pub fn load_zlog_sequential(path: &str, expected_vocab_size: usize) -> Result<Vec<Vec<f32>>> {
    let data = std::fs::read(path)
        .map_err(|e| candle_core::Error::Msg(format!("Cannot read zlog: {}", e)))?;
    if data.len() < 4 {
        return Err(candle_core::Error::Msg("File too short".into()));
    }
    let mut offset = 0;
    let num_texts = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
    offset += 4;
    let mut all_logits = Vec::with_capacity(num_texts);
    for _ in 0..num_texts {
        if offset + 4 > data.len() { break; }
        let len = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        if len == 0 {
            all_logits.push(Vec::new());
            continue;
        }
        if len != expected_vocab_size {
            return Err(candle_core::Error::Msg(format!("Vocab mismatch: {} != {}", len, expected_vocab_size)));
        }
        let mut logits = Vec::with_capacity(len);
        for _ in 0..len {
            if offset + 2 > data.len() { break; }
            let bits = u16::from_le_bytes(data[offset..offset+2].try_into().unwrap());
            logits.push(half::f16::from_bits(bits).to_f32());
            offset += 2;
        }
        all_logits.push(logits);
    }
    Ok(all_logits)
}

