use candle_core::{Tensor, Result, Device, D};
use candle_nn::{VarMap, Optimizer, Module, AdamW, ParamsAdamW};
use crate::layers::ZumarModel;
use crate::tokenizer::ZumarTokenizer;
use crate::layers::vocab_aligner::{VocabAligner, VocabAlignment};
use crate::layers::ewc::{EWC, DistillCheckpoint};
use std::collections::HashMap;
use std::time::Instant;
use std::path::Path;
use std::cell::RefCell;
use half::{f16, bf16};

const EWC_PATH:   &str = "models/zumar-v1/ewc_state.json";
const CKPT_PATH:  &str = "models/zumar-v1/distill_checkpoint.json";
const MODEL_PATH: &str = "models/zumar-v1/model.safetensors";

// ════════════════════════════════════════════════════════════
// TeacherLogitsCache — نظام التخزين المؤقت الهجين
// ════════════════════════════════════════════════════════════
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
        
        // ضرب cache
        if let Some(cached) = self.cache.get(&key) {
            self.hits += 1;
            if self.hits == 1 {
                eprintln!("🟢 Logits Cache HIT (using cached logits)");
            }
            return Ok(cached.clone());
        }
        
        self.misses += 1;
        if self.misses == 1 {
            eprintln!("🟡 Logits Cache MISS — computing...");
        }
        
        // محاولة causal attention أولاً
        let logits = match teacher.predict_with_causal_attention(tokens) {
            Ok(l) => {
                self.causal_successes += 1;
                if self.causal_successes == 1 {
                    eprintln!("   ✅ Causal Attention SUCCESS");
                }
                l
            }
            Err(_) => {
                self.causal_failures += 1;
                if self.causal_failures == 1 {
                    eprintln!("   ⚠️  Causal Attention failed — fallback to Embeddings");
                }
                teacher.predict_with_embeddings(tokens)?
            }
        };
        // تخطي causal attention لتوفير الذاكرة
      // let logits = teacher.predict_with_embeddings(tokens)?;
      // self.causal_failures += 1; // كلها ستحسب كـ fallback الآن

        // تخزين في cache
        self.cache.insert(key, logits.clone());
        Ok(logits)
    }
    
    pub fn stats(&self) -> (usize, usize, usize, usize) {
        (self.hits, self.misses, self.causal_successes, self.causal_failures)
    }
}

// ════════════════════════════════════════════════════════════
// DistillConfig
// ════════════════════════════════════════════════════════════
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

// ════════════════════════════════════════════════════════════
// TrueDistiller
// ════════════════════════════════════════════════════════════
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
        println!("🧬 MULTI-TEACHER DISTILLATION (Hybrid Cache + Causal + Embedding)");
        println!("   Teachers: {}  |  Samples: {}", teachers.len(), data.len());
        println!("   Resume:   teacher={} epoch={}", checkpoint.teacher_index, checkpoint.epoch);
        ewc.report();
        println!("{}", "═".repeat(60));

        for (idx, (teacher, alignment)) in teachers.iter().enumerate() {
            let name = &teacher.config.arch_type;
            if checkpoint.is_teacher_done(name) && checkpoint.epoch >= self.config.epochs {
                println!("\n   ⏭️  Skipping '{}' (already done with max epochs)", name);
                continue;
            }
            if checkpoint.is_teacher_done(name) && checkpoint.epoch < self.config.epochs {
                println!("\n   🔄 Resuming '{}' for more epochs ({} → {})", 
                    name, checkpoint.epoch, self.config.epochs);
                // نسمح بالمرور للأسفل لبدء التدريب من حيث توقف
            }
            if idx > checkpoint.teacher_index { checkpoint.epoch = 0; }

            println!("\n{}", "═".repeat(60));
            println!("🧬 Teacher {}/{}: {}", idx + 1, teachers.len(), name);
            println!("   {}", alignment.report());
            println!("{}", "═".repeat(60));

            self.distill_one(student, varmap, teacher, alignment, &mut ewc, &mut checkpoint, data, tokenizer)?;
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

        let mut opt = candle_nn::SGD::new(varmap.all_vars(), lr)?;
        // let mut opt = candle_nn::AdamW::new(
        //     varmap.all_vars(),
        //     candle_nn::ParamsAdamW {
        //         lr,
        //         beta1: 0.9,
        //         beta2: 0.999,
        //         eps: 1e-8,
        //         weight_decay: 0.01,
        //     },
        // )?;

        let start = Instant::now();
        let mut total_tokens = 0u64;
        let mut fisher_losses: Vec<Tensor> = Vec::new();
        let mut graph_reset_counter = 0usize;  // جديد

        for epoch in start_epoch..self.config.epochs {
            let mut loss_sum    = 0.0f32;
            let mut token_count = 0u32;
            let mut accum_loss: Option<Tensor> = None;

            for text in data.iter() {
                let zumar_tokens = match tokenizer.encode_ids(text) {
                    Ok(t) if t.len() >= 2 => t,
                    _ => continue,
                };
                let teacher_tokens = teacher.tokenize(text);

                for i in 0..zumar_tokens.len().saturating_sub(1) {
                // let max_tokens = 10.min(zumar_tokens.len().saturating_sub(1));  // نأخذ أول 10 رموز فقط
                // for i in 0..max_tokens {
                    let t_end = (i + 1).min(teacher_tokens.len());

                    // استخدام cache الهجين
                    let teacher_logits = LOGITS_CACHE.with(|cache| {
                        cache.borrow_mut().get_or_compute(&teacher_tokens[..t_end], teacher)
                    })?;
                    if total_tokens == 0 && epoch == start_epoch {
                        let t_sum: f32 = teacher_logits.iter().map(|v| v.abs()).sum();
                        let t_max = teacher_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        eprintln!("🔬 Teacher logits (hybrid): sum={:.4} max={:.4} len={}", t_sum, t_max, teacher_logits.len());
                    }

                    let input = Tensor::new(&zumar_tokens[..=i], &self.device)?;
                    let emb   = student.embedding.forward(&input)?.unsqueeze(0)?;
                    let mut h = emb;
                    for layer in &mut student.layers {
                        h = layer.forward_checkpointed(&h)?;
                    }
                     // أضف هذا السطر:
                    let _ = h.clone(); // نجبر الاحتفاظ بـ h فقط
                    h = student.final_norm.forward(&h)?;
                    
                    let logits  = student.lm_head.forward(&h)?;
                    let seq_len = logits.dim(1)?;
                    let last    = logits.narrow(1, seq_len - 1, 1)?.squeeze(0)?.squeeze(0)?;
                    if total_tokens == 0 && epoch == start_epoch {
                        let s = last.to_vec1::<f32>().unwrap_or_default();
                        let s_sum: f32 = s.iter().map(|v| v.abs()).sum();
                        let s_max = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        eprintln!("🔬 Student logits: sum={:.4} max={:.4} len={}", s_sum, s_max, s.len());
                    }
                    
                    let kl = alignment.kl_divergence(
                        &teacher_logits,
                        &last,
                        self.config.temperature,
                        &self.device,
                    )?;
                      
                    // واستبدل بهذا:
                    let ewc_loss   = ewc.loss_differentiable(varmap, &self.device)?;
                    let total_loss = (&kl + &ewc_loss)?;
                    // let ewc_val = ewc.loss(varmap, &self.device)?;
                    // let ewc_scalar = ewc_val.to_scalar::<f32>().unwrap_or(0.0);
                    // let total_loss = (&kl + ewc_scalar as f64)?;

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
                            if fisher_losses.len() < 10 {
                                fisher_losses.push(loss.clone());
                            }
                            opt.backward_step(&loss)?;
                            
                            // ✅ تحرير computation graph كل 4 خطوات
                            graph_reset_counter += 1;
                            if graph_reset_counter >= 4 {
                                // تفريغ المتغيرات الوسيطة وإجبار تحرير الذاكرة
                                drop(loss);
                                graph_reset_counter = 0;
                            }
                        }
                    }
                }
            }
            
            if let Some(loss) = accum_loss.take() {
                if fisher_losses.len() < 10 {
                    fisher_losses.push(loss.clone());
                }
                opt.backward_step(&loss)?;
                drop(loss);
                graph_reset_counter = 0;
            }
            

            let avg_loss = loss_sum / token_count.max(1) as f32;
            checkpoint.epoch = epoch + 1;
            checkpoint.total_epochs += token_count as usize;
            if avg_loss < checkpoint.best_loss { checkpoint.best_loss = avg_loss; }

            if (epoch + 1) % self.config.save_every == 0 || epoch == self.config.epochs - 1 {
                std::fs::create_dir_all("models/zumar-v1").ok();
                varmap.save(MODEL_PATH)?;
                ewc.save(EWC_PATH)?;
                checkpoint.save(CKPT_PATH)?;
                fisher_losses = fisher_losses.into_iter().rev().take(10).rev().collect();
            }

            let tps = total_tokens as f64 / start.elapsed().as_secs_f64().max(0.1);
            println!("  Ep {:>4}/{}: KL={:.8}  {:.0} tok/s", epoch + 1, self.config.epochs, avg_loss, tps);
        }

        // Fisher
        println!("\n   📐 Computing Fisher for '{}'...", teacher_name);
        let single_loss = fisher_losses.first().map(|l| vec![l.clone()]).unwrap_or_default();
        ewc.update(varmap, &single_loss, &teacher_name, self.config.epochs, &self.device)?;
        ewc.save(EWC_PATH)?;
        // ewc.cumulative_fisher.clear();
        // ewc.cumulative_optimal.clear();
        checkpoint.mark_teacher_done(&teacher_name);
        checkpoint.save(CKPT_PATH)?;

        // إحصائيات cache
        LOGITS_CACHE.with(|c| {
            let (hits, misses, causal_ok, causal_fail) = c.borrow().stats();
            if hits + misses > 0 {
                println!("   📊 Cache: {} hits / {} misses ({:.0}% hit)", hits, misses, hits as f64/(hits+misses) as f64*100.0);
                println!("   📊 Causal: {} OK / {} fallback", causal_ok, causal_fail);
            }
        });

        println!("\n⏱ Total: {:.1}s", start.elapsed().as_secs_f64());
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════
// prepare_alignments_from_dir
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
            println!("   ⚠️  No tokenizer.json for '{}' — limited alignment", name);
            (HashMap::new(), 50257)
        };
        let alignment = aligner.align(&teacher_vocab, t_size, &name);
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
            // GPT-2: c_attn [768, 2304] → [hidden, 3*hidden]
            let attn_hidden = c_attn.dim(0)?; // 768
            let qkv = x.reshape((batch * seq_len, hidden_dim))?
                        .matmul(&c_attn.t()?)?
                        .reshape((batch, seq_len, attn_hidden))?;
            
            let head_dim = attn_hidden / 3;
            let q = qkv.narrow(2, 0, head_dim)?;
            let k = qkv.narrow(2, head_dim, head_dim)?;
            let v = qkv.narrow(2, head_dim * 2, head_dim)?;
            
            // Scaled dot-product attention
            let scale = (head_dim as f64).sqrt();
            let scores = (q.matmul(&k.transpose(1, 2)?)? / scale)?;
            let scores = (scores + mask)?;
            let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
            
            // إسقاط
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
        
        // لا يوجد نمط مدعوم - ارجاع الخطأ
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
}