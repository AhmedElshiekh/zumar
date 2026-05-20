use std::collections::HashMap;
use candle_core::{Tensor, Result, Device};

// ════════════════════════════════════════════════════════════
// VocabAlignment — خريطة التقاطع بين مفردتين
// ════════════════════════════════════════════════════════════

/// يُمثّل التقاطع بين مفردة المعلم ومفردة الطالب
#[derive(Debug, Clone)]
pub struct VocabAlignment {
    /// معرّفات الكلمات المشتركة في مفردة المعلم
    pub teacher_ids: Vec<u32>,
    /// معرّفات نفس الكلمات في مفردة الطالب
    pub student_ids: Vec<u32>,
    /// الكلمات المشتركة نفسها (للـ debugging)
    pub shared_words: Vec<String>,
    /// نسبة تغطية مفردة المعلم
    pub teacher_coverage: f32,
    /// نسبة تغطية مفردة الطالب
    pub student_coverage: f32,
    /// وزن هذا المعلم في الـ ensemble (بحسب التغطية)
    pub ensemble_weight: f32,
}

impl VocabAlignment {
      /// عدد الكلمات المشتركة
    pub fn overlap_size(&self) -> usize {
        self.shared_words.len()
    }

      /// هل التغطية كافية للتقطير؟
    pub fn is_usable(&self) -> bool {
        self.overlap_size() >= 1000 && self.teacher_coverage >= 0.3
    }

      /// تقرير التغطية
    pub fn report(&self) -> String {
        format!(
            "Shared: {} tokens | Teacher coverage: {:.1}% | Student coverage: {:.1}% | Weight: {:.3}",
            self.overlap_size(),
            self.teacher_coverage * 100.0,
            self.student_coverage * 100.0,
            self.ensemble_weight,
        )
    }

    /// ✅ Projection: استخراج logits الكلمات المشتركة فقط
    /// teacher_logits: [vocab_teacher] → [overlap]
    pub fn project_teacher(&self, teacher_logits: &[f32]) -> Vec<f32> {
        self.teacher_ids.iter()
            .map(|&id| {
                let idx = id as usize;
                if idx < teacher_logits.len() { teacher_logits[idx] } else { 0.0 }
            })
            .collect()
    }

    /// ✅ Projection: استخراج logits الكلمات المشتركة من الطالب
    /// student_logits: Tensor[vocab_student] → Vec[overlap]
    pub fn project_student(&self, student_logits: &Tensor) -> Result<Vec<f32>> {
        let flat = student_logits.flatten_all()?.to_vec1::<f32>()?;
        Ok(self.student_ids.iter()
            .map(|&id| {
                let idx = id as usize;
                if idx < flat.len() { flat[idx] } else { 0.0 }
            })
            .collect())
    }

    /// ✅ Re-normalized softmax على الـ projected logits
      /// ضروري لأن الـ gather يُفقد بعض الـ probability mass
    pub fn softmax_projected(logits: &[f32], temperature: f32) -> Vec<f32> {
        if logits.is_empty() { return vec![]; }

        let temp  = temperature.max(1e-6);
        let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exp: Vec<f32> = logits.iter()
            .map(|&x| ((x - max_v) / temp).exp())
            .collect();

        let sum: f32 = exp.iter().sum();
        if sum < 1e-9 {
            return vec![1.0 / logits.len() as f32; logits.len()];
        }

        exp.iter().map(|&e| e / sum).collect()
    }

    /// ✅ KL Divergence: KL(teacher || student) على الـ projected space
    pub fn kl_divergence(
        &self,
        teacher_logits: &[f32],
        student_logits: &Tensor,
        temperature: f32,
        device: &Device,
    ) -> Result<Tensor> {
        if self.teacher_ids.is_empty() {
            return Ok(Tensor::zeros((), candle_core::DType::F32, device)?);
        }
    
        let temp = temperature as f64;
        let eps = 1e-9f64;
    
        // Teacher projection
        let t_proj: Vec<f32> = self.teacher_ids
            .iter()
            .map(|&id| {
                let idx = id as usize;
                if idx < teacher_logits.len() {
                    teacher_logits[idx]
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
    
        let t_sample_for_diag = t_proj.iter().take(10).copied().collect::<Vec<f32>>();
    
        let t_tensor = Tensor::from_vec(t_proj, self.teacher_ids.len(), device)?;
        let t_probs = candle_nn::ops::softmax(&(&t_tensor / temp)?, 0)?;
    
        // Student projection
        let s_ids = Tensor::new(self.student_ids.as_slice(), device)?;
        let s_proj = student_logits.index_select(&s_ids, 0)?;
        let s_probs = candle_nn::ops::softmax(&(&s_proj / temp)?, 0)?;
    
        // Diagnostic once
        use std::sync::atomic::{AtomicBool, Ordering};
        static FIRST: AtomicBool = AtomicBool::new(true);
        if FIRST.swap(false, Ordering::Relaxed) {
            let t_sum = t_probs.sum_all()?.to_scalar::<f32>()?;
            let s_sum = s_probs.sum_all()?.to_scalar::<f32>()?;
            let t_max = t_probs.max(0)?.to_scalar::<f32>()?;
            let s_max = s_probs.max(0)?.to_scalar::<f32>()?;
            eprintln!("🔍 KL diag: t_probs sum={:.6}, s_probs sum={:.6}", t_sum, s_sum);
            eprintln!("🔍 KL diag: t_probs max={:.6}, s_probs max={:.6}", t_max, s_max);
            eprintln!("🔍 First 10 teacher projected: {:?}", t_sample_for_diag);
            let s_sample = s_proj.to_vec1::<f32>()?.iter().take(10).copied().collect::<Vec<f32>>();
            eprintln!("🔍 First 10 student projected: {:?}", s_sample);
        }
    
        // KL divergence
        let log_t = (t_probs.clone() + eps)?.log()?;
        let log_s = (s_probs.clone() + eps)?.log()?;
        let kl = (&t_probs * (&log_t - &log_s)?)?.sum_all()?;
        let kl_scaled = (kl * (temp * temp))?.reshape(&[])?;  // ← هنا الحل
    
        let kl_val = kl_scaled.to_scalar::<f32>()?;
        if kl_val.is_nan() || kl_val.is_infinite() {
            eprintln!("⚠️ KL returned NaN/Inf, returning zero tensor");
            return Ok(Tensor::zeros((), candle_core::DType::F32, device)?);
        }
    
        Ok(kl_scaled)
    }
    
      /// تقليص عدد الكلمات المشتركة لتسريع حسابات KL
    pub fn truncate(&mut self, max_size: usize) {
        if self.teacher_ids.len() > max_size {
            self.teacher_ids.truncate(max_size);
            self.student_ids.truncate(max_size);
            self.shared_words.truncate(max_size);
        }
    }
    
    /// Cross Entropy Loss بدلاً من KL (أسرع بكثير)
    pub fn cross_entropy_loss(
        &self,
        teacher_logits: &[f32],
        student_logits: &Tensor,
        temperature: f32,
        device: &Device,
    ) -> Result<Tensor> {
        if self.teacher_ids.is_empty() {
            return Ok(Tensor::zeros((), candle_core::DType::F32, device)?);
        }
    
        let temp = temperature as f64;
        
        // Teacher: softmax للحصول على التوزيع الاحتمالي
        let t_proj: Vec<f32> = self.teacher_ids
            .iter()
            .map(|&id| teacher_logits.get(id as usize).copied().unwrap_or(f32::NEG_INFINITY))
            .collect();
        let t_tensor = Tensor::from_vec(t_proj, self.teacher_ids.len(), device)?;
        let t_probs = candle_nn::ops::softmax(&(&t_tensor / temp)?, 0)?;
        
        // أخذ فئة (class) ذات أعلى احتمال من المعلم (Hard Distillation)
        let teacher_class = t_probs.argmax(0)?.to_scalar::<u32>()?;
        
        // Student: استخراج logits للكلمات المشتركة
        let s_ids = Tensor::new(self.student_ids.as_slice(), device)?;
        let s_proj = student_logits.index_select(&s_ids, 0)?;
        
        // ✅ التصحيح: target يجب أن يكون [1] وليس [1, 1]
        let target = Tensor::new(&[teacher_class], device)?;
        
        // Cross Entropy Loss
        let loss = candle_nn::loss::cross_entropy(
            &s_proj.unsqueeze(0)?,  // input: [1, vocab_size]
            &target,                 // target: [1]
        )?;
        
        Ok(loss.reshape(&[])?)
    }
    
    pub fn mse_loss(
        &self,
        teacher_logits: &[f32],
        student_logits: &Tensor,
        temperature: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let temp = temperature as f64;
        
        // Teacher projection
        let t_proj: Vec<f32> = self.teacher_ids
            .iter()
            .map(|&id| teacher_logits.get(id as usize).copied().unwrap_or(0.0))
            .collect();
        let t_tensor = Tensor::from_vec(t_proj, self.teacher_ids.len(), device)?;
        let t_scaled = (&t_tensor / temp)?;
        
        // Student projection
        let s_ids = Tensor::new(self.student_ids.as_slice(), device)?;
        let s_proj = student_logits.index_select(&s_ids, 0)?;
        let s_scaled = (&s_proj / temp)?;
        
        // MSE Loss (أسرع بكثير من KL)
        let diff = (&t_scaled - &s_scaled)?;
        let loss = diff.powf(2.0)?.mean_all()?;
        
        Ok(loss.reshape(&[])?)
    }
}

// ════════════════════════════════════════════════════════════
// VocabAligner — يبني خرائط التقاطع لجميع المعلمين
// ════════════════════════════════════════════════════════════

pub struct VocabAligner {
    /// مفردة الطالب: كلمة → id
    student_vocab: HashMap<String, u32>,
    /// حجم مفردة الطالب
    student_vocab_size: usize,
}

impl VocabAligner {
    pub fn new(student_vocab: HashMap<String, u32>, student_vocab_size: usize) -> Self {
        Self { student_vocab, student_vocab_size }
    }

    /// بناء من ملف tokenizer.json للطالب
    pub fn from_tokenizer_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| candle_core::Error::Msg(format!("Cannot read {}: {}", path, e)))?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::Msg(format!("JSON error: {}", e)))?;

        let mut vocab = HashMap::new();

        // صيغة tokenizers: {"model": {"vocab": {"word": id}}}
        if let Some(v) = json.get("model").and_then(|m| m.get("vocab")) {
            if let Some(obj) = v.as_object() {
                for (word, id) in obj {
                    if let Some(id) = id.as_u64() {
                        vocab.insert(Self::normalize_token(word), id as u32);
                    }
                }
            }
        }
        // صيغة مباشرة: {"vocab": {"word": id}}
        else if let Some(v) = json.get("vocab") {
            if let Some(obj) = v.as_object() {
                for (word, id) in obj {
                    if let Some(id) = id.as_u64() {
                        vocab.insert(Self::normalize_token(word), id as u32);
                    }
                }
            }
        }

        let size = vocab.values().max().copied().unwrap_or(0) as usize + 1;
        println!("   📚 Student vocab loaded: {} tokens", vocab.len());

        Ok(Self { student_vocab: vocab, student_vocab_size: size })
    }

    /// ✅ بناء VocabAlignment بين مفردة معلم ومفردة الطالب
    pub fn align(
        &self,
        teacher_vocab:      &HashMap<String, u32>,
        teacher_vocab_size: usize,
        teacher_name:       &str,
    ) -> VocabAlignment {
        let mut teacher_ids   = Vec::new();
        let mut student_ids   = Vec::new();
        let mut shared_words  = Vec::new();

        // تصفية الرموز الخاصة والنادرة
        let skip_prefixes = ["<", "[", "▁<", "Ġ<"];

        for (word, &t_id) in teacher_vocab {
            // تجاهل الرموز الخاصة
            if skip_prefixes.iter().any(|p| word.starts_with(p)) { continue; }
            if word.len() < 2 { continue; }  // تجاهل الأحرف المنفردة

            let normalized = Self::normalize_token(word);

            if let Some(&s_id) = self.student_vocab.get(&normalized) {
                teacher_ids.push(t_id);
                student_ids.push(s_id);
                shared_words.push(word.clone());
            }
        }

        // ترتيب بحسب teacher_id للاتساق
        let mut triples: Vec<(u32, u32, String)> = teacher_ids.into_iter()
            .zip(student_ids.into_iter())
            .zip(shared_words.into_iter())
            .map(|((t, s), w)| (t, s, w))
            .collect();
        triples.sort_by_key(|(t, _, _)| *t);

        let teacher_ids:  Vec<u32>   = triples.iter().map(|(t, _, _)| *t).collect();
        let student_ids:  Vec<u32>   = triples.iter().map(|(_, s, _)| *s).collect();
        let shared_words: Vec<String> = triples.into_iter().map(|(_, _, w)| w).collect();

        let overlap          = shared_words.len();
        let teacher_coverage = overlap as f32 / teacher_vocab_size.max(1) as f32;
        let student_coverage = overlap as f32 / self.student_vocab_size.max(1) as f32;

        // وزن الـ ensemble بحسب جذر التغطية (لتجنب هيمنة المعلم ذي الـ vocab الكبير)
        let ensemble_weight  = teacher_coverage.sqrt();

        println!("   🔗 {} alignment: {} shared tokens ({:.1}% teacher, {:.1}% student)",
            teacher_name, overlap,
            teacher_coverage * 100.0,
            student_coverage * 100.0,
        );

        VocabAlignment {
            teacher_ids,
            student_ids,
            shared_words,
            teacher_coverage,
            student_coverage,
            ensemble_weight,
        }
    }

    /// بناء مفردة معلم من ملف safetensors header
    pub fn extract_teacher_vocab_from_safetensors(path: &str) -> Result<HashMap<String, u32>> {
        // نحاول قراءة tokenizer.json بجانب النموذج
        let base   = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
        let tok_path = base.join("tokenizer.json");

        if tok_path.exists() {
            return Self::load_vocab_from_tokenizer_json(tok_path.to_str().unwrap());
        }

        // إذا لم يوجد tokenizer.json، نُعيد مفردة فارغة
        println!("   ⚠️  No tokenizer.json found beside model — vocab alignment will be limited");
        Ok(HashMap::new())
    }

    /// تحميل مفردة من tokenizer.json
    pub fn load_vocab_from_tokenizer_json(path: &str) -> Result<HashMap<String, u32>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| candle_core::Error::Msg(format!("Cannot read {}: {}", path, e)))?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::Msg(format!("JSON error: {}", e)))?;

        let mut vocab = HashMap::new();

        // محاولة صيغ مختلفة
        let vocab_json = json.get("model")
            .and_then(|m| m.get("vocab"))
            .or_else(|| json.get("vocab"));

        if let Some(v) = vocab_json {
            if let Some(obj) = v.as_object() {
                for (word, id) in obj {
                    if let Some(id) = id.as_u64() {
                        vocab.insert(Self::normalize_token(word), id as u32);
                    }
                }
            }
        }

        Ok(vocab)
    }

    /// ✅ تطبيع الـ token لجعل المقارنة ممكنة بين أنواع tokenizers مختلفة
    ///
    /// GPT-2 يستخدم Ġ للمسافات: "Ġcat" → "cat"
    /// Llama يستخدم ▁ للمسافات:  "▁cat" → "cat"
    /// BERT يستخدم ## للأجزاء:   "##ing" → "ing"
    fn normalize_token(token: &str) -> String {
        token
            .replace('Ġ', "")      // GPT-2 space prefix
            .replace('▁', "")      // SentencePiece space prefix
            .replace("##", "")     // BERT continuation
            .replace("</w>", "")   // BPE end-of-word
            .replace("<0x", "")    // byte tokens
            .to_lowercase()
    }
}

// ════════════════════════════════════════════════════════════
// MultiTeacherAlignment — إدارة خرائط جميع المعلمين
// ════════════════════════════════════════════════════════════

pub struct MultiTeacherAlignment {
    pub alignments:   Vec<(String, VocabAlignment)>,  // (teacher_name, alignment)
    total_weight:     f32,
}

impl MultiTeacherAlignment {
    pub fn new() -> Self {
        Self { alignments: Vec::new(), total_weight: 0.0 }
    }

    pub fn add(&mut self, name: String, alignment: VocabAlignment) {
        self.total_weight += alignment.ensemble_weight;
        self.alignments.push((name, alignment));
    }

    /// وزن كل معلم بعد التطبيع
    pub fn normalized_weight(&self, idx: usize) -> f32 {
        if self.total_weight < 1e-9 { return 1.0 / self.alignments.len().max(1) as f32; }
        self.alignments[idx].1.ensemble_weight / self.total_weight
    }

    /// ✅ KL مجمّع من جميع المعلمين (weighted ensemble)
    pub fn ensemble_kl(
        &self,
        teacher_logits_all: &[Vec<f32>],   // logits كل معلم
        student_logits:     &Tensor,
        temperature:         f32,
        device:              &Device,
    ) -> Result<Tensor> {
        if self.alignments.is_empty() {
            return Tensor::new(&[0.0f32], device);
        }

        let mut total_kl = Tensor::new(&[0.0f32], device)?;

        for (idx, (_, alignment)) in self.alignments.iter().enumerate() {
            if idx >= teacher_logits_all.len() { break; }
            if !alignment.is_usable() {
                println!("   ⚠️  Skipping teacher {} — insufficient overlap", idx);
                continue;
            }

            let kl     = alignment.kl_divergence(
                &teacher_logits_all[idx],
                student_logits,
                temperature,
                device,
            )?;
            let weight = self.normalized_weight(idx);
            let kl_w   = (kl * weight as f64)?;
            total_kl   = (total_kl + kl_w)?;
        }

        Ok(total_kl)
    }

    pub fn report(&self) {
        println!("\n   📊 Vocabulary Alignment Summary:");
        for (idx, (name, alignment)) in self.alignments.iter().enumerate() {
            println!("     [{}] {} → {} | w={:.3}",
                idx, name,
                alignment.report(),
                self.normalized_weight(idx),
            );
        }
    }
}
