use std::collections::{HashMap, BTreeMap};

const EMB_DIM: usize = 256;

/// نظام استرجاع بسيط (بدون قاعدة بيانات خارجية)
pub struct SimpleRAG {
    documents:  Vec<String>,
    embeddings: Vec<Vec<f32>>,
    /// ✅ مفردات مرتبة ثابتة — تضمن نفس المتجه لنفس النص دائماً
    vocab:      Vec<String>,
}

impl SimpleRAG {
    pub fn new() -> Self {
        Self {
            documents:  Vec::new(),
            embeddings: Vec::new(),
            vocab:      Vec::new(),
        }
    }

    pub fn add_documents(&mut self, docs: Vec<String>) {
        // ── بناء المفردات الموحدة أولاً ─────────────────────────
        // نجمع كل الكلمات من جميع المستندات الجديدة
        for doc in &docs {
            for word in doc.split_whitespace() {
                let w = word.to_lowercase();
                if !self.vocab.contains(&w) {
                    self.vocab.push(w);
                }
            }
        }
        // ✅ ترتيب أبجدي ثابت — HashMap العشوائي هو المشكلة القديمة
        self.vocab.sort();
        self.vocab.dedup();
        self.vocab.truncate(EMB_DIM);

        // ── إعادة حساب embeddings جميع المستندات السابقة ─────────
        // ضروري لأن المفردات تغيّرت
        self.embeddings = self.documents.iter()
            .map(|d| self.embed_with_vocab(d))
            .collect();

        // ── إضافة المستندات الجديدة ──────────────────────────────
        for doc in docs {
            let emb = self.embed_with_vocab(&doc);
            self.embeddings.push(emb);
            self.documents.push(doc);
        }
    }

    /// ✅ تضمين حتمي: يستخدم self.vocab المرتبة أبجدياً
    /// نفس النص → نفس المتجه دائماً بغض النظر عن ترتيب الإدراج
    fn embed_with_vocab(&self, text: &str) -> Vec<f32> {
        // حساب تردد الكلمات
        let mut freq: BTreeMap<&str, f32> = BTreeMap::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let total = words.len() as f32;

        for word in &words {
            *freq.entry(word).or_insert(0.0) += 1.0 / total.max(1.0);
        }

        // ✅ ملء المتجه بترتيب self.vocab الثابت
        let mut vec = vec![0.0f32; EMB_DIM];
        for (i, vocab_word) in self.vocab.iter().enumerate() {
            if let Some(&v) = freq.get(vocab_word.as_str()) {
                vec[i] = v;
            }
        }

        // تطبيع L2
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            vec.iter_mut().for_each(|x| *x /= norm);
        }

        vec
    }

    /// البحث عن المستندات الأكثر صلة
    pub fn search(&self, query: &str, top_k: usize) -> Vec<String> {
        if self.documents.is_empty() {
            return Vec::new();
        }

        let query_emb = self.embed_with_vocab(query);

        let mut scores: Vec<(f32, usize)> = self.embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (self.cosine_sim(&query_emb, emb), i))
            .collect();

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal));

        scores.iter()
            .take(top_k)
            .filter(|(score, _)| *score > 1e-6)  // تخطي نتائج عديمة الصلة
            .map(|(_, idx)| self.documents[*idx].clone())
            .collect()
    }

    fn cosine_sim(&self, a: &[f32], b: &[f32]) -> f32 {
        // المتجهات مطبّعة مسبقاً — الجداء النقطي = cosine similarity
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    pub fn augment_prompt(&mut self, prompt: &str) -> String {
        let docs = self.search(prompt, 3);
        if docs.is_empty() {
            return prompt.to_string();
        }
        format!("Context:\n{}\n\nQuestion: {}", docs.join("\n---\n"), prompt)
    }

    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl Default for SimpleRAG {
    fn default() -> Self { Self::new() }
}
