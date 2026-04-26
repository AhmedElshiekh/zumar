use std::collections::HashMap;

/// نظام استرجاع بسيط (بدون قاعدة بيانات خارجية)
pub struct SimpleRAG {
    /// المستندات المخزنة
    documents: Vec<String>,
    /// تضمينات المستندات (محسوبة مسبقاً)
    embeddings: Vec<Vec<f32>>,
}

impl SimpleRAG {
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
            embeddings: Vec::new(),
        }
    }
    
    /// إضافة مستندات للنظام
    pub fn add_documents(&mut self, docs: Vec<String>) {
        for doc in docs {
            // تضمين بسيط: تردد الكلمات
            let emb = self.simple_embed(&doc);
            self.embeddings.push(emb);
            self.documents.push(doc);
        }
    }
    
    /// تضمين بسيط (TF-IDF مصغر)
    fn simple_embed(&self, text: &str) -> Vec<f32> {
        let mut freq: HashMap<String, f32> = HashMap::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for word in &words {
            *freq.entry(word.to_lowercase()).or_insert(0.0) += 1.0;
        }
        
        // تطبيع
        let total = words.len() as f32;
        freq.values_mut().for_each(|v| *v /= total.max(1.0));
        
        // تحويل إلى متجه (بسيط: أول 256 كلمة فريدة)
        let mut vec = vec![0.0f32; 256];
        for (i, (_, v)) in freq.iter().enumerate().take(256) {
            vec[i] = *v;
        }
        
        vec
    }
    
    /// البحث عن المستندات الأكثر صلة
    pub fn search(&self, query: &str, top_k: usize) -> Vec<String> {
        let query_emb = self.simple_embed(query);
        
        // حساب التشابه (cosine similarity بسيط)
        let mut scores: Vec<(f32, usize)> = self.embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let score = self.cosine_sim(&query_emb, emb);
                (score, i)
            })
            .collect();
        
        // ترتيب تنازلي
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // إرجاع أفضل k
        scores.iter()
            .take(top_k)
            .map(|(_, idx)| self.documents[*idx].clone())
            .collect()
    }
    
    /// تشابه جيب التمام
    fn cosine_sim(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a < 1e-6 || norm_b < 1e-6 {
            return 0.0;
        }
        
        dot / (norm_a * norm_b)
    }
    
    /// تحسين النص المدخل بمعلومات مسترجعة
    pub fn augment_prompt(&mut self, prompt: &str) -> String {
        let docs = self.search(prompt, 3);
        
        if docs.is_empty() {
            return prompt.to_string();
        }
        
        let context = docs.join("\n");
        format!("Context:\n{}\n\nQuestion: {}", context, prompt)
    }
}