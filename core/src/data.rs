use std::fs;

/// بيانات تدريب بسيطة مدمجة
pub struct TrainingData {
    pub texts: Vec<String>,
}

impl TrainingData {
    /// تحميل بيانات من ملف أو استخدام البيانات المدمجة
    pub fn load(path: Option<&str>) -> Self {
        if let Some(p) = path {
            if let Ok(content) = fs::read_to_string(p) {
                let texts: Vec<String> = content
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| l.to_string())
                    .collect();
                if !texts.is_empty() {
                    println!("📚 Loaded {} texts from {}", texts.len(), p);
                    return Self { texts };
                }
            }
        }
        
        println!("📚 Using built-in training data");
        Self {
            texts: builtin_texts(),
        }
    }
    
    /// تكرار البيانات لعدد معين من المرات
    pub fn repeat(&self, times: usize) -> Vec<String> {
        let mut result = Vec::new();
        for _ in 0..times {
            result.extend(self.texts.clone());
        }
        result
    }
}

fn builtin_texts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog.".to_string(),
        "Artificial intelligence is transforming the world.".to_string(),
        "Machine learning models can learn from data.".to_string(),
        "Language models generate text based on patterns.".to_string(),
        "Deep learning uses neural networks with many layers.".to_string(),
        "Natural language processing helps computers understand text.".to_string(),
        "Transformers use attention mechanisms for sequence modeling.".to_string(),
        "Knowledge distillation transfers knowledge from large models to small ones.".to_string(),
        "The future of AI is efficient and accessible to everyone.".to_string(),
        "Hello how are you doing today".to_string(),
        "I love programming and building new things".to_string(),
        "Science and technology advance together".to_string(),
        "The earth revolves around the sun".to_string(),
        "Water is essential for all known forms of life".to_string(),
        "Mathematics is the language of the universe".to_string(),
        "Music can express emotions that words cannot".to_string(),
        "History teaches us lessons for the future".to_string(),
        "Reading books expands our knowledge and imagination".to_string(),
        "Friendship is one of the most valuable things in life".to_string(),
        "Innovation comes from thinking differently".to_string(),
        "The best way to learn is by doing".to_string(),
        "Practice makes perfect and patience is key".to_string(),
        "Every day is a new opportunity to grow".to_string(),
        "Success is the sum of small efforts repeated daily".to_string(),
        "Curiosity is the engine of achievement".to_string(),
        "We are building the future together".to_string(),
        "Technology should serve humanity".to_string(),
        "Simple solutions are often the best".to_string(),
        "Learning never stops no matter how old you are".to_string(),
        "The journey begins with a single step".to_string(),
    ]
}