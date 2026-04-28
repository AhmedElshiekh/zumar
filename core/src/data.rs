use std::fs;
use std::path::Path;

pub struct TrainingData {
    pub texts: Vec<String>,
}

impl TrainingData {
    /// تحميل بيانات من ملف أو مجلد أو استخدام البيانات المدمجة
    pub fn load(path: Option<&str>) -> Self {
        if let Some(p) = path {
            let path = Path::new(p);
            
            if path.is_dir() {
                // 📂 مجلد: اقرأ كل ملفات .txt فيه
                let mut all_texts = Vec::new();
                if let Ok(entries) = fs::read_dir(path) {
                    for entry in entries.flatten() {
                        let file_path = entry.path();
                        if file_path.extension().map_or(false, |e| e == "txt") {
                            if let Ok(content) = fs::read_to_string(&file_path) {
                                let texts: Vec<String> = content
                                    .lines()
                                    .filter(|l| !l.trim().is_empty())
                                    .map(|l| l.to_string())
                                    .collect();
                                println!("   📄 {}: {} lines", file_path.file_name().unwrap().to_string_lossy(), texts.len());
                                all_texts.extend(texts);
                            }
                        }
                    }
                }
                if !all_texts.is_empty() {
                    println!("📚 Loaded {} total texts from folder {}", all_texts.len(), p);
                    return Self { texts: all_texts };
                }
            } else if path.is_file() {
                // 📄 ملف واحد
                if let Ok(content) = fs::read_to_string(path) {
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
        }
        
        // 🆒 افتراضي: البيانات المدمجة
        println!("📚 Using built-in training data");
        Self { texts: builtin_texts() }
    }
    
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
        "Hello how are you doing today".to_string(),
        "I love programming and building new things".to_string(),
        "The best way to learn is by doing".to_string(),
        "Practice makes perfect and patience is key".to_string(),
    ]
}