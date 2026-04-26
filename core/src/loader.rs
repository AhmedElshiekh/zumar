use std::path::PathBuf;
use std::process::Command;
use candle_core::{Device, Result, DType};
use candle_nn::VarBuilder;

pub struct ZumarLoader {
    base_path: PathBuf,
    teacher_dir: PathBuf,
}

impl ZumarLoader {
    pub fn new(relative_path: &str) -> Self {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        
        let mut base_p = path.clone();
        base_p.push(relative_path);
        
        let mut teacher_p = path.clone();
        teacher_p.push("models/teacher");

        Self { base_path: base_p, teacher_dir: teacher_p }
    }

    pub fn get_tokenizer_path(&self) -> String {
        let mut p = self.base_path.clone();
        p.push("tokenizer.json");
        p.to_string_lossy().to_string()
    }

    pub fn load_weights(&self, device: &Device) -> Result<VarBuilder<'static>> {
        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");
        
        // 1. أوزان Zumar موجودة → تحميل
        if weights_path.exists() {
            println!("\x1b[1;32m✅ Found Zumar weights\x1b[0m");
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
            };
            return Ok(vb);
        }
        
        // 2. ابحث عن كل ملفات المعلم
        println!("\x1b[1;33m⚠️  Zumar weights not found\x1b[0m");
        println!("\x1b[1;36m🔍 Searching for teacher models...\x1b[0m");
        
        let teacher_files = self.find_all_teacher_models();
        
        if teacher_files.is_empty() {
            println!("\x1b[1;33m⚠️  No teacher models found. Using random weights.\x1b[0m");
            println!("\x1b[1;33m   💡 Place models in models/teacher/\x1b[0m");
            return Ok(VarBuilder::zeros(DType::F32, device));
        }
        
        // 3. قطّر كل ملف
        println!("\x1b[1;32m📂 Found {} teacher model(s)\x1b[0m", teacher_files.len());
        for (i, file) in teacher_files.iter().enumerate() {
            println!("\n\x1b[1;36m══════════════════════════════════════\x1b[0m");
            println!("\x1b[1;36m🧬 Distilling {}/{}: {}\x1b[0m", 
                i+1, teacher_files.len(), file.file_name().unwrap().to_string_lossy());
            println!("\x1b[1;36m══════════════════════════════════════\x1b[0m");
            
            match self.run_distillation(file) {
                Ok(_) => {
                    println!("\x1b[1;32m   ✅ Distillation successful!\x1b[0m");
                    
                    // استخدام أول تقطير ناجح
                    if weights_path.exists() {
                        println!("\x1b[1;36m📥 Loading distilled weights...\x1b[0m");
                        let vb = unsafe {
                            VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, device)?
                        };
                        return Ok(vb);
                    }
                }
                Err(e) => {
                    println!("\x1b[1;31m   ❌ Distillation failed: {}\x1b[0m", e);
                    println!("\x1b[1;33m   ⏭  Trying next model...\x1b[0m");
                }
            }
        }
        
        // 4. فشل كل شيء
        println!("\x1b[1;33m⚠️  All distillations failed. Using random weights.\x1b[0m");
        Ok(VarBuilder::zeros(DType::F32, device))
    }
    
    /// البحث عن جميع ملفات النماذج المدعومة
    fn find_all_teacher_models(&self) -> Vec<PathBuf> {
        let mut models = Vec::new();
        
        if !self.teacher_dir.exists() {
            return models;
        }
        
        let supported = ["safetensors", "gguf", "pt", "bin", "pth", "h5", "npz", "ckpt"];
        
        if let Ok(entries) = std::fs::read_dir(&self.teacher_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                
                // تأكد من أنه ملف وليس مجلد
                if !path.is_file() {
                    continue;
                }
                
                // تحقق من الامتداد
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if supported.contains(&ext.as_str()) {
                        let size_mb = std::fs::metadata(&path)
                            .map(|m| m.len() as f64 / 1_048_576.0)
                            .unwrap_or(0.0);
                        
                        println!("   📄 {} ({:.1} MB)", 
                            path.file_name().unwrap().to_string_lossy(), size_mb);
                        models.push(path);
                    }
                }
            }
        }
        
        // ترتيب: الأكبر أولاً
        models.sort_by(|a, b| {
            let size_a = std::fs::metadata(a).map(|m| m.len()).unwrap_or(0);
            let size_b = std::fs::metadata(b).map(|m| m.len()).unwrap_or(0);
            size_b.cmp(&size_a)
        });
        
        models
    }
    
    /// تشغيل التقطير لملف معين
    fn run_distillation(&self, teacher_file: &PathBuf) -> Result<()> {
        let mut script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        script_path.push("injection/universal_distill.py");
        
        if !script_path.exists() {
            script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            script_path.push("injection/distill_to_zumar.py");
        }
        
        if !script_path.exists() {
            return Err(candle_core::Error::Msg(
                "Distillation script not found".to_string()
            ));
        }
        
        let output = Command::new("python3")
            .arg(&script_path)
            .arg(teacher_file)
            .output()
            .map_err(|e| candle_core::Error::Msg(
                format!("Cannot run Python: {}", e)
            ))?;
        
        // عرض مخرجات بايثون
        if !output.stdout.is_empty() {
            println!("{}", String::from_utf8_lossy(&output.stdout));
        }
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(candle_core::Error::Msg(
                format!("Distillation error: {}", error)
            ));
        }
        
        Ok(())
    }
}