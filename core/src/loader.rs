use std::path::PathBuf;
use std::process::Command;
use candle_core::{Device, Result, DType};
use candle_nn::VarBuilder;

pub struct ZumarLoader {
    base_path: PathBuf,
    teacher_path: PathBuf,
}

impl ZumarLoader {
    pub fn new(relative_path: &str) -> Self {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop(); // الخروج من core/ للوصول للمجلد الرئيسي zumar/

        let mut teacher_p = path.clone();
        teacher_p.push("models/teacher/model.safetensors");

        let mut base_p = path.clone();
        base_p.push(relative_path);

        Self { 
            base_path: base_p,
            teacher_path: teacher_p,
        }
    }

    pub fn get_tokenizer_path(&self) -> String {
        let mut p = self.base_path.clone();
        p.push("tokenizer.json");
        if !p.exists() {
            println!("\x1b[1;33m⚠️  Tokenizer missing. Fetching logic-gate maps...\x1b[0m");
            let mut script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            script_path.push("injection/get_tokenizer.py");
            let _ = Command::new("python3").arg(&script_path).status();
        }
        p.to_string_lossy().to_string()
    }

    pub fn load_weights(&self, device: &Device) -> Result<VarBuilder<'static>> {
        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");
        
        // 1. التحقق من وجود الأوزان النهائية (model.safetensors)
        if !weights_path.exists() {
            println!("\x1b[1;33m⚠️  Zumar Brain (model.safetensors) not found.\x1b[0m");

            // 2. التحقق من وجود المعلم (Teacher) قبل استدعاء البايثون
            if !self.teacher_path.exists() {
                println!("\x1b[1;31m❌ Error: Teacher model missing!\x1b[0m");
                println!("\x1b[1;36m💡 Please place the teacher model in: models/teacher/model.safetensors\x1b[0m");
                // التوقف عن العمل لأننا لا نستطيع توليد أوزان بدون معلم
                return Err(candle_core::Error::Msg("Missing Teacher Model".to_string()));
            }

            // 3. إذا وجد المعلم، نقوم بتشغيل سكريبت الحقن
            println!("\x1b[1;32m✅ Teacher model found. Starting Logic Injection via Python...\x1b[0m");
            let mut script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            script_path.push("injection/distill_to_zumar.py");
            
            let status = Command::new("python3")
                .arg(&script_path)
                .status()
                .map_err(|e| candle_core::Error::Msg(format!("Python Execution Error: {}", e)))?;
            
            if !status.success() {
                return Err(candle_core::Error::Msg("Logic Injection Failed during execution.".to_string()));
            }
            
            println!("\x1b[1;32m✨ Injection Complete. Proceeding to load weights...\x1b[0m");
        }

        println!("\x1b[1;36m📥 ZUMAR Engine: Mapping Sovereign Intelligence to RAM...\x1b[0m");

        // استخدام mmap لتقليل استهلاك الرام في أندرويد
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
        };

        Ok(vb)
    }
}
