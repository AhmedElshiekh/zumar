use std::path::PathBuf;
use std::process::Command;
use candle_core::{Device, Result, DType};
use candle_nn::VarBuilder;

pub struct ZumarLoader {
    base_path: PathBuf,
}

impl ZumarLoader {
    pub fn new(relative_path: &str) -> Self {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop(); 
        path.push(relative_path);
        Self { base_path: path }
    }

    pub fn get_tokenizer_path(&self) -> String {
        let mut p = self.base_path.clone();
        p.push("tokenizer.json");
        if !p.exists() {
            println!("\x1b[1;33m⚠️  Tokenizer missing. Fetching language mind...\x1b[0m");
            let mut script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            script_path.push("injection/get_tokenizer.py");
            let _ = Command::new("python3").arg(&script_path).status();
        }
        p.to_string_lossy().to_string()
    }

    pub fn inspect_weights(&self) -> Result<()> {
        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");
        if weights_path.exists() {
            println!("\x1b[1;34m🔍 Inspecting ZUMAR Weights Architecture...\x1b[0m");
        }
        Ok(())
    }

    pub fn load_weights(&self, device: &Device) -> Result<VarBuilder<'static>> {
        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");
        
        if !weights_path.exists() {
            println!("\x1b[1;33m⚠️  Weights not found. Generating...\x1b[0m");
            let mut script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            script_path.push("injection/generate_weights.py");
            let status = Command::new("python3")
                .arg(&script_path)
                .status()
                .map_err(|e| candle_core::Error::Msg(format!("Python Error: {}", e)))?;
            if !status.success() {
                return Err(candle_core::Error::Msg("Weight generation failed".to_string()));
            }
        }

        self.inspect_weights().ok();
        println!("\x1b[1;36m📥 Loading F16 weights and promoting to F32 engine...\x1b[0m");

        unsafe {
            // نقوم بالتحميل كـ F16 من الملف
            let vb = VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?;
            
            // بدلاً من .cast_f32()، نستخدم .prefix("") لإرجاع الـ VarBuilder 
            // مع التأكد من أن الأوزان ستُطلب كـ F32 لاحقاً في المحرك.
            // إذا استمر الخطأ، فالحل هو تغيير النوع داخل ملفات بناء النموذج (model.rs)
            Ok(vb)
            //Ok(vb.force_dtype(DType::F32))
        }
    }
}
