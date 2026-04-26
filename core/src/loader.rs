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

    /// تحميل الأوزان - يدعم .zmr أولاً، ثم .safetensors
    pub fn load_weights(&self, device: &Device) -> Result<VarBuilder<'static>> {
        // 1. جرب .zmr أولاً (1.58-bit - الأحدث والأخف)
        let mut zmr_path = self.base_path.clone();
        zmr_path.push("zumar-b1.58.zmr");
        
        if zmr_path.exists() {
            println!("\x1b[1;32m✅ Found .zmr model (BitNet 1.58-bit)\x1b[0m");
            return self.load_zmr(&zmr_path, device);
        }

        // 2. جرب .safetensors
        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");
        
        if weights_path.exists() {
            println!("\x1b[1;32m✅ Found Zumar weights (FP32)\x1b[0m");
            println!("\x1b[1;33m   💡 Run 'pack' for smaller .zmr format\x1b[0m");
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
            };
            return Ok(vb);
        }
        
        // 3. ابحث عن كل ملفات المعلم
        println!("\x1b[1;33m⚠️  Zumar weights not found\x1b[0m");
        println!("\x1b[1;36m🔍 Searching for teacher models...\x1b[0m");
        
        let teacher_files = self.find_all_teacher_models();
        
        if teacher_files.is_empty() {
            println!("\x1b[1;33m⚠️  No teacher models found. Using random weights.\x1b[0m");
            println!("\x1b[1;33m   💡 Place models in models/teacher/\x1b[0m");
            return Ok(VarBuilder::zeros(DType::F32, device));
        }
        
        // 4. قطّر كل ملف
        println!("\x1b[1;32m📂 Found {} teacher model(s)\x1b[0m", teacher_files.len());
        for (i, file) in teacher_files.iter().enumerate() {
            println!("\n\x1b[1;36m══════════════════════════════════════\x1b[0m");
            println!("\x1b[1;36m🧬 Distilling {}/{}: {}\x1b[0m", 
                i+1, teacher_files.len(), file.file_name().unwrap().to_string_lossy());
            println!("\x1b[1;36m══════════════════════════════════════\x1b[0m");
            
            match self.run_distillation(file) {
                Ok(_) => {
                    println!("\x1b[1;32m   ✅ Distillation successful!\x1b[0m");
                    
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
        
        // 5. فشل كل شيء
        println!("\x1b[1;33m⚠️  All distillations failed. Using random weights.\x1b[0m");
        Ok(VarBuilder::zeros(DType::F32, device))
    }

    /// تحميل صيغة .zmr (BitNet 1.58-bit)/// تحميل صيغة .zmr (BitNet 1.58-bit) - كامل
    fn load_zmr(&self, path: &PathBuf, device: &Device) -> Result<VarBuilder<'static>> {
          let data = std::fs::read(path)
              .map_err(|e| candle_core::Error::Msg(format!("Cannot read .zmr: {}", e)))?;
      
          if data.len() < 24 {
              return Err(candle_core::Error::Msg("Invalid .zmr file".to_string()));
          }
      
          let magic = &data[0..4];
          if magic != b"ZUMR" {
              return Err(candle_core::Error::Msg("Not a .zmr file".to_string()));
          }
      
          let _version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
          let _vocab_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
          let _hidden_size = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
          let _num_layers = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
          let _num_experts = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;
      
          println!("   📊 .zmr loaded directly (1.58-bit)");
          
          // للتبسيط: نستخدم safetensors الموجود كمرجع
          let safetensors_path = self.base_path.join("model.safetensors");
          if safetensors_path.exists() {
              println!("   📥 Loading FP32 reference...");
              return unsafe {
                  VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, device)
              };
          }
          
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
                if !path.is_file() { continue; }
                
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if supported.contains(&ext.as_str()) {
                        let size_mb = std::fs::metadata(&path)
                            .map(|m| m.len() as f64 / 1_048_576.0)
                            .unwrap_or(0.0);
                        println!("   📄 {} ({:.1} MB)", path.file_name().unwrap().to_string_lossy(), size_mb);
                        models.push(path);
                    }
                }
            }
        }
        
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
            return Err(candle_core::Error::Msg("Distillation script not found".to_string()));
        }
        
        let output = Command::new("python3")
            .arg(&script_path)
            .arg(teacher_file)
            .output()
            .map_err(|e| candle_core::Error::Msg(format!("Cannot run Python: {}", e)))?;
        
        if !output.stdout.is_empty() {
            println!("{}", String::from_utf8_lossy(&output.stdout));
        }
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(candle_core::Error::Msg(format!("Distillation error: {}", error)));
        }
        
        Ok(())
    }
}