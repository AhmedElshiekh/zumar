use std::path::PathBuf;
use candle_core::{Device, Result, DType};
use candle_nn::VarBuilder;

pub struct ZumarLoader {
    base_path: PathBuf,
}

impl ZumarLoader {
    pub fn new(relative_path: &str) -> Self {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push(relative_path);
        
        println!("📂 Weights directory set to: {:?}", path);
        Self { base_path: path }
    }

    /// الحصول على مسار التوكينايزر بشكل آمن
    pub fn get_tokenizer_path(&self) -> String {
        let mut p = self.base_path.clone();
        p.push("tokenizer.json");
        p.to_string_lossy().to_string()
    }

    /// تحميل الأوزان باستخدام VarBuilder
    /// تعيد VarBuilder يحتوي على الأوزان من الملف، أو أوزان صفرية إذا لم يوجد
    pub fn load_weights(&self, device: &Device) -> Result<VarBuilder<'_>> {
        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");
        
        if weights_path.exists() {
            println!("⚙️ Loading SafeTensors from: {:?}", weights_path);
            
            // استخدام mmap لتحميل الأوزان بسرعة عالية جداً واستهلاك أقل للرام
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[weights_path], 
                    DType::F32, 
                    device
                )
            }
        } else {
            println!("⚠️ No weights found, initializing with zeroed parameters.");
            // إنشاء VarBuilder وهمي (Zeros) لضمان استمرار تشغيل المحرك حتى بدون ملفات
            Ok(VarBuilder::zeros(DType::F32, device))
        }
    }
}
