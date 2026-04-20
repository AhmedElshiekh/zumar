use std::path::PathBuf;
use std::process::Command;
use candle_core::{Device, Result, DType};
use candle_nn::VarBuilder;
use candle_core::safetensors::MmapedSafetensors;

pub struct ZumarLoader {
    base_path: PathBuf,
}

impl ZumarLoader {
    pub fn new(relative_path: &str) -> Self {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push(relative_path);
        Self { base_path: path }
    }

    pub fn get_tokenizer_path(&self) -> String {
        let mut p = self.base_path.clone();
        p.push("tokenizer.json");
        p.to_string_lossy().to_string()
    }

    /// وظيفة الفحص الذكي: تطبع هيكلية المصفوفات للتأكد من مطابقتها للكود
    pub fn inspect_weights(&self) -> Result<()> {
        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");

        if weights_path.exists() {
            println!("🔍 Inspecting ZUMAR Weights Architecture...");
            let tensors = unsafe { MmapedSafetensors::new(&weights_path)? };
            let mut count = 0;
            /*for (name, view) in tensors.tensors() {
                if count < 10 { // طباعة أول 10 مصفوفات فقط لتجنب ازدحام الشاشة
                    println!("   - Found Tensor: {:<30} | Shape: {:?}", name, view.shape());
                }
                count += 1;
            }*/
            println!("   ... total {} tensors found.", count);
        }
        Ok(())
    }

    /// تحميل الأوزان مع خاصية التوليد التلقائي إذا كانت مفقودة
    pub fn load_weights(&self, device: &Device) -> Result<VarBuilder<'static>> {
        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");
        
        // التحقق من وجود الملف، وإذا لم يوجد يتم تشغيل سكريبت التوليد
        if !weights_path.exists() {
            println!("⚠️  Weights not found at: {:?}", weights_path);
            println!("🚀 ZUMAR Sovereign is generating weights automatically...");
            
            // تحديد مسار سكريبت البايثون (يفترض وجوده في core/src/)
            let mut script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            script_path.push("src/generate_weights.py");

            let status = Command::new("python3")
                .arg(&script_path)
                .status()
                .map_err(|e| candle_core::Error::Msg(format!("Python Execution Error: {}", e)))?;

            if !status.success() {
                return Err(candle_core::Error::Msg("Automatic weight generation failed! Check your Python script.".to_string()));
            }
            println!("✅ Weights generated and saved successfully.");
        }

        // نقوم بالفحص السريع قبل التحميل الفعلي
        self.inspect_weights().ok();

        println!("📥 Loading weights into memory...");
        unsafe {
            // تحميل المصفوفات باستخدام Memory Mapping لسرعة قصوى وتقليل استهلاك الرام
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
        }
    }
}
