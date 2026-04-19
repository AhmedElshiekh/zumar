use std::path::PathBuf;
use candle_core::{Device, Result};
//use crate::config::ZumarConfig;

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

    // الحصول على مسار التوكينايزر
    pub fn get_tokenizer_path(&self) -> String {
        let mut p = self.base_path.clone();
        p.push("tokenizer.json");
        p.to_str().unwrap_or("tokenizer.json").to_string()
    }

    // دالة مستقبلية لتحميل أوزان الـ SafeTensors
    pub fn load_weights(&self, _device: &Device) -> Result<()> {
        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");
        
        if weights_path.exists() {
            println!("⚙️ Found weights at: {:?}", weights_path);
            // هنا سنضيف كود candle-core لتحميل المصفوفات
        } else {
            println!("⚠️ No weights found, starting with empty (random) parameters.");
        }
        Ok(())
    }
}
