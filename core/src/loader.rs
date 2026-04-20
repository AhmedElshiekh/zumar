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
        path.push(relative_path);
        Self { base_path: path }
    }

    pub fn get_tokenizer_path(&self) -> String {
        let mut p = self.base_path.clone();
        p.push("tokenizer.json");
        p.to_string_lossy().to_string()
    }

    pub fn load_weights(&self, device: &Device) -> Result<VarBuilder<'static>> {
        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");
        
        if !weights_path.exists() {
            println!("🚀 No weights found. Running LITE Python Injector (NumPy)...");
            
            let status = Command::new("python3")
                .arg("core/src/generate_weights.py")
                .status()
                .map_err(|e| candle_core::Error::Msg(format!("Python Error: {}", e)))?;

            if !status.success() {
                return Err(candle_core::Error::Msg("LITE Weight generation failed".to_string()));
            }
        }

        unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
        }
    }
}
