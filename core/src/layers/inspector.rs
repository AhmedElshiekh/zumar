use candle_core::{Device, Result};
use std::path::Path;

pub fn inspect_model_weights<P: AsRef<Path>>(model_path: P) -> Result<()> {
    println!("🔍 Inspecting weights in: {:?}", model_path.as_ref());
    
    // البحث عن جميع ملفات safetensors في المجلد
    let files = std::fs::read_dir(model_path)
        .map_err(candle_core::Error::wrap)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
        .collect::<Vec<_>>();

    for file in files {
        let path = file.path();
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path)? };
        
        println!("📄 File: {:?}", path.file_name().unwrap());
        for (name, view) in tensors.tensors() {
            println!("   - Matrix: {:<30} | Shape: {:?} | DType: {:?}", name, view.shape(), view.dtype());
        }
    }
    Ok(())
}
