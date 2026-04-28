use std::path::PathBuf;
use std::process::Command;
use candle_core::{Device, Result, DType};
use candle_nn::VarBuilder;
use crate::layers::PackedBlockRef;

pub struct ZumarLoader {
    pub packed_blocks: Option<Vec<PackedBlockRef>>,
    base_path: PathBuf,
    teacher_dir: PathBuf,
    pub zmr_config: Option<ZmrConfig>,
}

pub struct ZmrConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
}

impl ZumarLoader {
    pub fn new(relative_path: &str) -> Self {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        let mut base_p = path.clone();
        base_p.push(relative_path);
        let mut teacher_p = path.clone();
        teacher_p.push("models/teacher");
        Self { base_path: base_p, teacher_dir: teacher_p, zmr_config: None, packed_blocks: None }
    }

    pub fn get_tokenizer_path(&self) -> String {
        let mut p = self.base_path.clone();
        p.push("tokenizer.json");
        p.to_string_lossy().to_string()
    }

    pub fn load_weights(&mut self, device: &Device) -> Result<VarBuilder<'static>> {
        // 1️⃣ .zmr موجود ← تحميل مباشر (50MB RAM)
        let mut zmr_path = self.base_path.clone();
        zmr_path.push("zumar-b1.58.zmr");
        if zmr_path.exists() {
            println!("\x1b[1;32m✅ Found .zmr model (BitNet 1.58-bit)\x1b[0m");
            println!("\x1b[1;36m   ⚡ Direct 2-bit mode (50MB RAM, no decompression)\x1b[0m");
            return self.load_zmr_packed(&zmr_path, device);
        }

        // 2️⃣ .safetensors موجود ← تحميل عادي (970MB RAM)
        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");
        if weights_path.exists() {
            println!("\x1b[1;33m✅ Found .safetensors model (FP32)\x1b[0m");
            println!("\x1b[1;33m   ⚠️  Full decompression mode (970MB RAM)\x1b[0m");
            return unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device) };
        }

        // 3️⃣ لا شيء ← تقطير من معلم
        println!("\x1b[1;33m⚠️  No model found. Searching for teacher...\x1b[0m");
        let teacher_files = self.find_all_teacher_models();
        if !teacher_files.is_empty() {
            for (i, file) in teacher_files.iter().enumerate() {
                println!("🧬 Distilling {}/{}: {}", i+1, teacher_files.len(), file.file_name().unwrap().to_string_lossy());
                match self.run_distillation(file) {
                    Ok(_) => {
                        if weights_path.exists() {
                            return unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device) };
                        }
                    }
                    Err(e) => println!("   ❌ Failed: {}", e),
                }
            }
        }

        println!("\x1b[1;33m⚠️  Using random weights.\x1b[0m");
        Ok(VarBuilder::zeros(DType::F32, device))
    }

    /// تحميل .zmr مباشرة (packed 2-bit - بدون فك ضغط)
    fn load_zmr_packed(&mut self, path: &PathBuf, device: &Device) -> Result<VarBuilder<'static>> {
        let data = std::fs::read(path)
            .map_err(|e| candle_core::Error::Msg(format!("Cannot read .zmr: {}", e)))?;

        if data.len() < 24 || &data[0..4] != b"ZUMR" {
            return Err(candle_core::Error::Msg("Invalid .zmr file".to_string()));
        }

        let vocab_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let hidden_size = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let num_layers = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
        let num_experts = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;

        self.zmr_config = Some(ZmrConfig { vocab_size, hidden_size, num_layers, num_experts });

        println!("   📊 .zmr: {}d, {}L, {} experts, vocab={}", hidden_size, num_layers, num_experts, vocab_size);

        let mut offset = 24;
        let mut packed_blocks = Vec::new();

        while offset + 8 <= data.len() {
            let scale = f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
            offset += 4;
            let num_elements = u32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]) as usize;
            offset += 4;
            let packed_len = (num_elements + 3) / 4;
            if offset + packed_len > data.len() { break; }

            // تخزين packed مباشرة (بدون فك ضغط!)
            packed_blocks.push(PackedBlockRef {
                data: data[offset..offset+packed_len].to_vec(),
                scale,
            });
            offset += packed_len;
        }

        self.packed_blocks = Some(packed_blocks);
        
        let zmr_size_mb = data.len() as f64 / 1_048_576.0;
        let blocks_count = self.packed_blocks.as_ref().map(|v| v.len()).unwrap_or(0);
        println!("   ✅ Stored {} packed blocks ({:.1} MB in RAM - NO DECOMPRESSION)", blocks_count, zmr_size_mb);

        // إرجاع فارغ - سنستخدم packed_blocks مباشرة
        Ok(VarBuilder::zeros(DType::F32, device))
    }

    pub fn get_zmr_config(&self) -> Option<&ZmrConfig> { self.zmr_config.as_ref() }

    fn find_all_teacher_models(&self) -> Vec<PathBuf> {
        let mut models = Vec::new();
        if !self.teacher_dir.exists() { return models; }
        if let Ok(entries) = std::fs::read_dir(&self.teacher_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().map_or(false, |e| e == "safetensors") {
                    models.push(path);
                }
            }
        }
        models.sort_by(|a, b| {
            let sa = std::fs::metadata(a).map(|m| m.len()).unwrap_or(0);
            let sb = std::fs::metadata(b).map(|m| m.len()).unwrap_or(0);
            sb.cmp(&sa)
        });
        models
    }

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
        let output = Command::new("python3").arg(&script_path).arg(teacher_file).output()
            .map_err(|e| candle_core::Error::Msg(format!("Cannot run Python: {}", e)))?;
        if !output.stdout.is_empty() { println!("{}", String::from_utf8_lossy(&output.stdout)); }
        if !output.status.success() {
            return Err(candle_core::Error::Msg(format!("Distillation error: {}", String::from_utf8_lossy(&output.stderr))));
        }
        Ok(())
    }
}