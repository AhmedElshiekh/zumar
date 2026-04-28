use std::path::PathBuf;
use std::process::Command;
use candle_core::{Device, Result, DType, Tensor};
use candle_nn::VarBuilder;

pub struct ZumarLoader {
    base_path: PathBuf,
    teacher_dir: PathBuf,
    zmr_config: Option<ZmrConfig>,
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
        Self { base_path: base_p, teacher_dir: teacher_p, zmr_config: None }
    }

    pub fn get_tokenizer_path(&self) -> String {
        let mut p = self.base_path.clone();
        p.push("tokenizer.json");
        p.to_string_lossy().to_string()
    }

    pub fn load_weights(&mut self, device: &Device) -> Result<VarBuilder<'static>> {
        let mut zmr_path = self.base_path.clone();
        zmr_path.push("zumar-b1.58.zmr");
        if zmr_path.exists() {
            println!("\x1b[1;32m✅ Found .zmr model (BitNet 1.58-bit)\x1b[0m");
            return self.load_zmr_direct(&zmr_path, device);
        }

        let mut weights_path = self.base_path.clone();
        weights_path.push("model.safetensors");
        if weights_path.exists() {
            println!("\x1b[1;32m✅ Found .safetensors model (FP32)\x1b[0m");
            return unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device) };
        }

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

    fn load_zmr_direct(&mut self, path: &PathBuf, device: &Device) -> Result<VarBuilder<'static>> {
        let data = std::fs::read(path)
            .map_err(|e| candle_core::Error::Msg(format!("Cannot read .zmr: {}", e)))?;

        if data.len() < 24 {
            return Err(candle_core::Error::Msg("Invalid .zmr file".to_string()));
        }
        if &data[0..4] != b"ZUMR" {
            return Err(candle_core::Error::Msg("Not a .zmr file".to_string()));
        }

        let vocab_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let hidden_size = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let num_layers = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
        let num_experts = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;

        self.zmr_config = Some(ZmrConfig { vocab_size, hidden_size, num_layers, num_experts });

        println!("   📊 .zmr: {}d, {}L, {} experts, vocab={}", hidden_size, num_layers, num_experts, vocab_size);
        println!("   ⚡ Loading 2-bit directly (no decompression)...");

        // فك ضغط الأوزان إلى FP32 لبناء safetensors (مرة واحدة للتحميل)
        let mut offset = 24;
        let mut weight_blocks: Vec<Vec<f32>> = Vec::new();
        let map = [-1.0f32, 0.0f32, 1.0f32, 0.0f32];

        while offset + 8 <= data.len() {
            let scale = f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
            offset += 4;
            let num_elements = u32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]) as usize;
            offset += 4;
            let packed_len = (num_elements + 3) / 4;
            if offset + packed_len > data.len() { break; }

            let mut f32_weights = Vec::with_capacity(num_elements);
            for i in 0..packed_len {
                let byte = data[offset + i];
                for bit in 0..4 {
                    if f32_weights.len() >= num_elements { break; }
                    let bits = (byte >> (bit * 2)) & 0b11;
                    f32_weights.push(map[bits as usize] * scale);
                }
            }
            f32_weights.truncate(num_elements);
            offset += packed_len;
            weight_blocks.push(f32_weights);
        }

        println!("   ✅ Decompressed {} blocks", weight_blocks.len());

        // بناء safetensors
        let mut safetensors_bytes = Vec::new();
        let mut header = serde_json::Map::new();
        let mut data_offset = 0usize;
        let mut all_data = Vec::new();
        let mut block_idx = 0;

        let mut add = |name: &str, block: &[f32], shape: Vec<usize>| {
            let mut bytes = Vec::with_capacity(block.len() * 4);
            for &w in block { bytes.extend_from_slice(&w.to_le_bytes()); }
            let len = bytes.len();
            header.insert(name.to_string(), serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [data_offset, data_offset + len]
            }));
            data_offset += len;
            all_data.extend_from_slice(&bytes);
        };

        let ones = vec![1.0f32; hidden_size];
        let zeros = vec![0.0f32; hidden_size];
        let zeros_vocab = vec![0.0f32; vocab_size];
        let zeros_exp = vec![0.0f32; num_experts];

        // Embedding
        if block_idx < weight_blocks.len() {
            add("model.embed_tokens.weight", &weight_blocks[block_idx], vec![vocab_size, hidden_size]);
            block_idx += 1;
        }

        // Layers
        for i in 0..num_layers {
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                if block_idx < weight_blocks.len() {
                    add(&format!("model.layers.{}.self_attn.{}.weight", i, proj), &weight_blocks[block_idx], vec![hidden_size, hidden_size]);
                    add(&format!("model.layers.{}.self_attn.{}.bias", i, proj), &zeros, vec![hidden_size]);
                    block_idx += 1;
                }
            }
            if block_idx < weight_blocks.len() {
                add(&format!("model.layers.{}.mlp.gate.weight", i), &weight_blocks[block_idx], vec![num_experts, hidden_size]);
                add(&format!("model.layers.{}.mlp.gate.bias", i), &zeros_exp, vec![num_experts]);
                block_idx += 1;
            }
            for e in 0..num_experts {
                if block_idx < weight_blocks.len() {
                    add(&format!("model.layers.{}.mlp.expert_{}.weight", i, e), &weight_blocks[block_idx], vec![hidden_size, hidden_size]);
                    add(&format!("model.layers.{}.mlp.expert_{}.bias", i, e), &zeros, vec![hidden_size]);
                    block_idx += 1;
                }
            }
            add(&format!("model.layers.{}.input_layernorm.weight", i), &ones, vec![hidden_size]);
            add(&format!("model.layers.{}.input_layernorm.bias", i), &zeros, vec![hidden_size]);
            add(&format!("model.layers.{}.post_attention_layernorm.weight", i), &ones, vec![hidden_size]);
            add(&format!("model.layers.{}.post_attention_layernorm.bias", i), &zeros, vec![hidden_size]);
        }

        // LM Head
        if block_idx < weight_blocks.len() {
            add("lm_head.weight", &weight_blocks[block_idx], vec![vocab_size, hidden_size]);
        }
        add("lm_head.bias", &zeros_vocab, vec![vocab_size]);

        // Final Norm
        add("model.norm.weight", &ones, vec![hidden_size]);
        add("model.norm.bias", &zeros, vec![hidden_size]);

        // كتابة safetensors مؤقت
        let header_json = serde_json::to_string(&header).unwrap();
        let header_bytes = header_json.as_bytes();
        let header_size = (header_bytes.len() as u64).to_le_bytes();
        safetensors_bytes.extend_from_slice(&header_size);
        safetensors_bytes.extend_from_slice(header_bytes);
        safetensors_bytes.extend_from_slice(&all_data);

        let temp_path = self.base_path.join("_zumar_from_zmr.safetensors");
        std::fs::write(&temp_path, &safetensors_bytes)?;
        let temp_clone = temp_path.clone();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[temp_clone], DType::F32, device)? };
        let _ = std::fs::remove_file(&temp_path);

        println!("   ✅ Loaded ({:.1} MB)", safetensors_bytes.len() as f64 / 1_048_576.0);
        Ok(vb)
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