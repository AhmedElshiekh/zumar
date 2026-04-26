mod layers;
mod routing;
mod config;
mod tokenizer;
mod loader;
mod kv_cache;
mod kernels;
mod data;
mod distill;
mod train;
mod true_distill;

use crate::layers::ZumarModel;
use crate::layers::bitlinear::ZumarBitLinear;
use candle_core::{Result, Tensor};
use candle_nn::VarMap;
use std::io::{self, Write};

fn print_banner() {
    println!("\x1b[1;35m");
    println!(r#"
    ███████╗██╗   ██╗███╗   ███╗ █████╗ ██████╗ 
    ╚══███╔╝██║   ██║████╗ ████║██╔══██╗██╔══██╗
      ███╔╝ ██║   ██║██╔████╔██║███████║██████╔╝
     ███╔╝  ██║   ██║██║╚██╔╝██║██╔══██║██╔══██╗
    ███████╗╚██████╔╝██║ ╚═╝ ██║██║  ██║██║  ██║
    ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
    "#);
    println!("\x1b[0m");
    println!("\x1b[1;36m--- 🌌 ZUMAR SOVEREIGN CORE v6.0 ---\x1b[0m");
    println!("\x1b[90m--------------------------------------------------\x1b[0m");
}

fn print_usage() {
    println!("\nUsage:");
    println!("  distill <epochs>     - True distillation from ALL safetensors in teacher/");
    println!("  train <epochs>       - Self-training on built-in data");
    println!("  chat                 - Chat mode (default)");
    println!("  pack                 - Export to .zmr (1.58-bit) + .gguf");
    println!("\nExamples:");
    println!("  cargo run -p core --release -- distill 1000");
    println!("  cargo run -p core --release -- train 100");
    println!("  cargo run -p core --release");
}

#[tokio::main]
async fn main() -> Result<()> {
    print!("\x1b[2J");
    print_banner();
    
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("chat");
    
    let hidden_size: usize = 1024;
    let num_layers: usize = 12;
    let n_heads: usize = 16;
    let vocab_size: usize = 50257;
    let num_experts: usize = 8;
    let top_k: usize = 2;
    
    let router = routing::HardwareRouter::new();
    let device = router.route("Inference Task");
    
    match mode {
        // ==========================================
        // تقطير حقيقي من كل النماذج في teacher/
        // ==========================================
        "distill" => {
            print_usage();
            println!("\n🧠 TRUE KNOWLEDGE DISTILLATION (ALL MODELS)\n");
            
            let teacher_dir = std::path::Path::new("models/teacher");
            if !teacher_dir.exists() {
                println!("\x1b[1;31m❌ models/teacher/ not found\x1b[0m");
                return Ok(());
            }
            
            let mut teacher_files: Vec<std::path::PathBuf> = Vec::new();
            if let Ok(entries) = std::fs::read_dir(teacher_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map_or(false, |e| e == "safetensors") {
                        teacher_files.push(path);
                    }
                }
            }
            
            teacher_files.sort_by(|a, b| {
                let sa = std::fs::metadata(a).map(|m| m.len()).unwrap_or(0);
                let sb = std::fs::metadata(b).map(|m| m.len()).unwrap_or(0);
                sb.cmp(&sa)
            });
            
            if teacher_files.is_empty() {
                println!("\x1b[1;31m❌ No safetensors found in models/teacher/\x1b[0m");
                return Ok(());
            }
            
            println!("\x1b[1;32m📂 Found {} teacher model(s):\x1b[0m", teacher_files.len());
            for f in &teacher_files {
                let size = std::fs::metadata(f).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
                println!("   📄 {} ({:.1} MB)", f.file_name().unwrap().to_string_lossy(), size);
            }
            
            let total_epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
            let save_interval = 100;
            
            for (i, teacher_path) in teacher_files.iter().enumerate() {
                println!("\n{}", "=".repeat(60));
                println!("🧬 Model {}/{}: {}", i + 1, teacher_files.len(), teacher_path.file_name().unwrap().to_string_lossy());
                println!("{}", "=".repeat(60));
                
                let varmap = VarMap::new();
                let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
                let mut model = ZumarModel::new(vocab_size, hidden_size, num_layers, num_experts, top_k, n_heads, vs.clone())?;
                let training_data = data::TrainingData::load(None);
                let all_texts = training_data.repeat(10);
                let mut all_ok = true;
                
                for chunk_start in (0..total_epochs).step_by(save_interval) {
                    let chunk_epochs = std::cmp::min(save_interval, total_epochs - chunk_start);
                    println!("\n🔄 Section {}/{}, epochs {}-{}", chunk_start / save_interval + 1, (total_epochs + save_interval - 1) / save_interval, chunk_start + 1, chunk_start + chunk_epochs);
                    
                    let config = true_distill::DistillConfig { epochs: chunk_epochs, learning_rate: 0.001, temperature: 3.0 };
                    let distiller = true_distill::TrueDistiller::new(config, device.clone());
                    
                    match distiller.distill(&mut model, &varmap, teacher_path.to_str().unwrap(), &all_texts) {
                        Ok(_) => {
                            let global_epoch = chunk_start + chunk_epochs;
                            let save_dir = std::path::Path::new("models/zumar-v1");
                            std::fs::create_dir_all(save_dir).ok();
                            let save_path = save_dir.join("model.safetensors");
                            if let Err(e) = varmap.save(&save_path) {
                                println!("\x1b[1;31m⚠️  Save failed: {}\x1b[0m", e);
                            } else {
                                let size_mb = std::fs::metadata(&save_path).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
                                println!("\x1b[1;32m💾 Saved at epoch {} ({:.1} MB)\x1b[0m", global_epoch, size_mb);
                            }
                        }
                        Err(e) => { println!("\x1b[1;31m❌ Failed at epoch {}: {}\x1b[0m", chunk_start, e); all_ok = false; break; }
                    }
                }
                
                if all_ok {
                    println!("\n\x1b[1;32m✅ All {} epochs complete!\x1b[0m", total_epochs);
                    
                    // ---- تصدير تلقائي بعد اكتمال التقطير ----
                    println!("\n\x1b[1;33m📦 Auto-exporting to .zmr + .gguf...\x1b[0m");
                    export_formats(&device, vocab_size, hidden_size, num_layers, num_experts, top_k, n_heads)?;
                    
                    println!("\n\x1b[1;36m🚀 Run: cargo run -p core --release\x1b[0m");
                    break;
                }
            }
        }
        
        // ==========================================
        // تصدير الصيغ (.zmr + .gguf)
        // ==========================================
        "pack" => {
            println!("\n📦 EXPORTING TO .zmr + .gguf (BitNet 1.58-bit)\n");
            export_formats(&device, vocab_size, hidden_size, num_layers, num_experts, top_k, n_heads)?;
        }
        
        // ==========================================
        // تدريب ذاتي
        // ==========================================
        "train" => {
            print_usage();
            println!("\n🎓 SELF-TRAINING MODE\n");
            let varmap = VarMap::new();
            let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
            let mut model = ZumarModel::new(vocab_size, hidden_size, num_layers, num_experts, top_k, n_heads, vs.clone())?;
            let epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            println!("   Config: {}d, {}L, {}h, {} experts", hidden_size, num_layers, n_heads, num_experts);
            println!("   Epochs: {}\n", epochs);
            train::run_training(&mut model, &varmap, &device, None, epochs)?;
            let save_dir = std::path::Path::new("models/zumar-v1");
            std::fs::create_dir_all(save_dir)?;
            varmap.save(save_dir.join("model.safetensors"))?;
            println!("\n💾 Saved!");
            println!("🚀 Run: cargo run -p core --release");
        }
        
        // ==========================================
        // مساعدة
        // ==========================================
        "help" | "--help" | "-h" => { print_usage(); }
        
        // ==========================================
        // محادثة (افتراضي)
        // ==========================================
        _ => {
            println!("\n💬 Chat Mode\n");
            
            // اكتشاف نوع النموذج
            let zmr_exists = std::path::Path::new("models/zumar-v1/zumar-b1.58.zmr").exists();
            let safetensors_exists = std::path::Path::new("models/zumar-v1/model.safetensors").exists();
            
            if zmr_exists {
                println!("\x1b[1;32m🚀 Using 1.58-bit .zmr model (fast!)\x1b[0m");
            } else if safetensors_exists {
                println!("\x1b[1;33m⚠️  Using FP32 .safetensors (slow)\x1b[0m");
                println!("\x1b[1;33m   Run 'pack' to create smaller .zmr\x1b[0m");
            }
            
            let loader = loader::ZumarLoader::new("models/zumar-v1");
        
            let vb = loader.load_weights(&device)?;
            println!("🔧 Building model...");
            let mut model = ZumarModel::new(vocab_size, hidden_size, num_layers, num_experts, top_k, n_heads, vb.clone())?;
            println!("✅ Ready.\n");
            
            let temperature: f64 = 0.8;
            let max_tokens: usize = 120;
            let penalty: f32 = 1.2;
            println!("\x1b[1;34m💡 Type your message or 'exit' to quit.\x1b[0m\n");
            
            loop {
                print!("\x1b[1;32mYou>\x1b[0m ");
                io::stdout().flush().ok();
                let mut input = String::new();
                if io::stdin().read_line(&mut input).is_err() { break; }
                let prompt = input.trim();
                if prompt == "exit" || prompt == "quit" { break; }
                if prompt.is_empty() { continue; }
                
                let tokens: Vec<u32> = prompt.chars().map(|c| (c as u32 % 256) + 3).collect();
                let mut current = *tokens.last().unwrap_or(&1);
                let start = std::time::Instant::now();
                let mut generated = Vec::new();
                
                print!("\x1b[1;36mZumar>\x1b[0m ");
                io::stdout().flush().ok();
                
                for _ in 0..max_tokens {
                    let emb = match model.embed(current, &device) { Ok(e) => e, Err(_) => break };
                    let out = match model.forward(&emb) { Ok(o) => o, Err(_) => break };
                    let flat = match out.flatten_all() { Ok(f) => f, Err(_) => break };
                    let v = match flat.to_vec1::<f32>() { Ok(vec) => vec, Err(_) => break };
                    
                    let mut logits = v.clone();
                    for &prev in &generated { let idx = prev as usize; if idx < logits.len() { logits[idx] /= penalty; } }
                    
                    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature as f32).collect();
                    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
                    let sum: f32 = exp.iter().sum();
                    let probs: Vec<f32> = exp.iter().map(|&v| v / (sum + 1e-9)).collect();
                    
                    let mut best_val = f32::NEG_INFINITY;
                    let mut best_idx = 0u32;
                    for (i, &val) in probs.iter().enumerate().take(512) { if val > best_val { best_val = val; best_idx = i as u32; } }
                    
                    current = best_idx;
                    generated.push(best_idx);
                    if best_idx > 3 && best_idx < 260 { print!("{}", (best_idx - 3) as u8 as char); }
                    io::stdout().flush().ok();
                    if best_idx == 1 { break; }
                }
                
                let elapsed = start.elapsed();
                let n = generated.len();
                let tps = if elapsed.as_secs_f64() > 0.0 { n as f64 / elapsed.as_secs_f64() } else { 0.0 };
                println!();
                println!("\x1b[90m──────────────────────────────────────\x1b[0m");
                println!("\x1b[90m📊 {} tokens in {:.1}s ({:.1} tok/s)\x1b[0m", n, elapsed.as_secs_f64(), tps);
            }
            println!("\n\x1b[1;35m🛡️  ZUMAR SHUTTING DOWN\x1b[0m");
        }
    }
    Ok(())
}

// ============================================================
// دالة التصدير المشتركة (.zmr + .gguf)
// ============================================================
fn export_formats(
    device: &candle_core::Device,
    vocab_size: usize, hidden_size: usize, num_layers: usize,
    num_experts: usize, top_k: usize, n_heads: usize,
) -> Result<()> {
    let save_path = std::path::Path::new("models/zumar-v1").join("model.safetensors");
    if !save_path.exists() {
        println!("\x1b[1;31m❌ No model found. Train first.\x1b[0m");
        return Ok(());
    }
    
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[save_path.clone()], candle_core::DType::F32, device)? };
    let model = ZumarModel::new(vocab_size, hidden_size, num_layers, num_experts, top_k, n_heads, vb)?;
    
    let orig_mb = std::fs::metadata(&save_path).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
    
    // Helper: تكميم BitNet b1.58 (-1, 0, +1)
    let quantize_bitnet = |data: &[f32]| -> (f32, Vec<u8>, Vec<u8>) {
        let sum_abs: f32 = data.iter().map(|v| v.abs()).sum();
        let scale = sum_abs / data.len() as f32;
        let scale = if scale < 1e-6 { 1.0 } else { scale };
        
        // 2-bit per weight: 00=-1, 01=0, 10=+1
        let mut q_2bit = Vec::with_capacity((data.len() + 3) / 4);
        // 1.58-bit unpacked (1 byte per weight for .zmr)
        let mut q_zmr = Vec::with_capacity(data.len());
        
        for chunk in data.chunks(4) {
            let mut byte: u8 = 0;
            for (i, &val) in chunk.iter().enumerate() {
                let ternary: u8 = if val / scale <= -0.33 { 0u8 }       // -1
                else if val / scale >= 0.33 { 2u8 }                      // +1
                else { 1u8 };                                             // 0
                byte |= ternary << (i * 2);
                q_zmr.push(ternary);
            }
            q_2bit.push(byte);
        }
        // أكمل .zmr إذا احتاج
        while q_zmr.len() < data.len() { q_zmr.push(1); }
        
        (scale, q_2bit, q_zmr)
    };
    
    // ---- بناء .zmr ----
    let mut zmr_data = Vec::new();
    zmr_data.extend_from_slice(b"ZUMR");                    // Magic
    zmr_data.extend_from_slice(&1u32.to_le_bytes());        // Version
    zmr_data.extend_from_slice(&(vocab_size as u32).to_le_bytes());
    zmr_data.extend_from_slice(&(hidden_size as u32).to_le_bytes());
    zmr_data.extend_from_slice(&(num_layers as u32).to_le_bytes());
    zmr_data.extend_from_slice(&(num_experts as u32).to_le_bytes());
    
    // ---- بناء GGUF ----
    let mut gguf_data = Vec::new();
    gguf_data.extend_from_slice(b"GGUF");
    gguf_data.extend_from_slice(&3u32.to_le_bytes());
    let tensor_count = 1 + 1 + (num_layers as u64) * (4 + 1 + num_experts as u64 + 2);
    gguf_data.extend_from_slice(&tensor_count.to_le_bytes());
    let n_kv = 6u64;
    gguf_data.extend_from_slice(&n_kv.to_le_bytes());
    
    // Metadata helper
    let wms = |d: &mut Vec<u8>, k: &str, v: &str| {
        d.extend_from_slice(&(k.len() as u64).to_le_bytes());
        d.extend_from_slice(k.as_bytes());
        d.extend_from_slice(&8u32.to_le_bytes());
        d.extend_from_slice(&(v.len() as u64).to_le_bytes());
        d.extend_from_slice(v.as_bytes());
    };
    let wmu = |d: &mut Vec<u8>, k: &str, v: u32| {
        d.extend_from_slice(&(k.len() as u64).to_le_bytes());
        d.extend_from_slice(k.as_bytes());
        d.extend_from_slice(&4u32.to_le_bytes());
        d.extend_from_slice(&v.to_le_bytes());
    };
    
    wms(&mut gguf_data, "general.architecture", "zumar");
    wms(&mut gguf_data, "zumar.quantization", "BitNet_b1.58");
    wmu(&mut gguf_data, "zumar.hidden_size", hidden_size as u32);
    wmu(&mut gguf_data, "zumar.num_layers", num_layers as u32);
    wmu(&mut gguf_data, "zumar.num_experts", num_experts as u32);
    wmu(&mut gguf_data, "zumar.vocab_size", vocab_size as u32);
    
    let mut gguf_tensor_infos: Vec<(String, u32, Vec<u32>, Vec<u8>)> = Vec::new();
    
    // معالجة كل الأوزان
    let mut process_weight = |name: &str, data: &[f32], shape: Vec<u32>, zmr: &mut Vec<u8>, gguf_info: &mut Vec<(String, u32, Vec<u32>, Vec<u8>)>| {
        let (scale, q_2bit, q_zmr) = quantize_bitnet(data);
        zmr.extend_from_slice(&scale.to_le_bytes());
        zmr.extend_from_slice(&(data.len() as u32).to_le_bytes());
        zmr.extend_from_slice(&q_zmr);
        
        let mut tdata = Vec::new();
        tdata.extend_from_slice(&scale.to_le_bytes());
        tdata.extend_from_slice(&q_2bit);
        gguf_info.push((name.to_string(), 7, shape, tdata));
    };
    
    // Embedding
    let emb = model.embedding.embeddings().flatten_all()?.to_vec1::<f32>()?;
    process_weight("model.embed_tokens.weight", &emb, vec![vocab_size as u32, hidden_size as u32], &mut zmr_data, &mut gguf_tensor_infos);
    
    // Layers
    for i in 0..num_layers {
        let layer = &model.layers[i];
        for (pn, proj) in [("q_proj", &layer.q_proj), ("k_proj", &layer.k_proj), ("v_proj", &layer.v_proj), ("o_proj", &layer.o_proj)] {
            let w = proj.latent_weight.flatten_all()?.to_vec1::<f32>()?;
            process_weight(&format!("model.layers.{}.self_attn.{}.weight", i, pn), &w, vec![hidden_size as u32, hidden_size as u32], &mut zmr_data, &mut gguf_tensor_infos);
        }
        let gw = layer.moe.gate.latent_weight.flatten_all()?.to_vec1::<f32>()?;
        process_weight(&format!("model.layers.{}.mlp.gate.weight", i), &gw, vec![num_experts as u32, hidden_size as u32], &mut zmr_data, &mut gguf_tensor_infos);
        for e in 0..num_experts {
            let ew = layer.moe.experts[e].latent_weight.flatten_all()?.to_vec1::<f32>()?;
            process_weight(&format!("model.layers.{}.mlp.expert_{}.weight", i, e), &ew, vec![hidden_size as u32, hidden_size as u32], &mut zmr_data, &mut gguf_tensor_infos);
        }
        for norm_name in ["input_layernorm", "post_attention_layernorm"] {
            gguf_tensor_infos.push((format!("model.layers.{}.{}.weight", i, norm_name), 1, vec![hidden_size as u32], vec![1u8; hidden_size * 2]));
        }
    }
    
    // LM Head
    let head = model.lm_head.latent_weight.flatten_all()?.to_vec1::<f32>()?;
    process_weight("lm_head.weight", &head, vec![vocab_size as u32, hidden_size as u32], &mut zmr_data, &mut gguf_tensor_infos);
    
    // Final norm
    gguf_tensor_infos.push(("model.norm.weight".to_string(), 1, vec![hidden_size as u32], vec![1u8; hidden_size * 2]));
    
    // ---- كتابة GGUF data ----
    let mut offset = gguf_data.len() as u64 + tensor_count * (8 + 4 + 8 * 3 + 4 + 8);
    for (name, dtype, dims, data) in &gguf_tensor_infos {
        gguf_data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        gguf_data.extend_from_slice(name.as_bytes());
        gguf_data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims { gguf_data.extend_from_slice(&(d as u64).to_le_bytes()); }
        gguf_data.extend_from_slice(&dtype.to_le_bytes());
        gguf_data.extend_from_slice(&offset.to_le_bytes());
        offset += data.len() as u64;
    }
    for (_, _, _, data) in &gguf_tensor_infos { gguf_data.extend_from_slice(data); }
    
    // ---- حفظ ----
    let zmr_path = std::path::Path::new("models/zumar-v1").join("zumar-b1.58.zmr");
    let gguf_path = std::path::Path::new("models/zumar-v1").join("zumar-b1.58.gguf");
    std::fs::write(&zmr_path, &zmr_data)?;
    std::fs::write(&gguf_path, &gguf_data)?;
    
    let zmr_mb = zmr_data.len() as f64 / 1_048_576.0;
    let gguf_mb = gguf_data.len() as f64 / 1_048_576.0;
    
    println!("\n╔══════════════════════════════════════╗");
    println!("║  📦 EXPORT COMPLETE                  ║");
    println!("╠══════════════════════════════════════╣");
    println!("║  Original (FP32):  {:>8.1} MB        ║", orig_mb);
    println!("║  .zmr (1.58-bit):  {:>8.1} MB        ║", zmr_mb);
    println!("║  .gguf (1.58-bit): {:>8.1} MB        ║", gguf_mb);
    println!("║  Ratio:            {:>8.1}x smaller   ║", orig_mb / zmr_mb.max(0.1));
    println!("╠══════════════════════════════════════╣");
    println!("║  Saved to:                           ║");
    println!("║  {}          ║", zmr_path.display());
    println!("║  {}          ║", gguf_path.display());
    println!("╚══════════════════════════════════════╝");
    println!("\n\x1b[1;36m🚀 Use .zmr with:   cargo run -p core --release");
    println!("🚀 Use .gguf with:  ./llama-cli -m {} -p \"Hello\"\x1b[0m", gguf_path.display());
    
    Ok(())
}