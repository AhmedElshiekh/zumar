// mod ewc;
// mod vocab_aligner;
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

// use crate::layers::vocab_aligner::VocabAlignment;
use crate::layers::vocab_aligner;
use crate::layers::ZumarModel;
use candle_core::Result;
use candle_nn::VarMap;
use std::io::{self, Write};
use crate::tokenizer::ZumarTokenizer;
use crate::layers::config::ModelConfig;
use crate::layers::ZumarModelDynamic;

use std::fs;
use std::path::PathBuf;

const MODELS_BASE_DIR: &str = "models/zumar";

/// هيكل المجلدات المقترحة:
/// models/zumar/
/// ├── 80M/
/// │   ├── model.safetensors
/// │   ├── ewc_state.json
/// │   └── distill_checkpoint.json
/// ├── 400M/
/// ├── 1.5B/
/// └── 7B/

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
    println!("  extract [--teacher NAME] [--force]  - Extract logits for a specific teacher");
    println!("  distill <epochs> [--teachers NAMES] - True distillation (resumes from last save)");
    println!("  train <epochs>       - Self-training on built-in data");
    println!("  chat                 - Chat mode (default)");
    println!("  pack                 - Export to .zmr + .gguf");
    println!("\nExamples:");
    println!("  # Extract from all teachers");
    println!("  cargo run -- extract /data");
    println!("\n  # Extract from a single teacher only");
    println!("  cargo run -- extract /data --teacher llama-13b");
    println!("\n  # Force re-extract (overwrite existing)");
    println!("  cargo run -- extract /data --teacher jais-13b --force");
    println!("\n  # Distill from specific teachers");
    println!("  cargo run -- distill 100 /data --teachers llama,jais");
}

#[tokio::main]
async fn main() -> Result<()> {
    print_banner();
    
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("chat");
    
    let hidden_size: usize = 512;
    let num_layers: usize = 12;
    let n_heads: usize = 16;   // بدلاً من 16
    let kv_heads: usize = 4;  // جديد: رأس واحد لـ K و V
    let vocab_size: usize = 50257;
    let num_experts: usize = 6;
    let top_k: usize = 2;
    
    let router = routing::HardwareRouter::new();
    let device = router.route("Inference Task");
    
    match mode {
        "distill" => {
            print_usage();
            println!("\n🧠 TRUE KNOWLEDGE DISTILLATION (ALL MODELS - CUMULATIVE)\n");
            
            distill_runner(
              &args, &device, 
              vocab_size, hidden_size, 
              num_layers, num_experts, 
              top_k, n_heads
            )?;
            
            
        }

        "pack" => {
            println!("\n📦 EXPORTING TO .zmr + .gguf (BitNet 1.58-bit)\n");
            let varmap = candle_nn::VarMap::new();
            
            // تحديد النموذج المراد تصديره
            let output_dir = if let Some(size_arg) = parse_model_size_interactive(&args) {
                // تم تحديد --size في سطر الأوامر
                let size_name = format_model_size_name(size_arg);
                format!("{}/{}", MODELS_BASE_DIR, size_name)
            } else {
                // عرض النماذج المتاحة واختيار واحد
                let models = list_available_models();
                if models.is_empty() {
                    println!("   ❌ No trained models found in {}/", MODELS_BASE_DIR);
                    println!("   Train a model first: cargo run -- distill 100 /data --size 1.5B");
                    return Ok(());
                }
                
                println!("\n📦 Available models to export:");
                for (i, (name, size, path)) in models.iter().enumerate() {
                    let model_file = path.join("model.safetensors");
                    let size_mb = fs::metadata(&model_file)
                        .map(|m| m.len() as f64 / 1_048_576.0)
                        .unwrap_or(0.0);
                    println!("   {}. {} ({:.1}B) - {:.0}MB", i + 1, name, size, size_mb);
                }
                
                print!("\nSelect model to export (1-{}): ", models.len());
                io::stdout().flush().unwrap();
                let mut input = String::new();
                io::stdin().read_line(&mut input).ok();
                let idx = input.trim().parse::<usize>().unwrap_or(0);
                
                if idx >= 1 && idx <= models.len() {
                    format!("{}/{}", MODELS_BASE_DIR, models[idx - 1].0)
                } else {
                    println!("   Invalid selection. Using default: models/zumar-v1");
                    "models/zumar-v1".to_string()
                }
            };
            
            // التحقق من وجود ملف النموذج
            let model_path = std::path::Path::new(&output_dir).join("model.safetensors");
            if !model_path.exists() {
                println!("\n❌ No model found in: {}", output_dir);
                println!("   Please train a model first with the correct size.");
                println!("   Example: cargo run -- distill 100 /data --size 1.5B\n");
                return Ok(());
            }
            
            println!("📁 Exporting from: {}", output_dir);
            
            // قراءة أبعاد النموذج من ملف التكوين أو من مجلد الإخراج
            let (vocab_size, hidden_size, num_layers, num_experts, top_k, n_heads) = 
                parse_model_dimensions_from_path(&output_dir)?;
            
            export_formats(
                &varmap, &device,
                vocab_size, hidden_size, num_layers,
                num_experts, top_k, n_heads,
                &output_dir,
            )?;
        }
        
        "train" => {
            print_usage();
            println!("\n🎓 SELF-TRAINING MODE\n");
            let varmap = VarMap::new();
            // let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
            // let mut model = ZumarModel::new(vocab_size, hidden_size, num_layers, num_experts, top_k, n_heads, vs.clone())?;
            // // تفعيل QLoRA
            // println!("   🧬 Activating QLoRA (NF4 + LoRA rank=8)...");
            // model.add_qlora(8, 16.0)?;
            // بناء النموذج مع QLoRA مباشرة
          let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
          let mut model = ZumarModel::new_qlora(
              vocab_size, hidden_size, num_layers,
              num_experts, top_k, n_heads, vs,
              8,   // rank
              16.0 // alpha
          )?;
            let epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            train::run_training(&mut model, &varmap, &device, None, epochs)?;
            let save_dir = std::path::Path::new("models/zumar-v1");
            std::fs::create_dir_all(save_dir)?;
            varmap.save(save_dir.join("model.safetensors"))?;
            println!("\n💾 Saved!");
        }
        
        "extract" => {
            println!("\n📡 SIGNAL EXTRACTION MODE\n");
            signal_extractor(&args, &device, vocab_size, hidden_size, num_layers, num_experts, top_k, n_heads)?;
            println!("\n✅ Signal extraction complete! Run: cargo run -- distill 10 /data");
        }
        
        "list-teachers" => {
            println!("\n📚 Available teachers:\n");
            let teacher_dir = std::path::Path::new("models/teacher");
            if teacher_dir.exists() {
                let mut paths = Vec::new();
                collect_teacher_files(teacher_dir, &mut paths);
                for path in paths {
                    let name = path.parent()
                        .and_then(|p| p.file_name())
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown");
                    let mb = std::fs::metadata(&path).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
                    println!("   • {} ({:.1} MB)", name, mb);
                }
            } else {
                println!("   ❌ No teachers found in models/teacher/");
            }
            return Ok(());
        }
        
        // "resume" => {
        //     println!("\n📋 Distillation Status:\n");
        //     let checkpoint = true_distill::DistillCheckpoint::load(CKPT_PATH)?;
        //     println!("   Teacher index: {}", checkpoint.teacher_index);
        //     println!("   Current epoch: {}", checkpoint.epoch);
        //     println!("   Total epochs: {}", checkpoint.total_epochs);
        //     println!("   Best loss: {:.6}", checkpoint.best_loss);
        //     println!("   Completed teachers: {:?}", checkpoint.done_teachers);
        //     return Ok(());
        // }
        
        // "reset" => {
        //     println!("\n🔄 Resetting distillation checkpoint...\n");
        //     let checkpoint = true_distill::DistillCheckpoint::new();
        //     checkpoint.save(CKPT_PATH)?;
        //     println!("   ✅ Checkpoint reset successfully!");
        //     return Ok(());
        // }
        
        "list-models" => {
            println!("\n📦 Available trained models:\n");
            let models = list_available_models();
            if models.is_empty() {
                println!("   ❌ No models found in {}/", MODELS_BASE_DIR);
                println!("   Train a model first: cargo run -- distill 100 /data --size 1.5B");
            } else {
                for (name, size, path) in models {
                    let model_file = path.join("model.safetensors");
                    let size_mb = fs::metadata(&model_file)
                        .map(|m| m.len() as f64 / 1_048_576.0)
                        .unwrap_or(0.0);
                    println!("   • {} ({:.1}B) - {:.0}MB", name, size, size_mb);
                }
            }
            return Ok(());
        }


        "help" | "--help" | "-h" => { print_usage(); }
        
        _ => {
            println!("\n💬 Chat Mode\n");
            
            // ✅ عرض النماذج المتاحة واختيار واحد للتشغيل المباشر
            let models = list_available_models();
            
            let selected_model = if let Some(size_arg) = parse_model_size_interactive(&args) {
                // تم تحديد --size في سطر الأوامر
                format_model_size_name(size_arg)
            } else if !models.is_empty() {
                // عرض قائمة تفاعلية واختيار نموذج للتشغيل
                println!("📂 Available trained models:\n");
                for (i, (name, size, path)) in models.iter().enumerate() {
                    let model_file = path.join("model.safetensors");
                    let size_mb = fs::metadata(&model_file)
                        .map(|m| m.len() as f64 / 1_048_576.0)
                        .unwrap_or(0.0);
                    let zmr_file = path.join("zumar-b1.58.zmr");
                    let has_zmr = zmr_file.exists();
                    println!("   {}. {} ({:.1}B) - {:.0}MB {}", 
                        i + 1, name, size, size_mb,
                        if has_zmr { "📦" } else { "✓" }
                    );
                }
                println!("\n   \x1b[1;33m0. Train new model\x1b[0m");
                print!("\n   \x1b[1;36mSelect model (1-{}): \x1b[0m", models.len());
                io::stdout().flush().unwrap();
                
                let mut input = String::new();
                io::stdin().read_line(&mut input).ok();
                let idx = input.trim().parse::<usize>().unwrap_or(0);
                
                if idx == 0 {
                    // اختيار تدريب نموذج جديد
                    println!("\n\x1b[1;33m🚀 Training new model...\x1b[0m");
                    println!("   Please specify model size:");
                    println!("   Example: --size 1.5B or --size 400M\n");
                    println!("   Or run: cargo run -- distill 100 /data --size 1.5B");
                    return Ok(());
                } else if idx >= 1 && idx <= models.len() {
                    models[idx - 1].0.clone()
                } else {
                    println!("\x1b[1;31m❌ Invalid selection. Exiting.\x1b[0m");
                    return Ok(());
                }
            } else {
                // لا يوجد نماذج مدربة
                println!("\x1b[1;33m⚠️  No trained models found.\x1b[0m");
                println!("\n📦 Available options:");
                println!("   1. Train a new model: cargo run -- distill 100 /data --size 1.5B");
                println!("   2. Exit\n");
                print!("   \x1b[1;36mSelect (1-2): \x1b[0m");
                io::stdout().flush().unwrap();
                
                let mut input = String::new();
                io::stdin().read_line(&mut input).ok();
                let choice = input.trim().parse::<usize>().unwrap_or(2);
                
                if choice == 1 {
                    println!("\n\x1b[1;33m🚀 Please run the training command manually:\x1b[0m");
                    println!("   cargo run -- distill 100 /data --size 1.5B\n");
                } else {
                    println!("\n\x1b[1;35mExiting...\x1b[0m");
                }
                return Ok(());
            };
            
            let model_dir = format!("{}/{}", MODELS_BASE_DIR, selected_model);
            println!("\n📂 Loading model from: {}", model_dir);
            
            // ✅ تحميل الأوزان من المجلد الصحيح
            let mut loader = loader::ZumarLoader::new(&model_dir);
            
            // محاولة تحميل النموذج المضغوط أولاً
            let zmr_path = std::path::Path::new(&model_dir).join("zumar-b1.58.zmr");
            let safetensors_path = std::path::Path::new(&model_dir).join("model.safetensors");
            
            if !zmr_path.exists() && !safetensors_path.exists() {
                println!("\x1b[1;31m❌ No model weights found in {}\x1b[0m", model_dir);
                println!("   Please train the model first:");
                println!("   cargo run -- distill 100 /data --size {}\n", selected_model);
                return Ok(());
            }
            
            let _ = loader.load_weights(&device)?;
            
            let (v, h, l, e) = if let Some(cfg) = loader.get_zmr_config() {
                (cfg.vocab_size, cfg.hidden_size, cfg.num_layers, cfg.num_experts)
            } else {
                let config = ModelConfig::from_params_billions(parse_size_from_folder_name(&selected_model).unwrap_or(0.083));
                (config.vocab_size, config.hidden_size, config.num_layers, config.num_experts)
            };
            
            println!("🔧 Building model ({}d, {}L, {} experts)...", h, l, e);
            
            let mut model = if let Some(ref packed) = loader.packed_blocks {
                println!("   ⚡ Direct 2-bit mode ({} blocks)", packed.len());
                ZumarModel::from_packed_blocks(
                    v, h, l, e, n_heads, 
                    packed,
                    loader.packed_embedding.as_ref(),
                    &device,
                )?
            } else if safetensors_path.exists() {
                println!("   📦 Using .safetensors (FP32)");
                let vb = unsafe { 
                    candle_nn::VarBuilder::from_mmaped_safetensors(
                        &[safetensors_path], candle_core::DType::F32, &device
                    )? 
                };
                ZumarModel::new(v, h, l, e, top_k, n_heads, vb)?
            } else {
                println!("\x1b[1;31m❌ Cannot load model\x1b[0m");
                return Ok(());
            };
            
            println!("✅ Ready! Type 'exit' to quit.\n");
            
            let temperature: f64 = 0.8;
            let max_tokens: usize = 120;
            let penalty: f32 = 1.2;
            
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
                    let emb = match model.embed(current, &device) {
                        Ok(e) => {
                            let e = e.to_dtype(candle_core::DType::F32).unwrap_or(e);
                            e
                        },
                        Err(_) => break
                    };
                    let out = match model.forward(&emb) { Ok(o) => o, Err(_) => break };
                    let flat = match out.flatten_all() { Ok(f) => f, Err(_) => break };
                    let v = match flat.to_vec1::<f32>() { Ok(vec) => vec, Err(_) => break };
                    
                    let mut logits = v.clone();
                    for &prev in &generated { 
                        let idx = prev as usize; 
                        if idx < logits.len() { 
                            logits[idx] /= penalty; 
                        } 
                    }
                    
                    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature as f32).collect();
                    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
                    let sum: f32 = exp.iter().sum();
                    let probs: Vec<f32> = exp.iter().map(|&v| v / (sum + 1e-9)).collect();
                    
                    let mut best_val = f32::NEG_INFINITY;
                    let mut best_idx = 0u32;
                    for (i, &val) in probs.iter().enumerate().take(512) { 
                        if val > best_val { 
                            best_val = val; 
                            best_idx = i as u32; 
                        } 
                    }
                    
                    current = best_idx;
                    generated.push(best_idx);
                    if best_idx > 3 && best_idx < 260 { 
                        print!("{}", (best_idx - 3) as u8 as char); 
                    }
                    io::stdout().flush().ok();
                    if best_idx == 1 { break; }
                }
                
                let elapsed = start.elapsed();
                let n = generated.len();
                let tps = if elapsed.as_secs_f64() > 0.0 { n as f64 / elapsed.as_secs_f64() } else { 0.0 };
                println!();
                println!("\x1b[90m📊 {} tokens in {:.1}s ({:.1} tok/s)\x1b[0m", n, elapsed.as_secs_f64(), tps);
            }
            println!("\n\x1b[1;35m🛡️  ZUMAR SHUTTING DOWN\x1b[0m");
        }
    }
    Ok(())
}


fn export_formats(
    varmap: &candle_nn::VarMap,
    device: &candle_core::Device,
    vocab_size: usize, hidden_size: usize, num_layers: usize,
    num_experts: usize, top_k: usize, n_heads: usize,
    output_dir: &str,
) -> Result<()> {
    let save_path = std::path::Path::new(output_dir).join("model.safetensors");
    
    if !save_path.exists() {
        println!("\x1b[1;31m❌ No model found in {}\x1b[0m", output_dir);
        println!("   Train first: cargo run -- distill 100 /data --size 1.5B");
        return Ok(());
    }
    
    let vs = candle_nn::VarBuilder::from_varmap(varmap, candle_core::DType::F32, device);
    let mut model = ZumarModel::new_qlora(
        vocab_size, hidden_size, num_layers,
        num_experts, top_k, n_heads, vs,
        8, 16.0
    )?;
    
    let orig_mb = std::fs::metadata(&save_path).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
    println!("\x1b[1;33m🔢 Quantizing to BitNet b1.58 (2-bit packed)...\x1b[0m");
    
    let quantize_bitnet = |data: &[f32]| -> (f32, Vec<u8>) {
        let sum_abs: f32 = data.iter().map(|v| v.abs()).sum();
        let scale = sum_abs / data.len() as f32;
        let scale = if scale < 1e-6 { 1.0 } else { scale };
        let mut packed = Vec::with_capacity((data.len() + 3) / 4);
        for chunk in data.chunks(4) {
            let mut byte: u8 = 0;
            for (i, &val) in chunk.iter().enumerate() {
                let ternary: u8 = if val / scale <= -0.33 { 0b00 } else if val / scale >= 0.33 { 0b10 } else { 0b01 };
                byte |= ternary << (i * 2);
            }
            packed.push(byte);
        }
        (scale, packed)
    };
    
    let mut zmr_data = Vec::new();
    zmr_data.extend_from_slice(b"ZUMR");
    zmr_data.extend_from_slice(&1u32.to_le_bytes());
    zmr_data.extend_from_slice(&(vocab_size as u32).to_le_bytes());
    zmr_data.extend_from_slice(&(hidden_size as u32).to_le_bytes());
    zmr_data.extend_from_slice(&(num_layers as u32).to_le_bytes());
    zmr_data.extend_from_slice(&(num_experts as u32).to_le_bytes());
    
    let mut gguf_data = Vec::new();
    gguf_data.extend_from_slice(b"GGUF");
    gguf_data.extend_from_slice(&3u32.to_le_bytes());
    let tensor_count = 1 + 1 + (num_layers as u64) * (4 + 1 + num_experts as u64 + 2);
    gguf_data.extend_from_slice(&tensor_count.to_le_bytes());
    gguf_data.extend_from_slice(&6u64.to_le_bytes());
    
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
    let process_weight = |name: &str, data: &[f32], shape: Vec<u32>, zmr: &mut Vec<u8>, gguf_info: &mut Vec<(String, u32, Vec<u32>, Vec<u8>)>| {
        let (scale, packed) = quantize_bitnet(data);
        zmr.extend_from_slice(&scale.to_le_bytes());
        zmr.extend_from_slice(&(data.len() as u32).to_le_bytes());
        zmr.extend_from_slice(&packed);
        let mut tdata = Vec::new(); 
        tdata.extend_from_slice(&scale.to_le_bytes()); 
        tdata.extend_from_slice(&packed);
        gguf_info.push((name.to_string(), 7, shape, tdata));
    };
    
    let emb = model.embedding.embeddings().flatten_all()?.to_vec1::<f32>()?;
    process_weight("model.embed_tokens.weight", &emb, vec![vocab_size as u32, hidden_size as u32], &mut zmr_data, &mut gguf_tensor_infos);
    
    for i in 0..num_layers {
        let layer = &model.layers[i];
        for (pn, proj) in [("q_proj", &layer.q_proj), ("k_proj", &layer.k_proj), ("v_proj", &layer.v_proj), ("o_proj", &layer.o_proj)] {
            let w = proj.latent_weight.flatten_all()?.to_vec1::<f32>()?;
            process_weight(&format!("model.layers.{}.self_attn.{}.weight", i, pn), &w, vec![hidden_size as u32, hidden_size as u32], &mut zmr_data, &mut gguf_tensor_infos);
        }
        let gw = layer.moe.gate.latent_weight.flatten_all()?.to_vec1::<f32>()?;
        process_weight(&format!("model.layers.{}.mlp.gate.weight", i), &gw, vec![num_experts as u32, hidden_size as u32], &mut zmr_data, &mut gguf_tensor_infos);
        
        for e in 0..num_experts {
            if let Some(p) = layer.moe.packed_experts.get(e) {
                let ew = p.to_bitlinear((hidden_size, hidden_size), device)?;
                let w = ew.latent_weight.flatten_all()?.to_vec1::<f32>()?;
                process_weight(&format!("model.layers.{}.mlp.expert_{}.weight", i, e), &w, vec![hidden_size as u32, hidden_size as u32], &mut zmr_data, &mut gguf_tensor_infos);
            } else if e < layer.moe.experts.len() {
                let w = layer.moe.experts[e].latent_weight.flatten_all()?.to_vec1::<f32>()?;
                process_weight(&format!("model.layers.{}.mlp.expert_{}.weight", i, e), &w, vec![hidden_size as u32, hidden_size as u32], &mut zmr_data, &mut gguf_tensor_infos);
            }
        }
        
        for norm_name in ["input_layernorm", "post_attention_layernorm"] {
            gguf_tensor_infos.push((format!("model.layers.{}.{}.weight", i, norm_name), 1, vec![hidden_size as u32], vec![1u8; hidden_size * 2]));
        }
    }
    
    let head = model.lm_head.latent_weight.flatten_all()?.to_vec1::<f32>()?;
    process_weight("lm_head.weight", &head, vec![vocab_size as u32, hidden_size as u32], &mut zmr_data, &mut gguf_tensor_infos);
    gguf_tensor_infos.push(("model.norm.weight".to_string(), 1, vec![hidden_size as u32], vec![1u8; hidden_size * 2]));
    
    let mut offset = gguf_data.len() as u64 + tensor_count * 32;
    for (name, dtype, dims, data) in &gguf_tensor_infos {
        gguf_data.extend_from_slice(&(name.len() as u64).to_le_bytes()); 
        gguf_data.extend_from_slice(name.as_bytes());
        gguf_data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims { 
            gguf_data.extend_from_slice(&(d as u64).to_le_bytes()); 
        }
        gguf_data.extend_from_slice(&dtype.to_le_bytes()); 
        gguf_data.extend_from_slice(&offset.to_le_bytes());
        offset += data.len() as u64;
    }
    for (_, _, _, data) in &gguf_tensor_infos { 
        gguf_data.extend_from_slice(data); 
    }
    
    // ✅ استخدام output_dir بدلاً من المسار الثابت
    let zmr_path = std::path::Path::new(output_dir).join("zumar-b1.58.zmr");
    let gguf_path = std::path::Path::new(output_dir).join("zumar-b1.58.gguf");
    
    std::fs::write(&zmr_path, &zmr_data)?;
    std::fs::write(&gguf_path, &gguf_data)?;
    
    let zmr_mb = zmr_data.len() as f64 / 1_048_576.0;
    let gguf_mb = gguf_data.len() as f64 / 1_048_576.0;
    println!("\n╔══════════════════════════════════════╗");
    println!("║  📦 EXPORT COMPLETE                  ║");
    println!("║  Original:  {:>8.1} MB               ║", orig_mb);
    println!("║  .zmr:      {:>8.1} MB               ║", zmr_mb);
    println!("║  .gguf:     {:>8.1} MB               ║", gguf_mb);
    println!("║  Ratio:     {:>8.1}x smaller         ║", orig_mb / zmr_mb.max(0.1));
    println!("╚══════════════════════════════════════╝");
    println!("\n🚀 Chat:  cargo run -p core --release");
    println!("🚀 llama: ./llama-cli -m {} -p \"Hello\"", gguf_path.display());
    
    Ok(())
}


fn distill_runner(
    args: &Vec<String>,
    device: &candle_core::Device,
    vocab_size: usize, hidden_size: usize, num_layers: usize,
    num_experts: usize, top_k: usize, n_heads: usize,
) -> Result<()> {
    println!("\n{}", "═".repeat(60));
    println!("🧬 ZUMAR MULTI-TEACHER DISTILLATION");
    println!("   Shared Subword Projection + Online EWC");
    println!("{}", "═".repeat(60));

    // ── إعدادات ─────────────────────────────────────────────
    let total_epochs: usize = args.get(2)
        .and_then(|s| s.parse().ok()).unwrap_or(100);
    let ewc_lambda: f32     = args.get(3)
        .and_then(|s| s.parse().ok()).unwrap_or(400.0);

    // ── tokenizer الطالب ────────────────────────────────────
    println!("\n📖 Loading Zumar tokenizer...");
    let tokenizer = match ZumarTokenizer::load_or_train(
        "models/tokenizer/tokenizer.json",
        "data",
        vocab_size,
    ) {
        Ok(t) => {
            println!("   ✅ vocab={}", t.vocab_size());
            t
        }
        Err(e) => {
            println!("❌ Tokenizer error: {}", e);
            return Ok(());
        }
    };
    
    
    // ── جمع ملفات المعلمين ──────────────────────────────────
    let teacher_dir = std::path::Path::new("models/teacher");
    if !teacher_dir.exists() {
        println!("❌ models/teacher/ not found");
        return Ok(());
    }
    
    let mut all_teacher_paths: Vec<std::path::PathBuf> = Vec::new();
    collect_teacher_files(teacher_dir, &mut all_teacher_paths);
    
    if all_teacher_paths.is_empty() {
        println!("❌ No safetensors found in models/teacher/");
        return Ok(());
    }
    
    // ✅ تحليل المعلمين المطلوبين من سطر الأوامر
    let requested_teachers = parse_teacher_args(args);
    
    // ✅ تصفية المعلمين حسب الطلب (إنشاء نسخة جديدة)
    let mut filtered_paths: Vec<std::path::PathBuf> = if let Some(teachers_list) = requested_teachers {
        println!("\n🎯 Filtering teachers: {:?}", teachers_list);
        let mut filtered = Vec::new();
        for path in &all_teacher_paths {  // استخدم & لتجنب move
            let teacher_name = path.parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            if teachers_list.iter().any(|req| teacher_name.contains(req)) {
                filtered.push(path.clone());
            }
        }
        if filtered.is_empty() {
            println!("⚠️  No matching teachers found. Using all teachers.");
            all_teacher_paths.clone()  // استخدم clone() بدلاً من move
        } else {
            println!("   ✅ Selected {} teachers", filtered.len());
            filtered
        }
    } else {
        all_teacher_paths.clone()  // استخدم clone() بدلاً من move
    };
    
    // ✅ ترتيب حسب الحجم (الأصغر أولاً للـ Curriculum Learning)
    filtered_paths.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    
    println!("\n📂 Found {} teacher(s):", filtered_paths.len());
    for p in &filtered_paths {
        let mb = std::fs::metadata(p).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
        let name = p.parent().and_then(|p| p.file_name()).unwrap_or_default();
        println!("   📄 {} ({:.1} MB)", name.to_string_lossy(), mb);
    }
    
  
    


    // ── تحميل النموذج ديناميكياً ───────────────────────────
       // ── الحصول على تكوين النموذج ومسار الحفظ ──
    let (model_config, output_dir) = get_model_config_with_path(args);
    println!("📊 {}", model_config.description());
    println!("📁 Output directory: {}", output_dir);
    
        // إنشاء المجلد إذا لم يكن موجوداً
    std::fs::create_dir_all(&output_dir)?;
    
    // تحديث المسارات الثابتة
    let ewc_path = format!("{}/ewc_state.json", output_dir);
    let ckpt_path = format!("{}/distill_checkpoint.json", output_dir);
    let model_path = format!("{}/model.safetensors", output_dir);
    
    let mut varmap = candle_nn::VarMap::new();
    // let model_path = std::path::Path::new("models/zumar-v1/model.safetensors");

    if std::path::Path::new(&model_path).exists() {
    // if model_path.exists() {
        println!("\n♻️  Resuming from checkpoint...");
        varmap.load(model_path)?;
    }
    
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    // let mut model = ZumarModelDynamic::new(model_config, vs)?;
    let mut model = ZumarModel::new(
        model_config.vocab_size,
        model_config.hidden_size,
        model_config.num_layers,
        model_config.num_experts,
        model_config.top_k,
        model_config.num_heads,
        vs,
    )?;
    
    // ── بيانات التدريب ───────────────────────────────────
    let data_path = args.get(3).map(|s| s.as_str());  // لاحظ الرقم 3 بدلاً من 4───
    let training_data = data::TrainingData::load(data_path);
    let all_texts = training_data.texts.clone();   // بدون repeat
    // let training_data = data::TrainingData::load(args.get(4).map(|s| s.as_str()));
    // let all_texts     = training_data.repeat(5);
    println!("   📊 Training samples: {}", all_texts.len());

    // ── VocabAligner ────────────────────────────────────────
    println!("\n🔗 Building vocabulary alignments...");
    let aligner = match true_distill::prepare_alignments_from_dir(
        &filtered_paths,  // استخدم filtered_paths بدلاً من teacher_paths
        "models/tokenizer/tokenizer.json",
    ) {
        Ok(a) => a,
        Err(e) => {
            println!("❌ Alignment error: {}", e);
            return Ok(());
        }
    };

    // ── تحميل المعلمين ──────────────────────────────────────
    let mut teachers_and_alignments: Vec<(String, true_distill::AutoTeacher, vocab_aligner::VocabAlignment)>
        = Vec::new();
    
    for (path, alignment) in filtered_paths.iter().zip(aligner.into_iter()) {  // استخدم filtered_paths
    // for (path, alignment) in teacher_paths.iter().zip(aligner.into_iter()) {
        let teacher_label = path.parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        println!("   🧬 Attempting: {}", teacher_label);
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        if file_size > 500_000_000 {
            eprintln!("   ⚠️  SKIPPING {} ({} MB) - too large for device memory",
                teacher_label, file_size / 1_048_576);
            continue;
        }
        match true_distill::AutoTeacher::load(path.to_str().unwrap(), &device) {
            Ok(teacher) => {
                if teacher.config.arch_type == "skip" {
                    println!("   ⚠️  Skipping incompatible: {}",
                        path.file_name().unwrap().to_string_lossy());
                    continue;
                }
                teachers_and_alignments.push((teacher_label, teacher, alignment));
            }
            Err(e) => println!("   ❌ Load failed: {}", e),
        }
    }

    if teachers_and_alignments.is_empty() {
        println!("❌ No usable teachers found");
        return Ok(());
    }

    // ── تشغيل التقطير ───────────────────────────────────────
    let config = true_distill::DistillConfig {
        epochs:      total_epochs,
        base_lr:     0.01,  //1e-3,
        temperature: 1.0,  //3.0,
        ewc_lambda,
        accum_steps: 128, //4,
        save_every:  5,  //10,
        lora_rank: 4,
        lora_alpha: 16.0,
    };
    let distiller = true_distill::TrueDistiller::new(config, device.clone());

    distiller.distill_multi(
        &mut model,
        &varmap,
        &teachers_and_alignments,
        &all_texts,
        &tokenizer,
        &output_dir,  // أضف هذا
    )?;


     // ── تصدير تلقائي ────────────────────────────────────────
    println!("\n📦 Exporting .zmr + .gguf...");
    export_formats(
        &varmap, &device,
        model_config.vocab_size,
        model_config.hidden_size,
        model_config.num_layers,
        model_config.num_experts,
        model_config.top_k,
        model_config.num_heads,
        &output_dir,  // أضف هذا المعامل الجديد
    )?;

    println!("\n🎉 DISTILLATION COMPLETE!");
    println!("   Run: cargo run -p core --release");
    Ok(())
}

/// البحث العودي عن ملفات `.safetensors` داخل مجلد المعلم
fn collect_teacher_files(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_teacher_files(&path, files); // تابع البحث في المجلدات الفرعية
            } else if path.extension().map_or(false, |e| e == "safetensors") {
                files.push(path);
            }
        }
    }
}


fn signal_extractor(
    args: &Vec<String>,
    device: &candle_core::Device,
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_experts: usize,
    top_k: usize,
    n_heads: usize,
) -> Result<()> {
    use crate::tokenizer::ZumarTokenizer;
    use crate::true_distill::AutoTeacher;
    use half::f16;
    use std::io::Write;

    println!("══════════════════════════════════════════════════");
    println!("📡 ZUMAR SIGNAL EXTRACTOR");
    println!("══════════════════════════════════════════════════");

    // ١. جمع ملفات المعلمين
    let teacher_dir = std::path::Path::new("models/teacher");
    if !teacher_dir.exists() {
        println!("❌ models/teacher/ not found");
        return Ok(());
    }

    let mut all_teacher_paths = Vec::new();
    collect_teacher_files(teacher_dir, &mut all_teacher_paths);
    all_teacher_paths.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));

    // ✅ تحليل المعلم المطلوب من سطر الأوامر (للاستخراج التسلسلي)
    let requested_teacher = parse_single_teacher_arg(args);
    
    // ✅ تصفية المعلمين حسب الطلب
    let teacher_paths: Vec<std::path::PathBuf> = if let Some(teacher_name) = requested_teacher {
        println!("\n🎯 Extracting only teacher: {}", teacher_name);
        let mut filtered = Vec::new();
        for path in &all_teacher_paths {
            let name = path.parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            if name == teacher_name || name.contains(&teacher_name) {
                filtered.push(path.clone());
                break;
            }
        }
        if filtered.is_empty() {
            println!("   ⚠️  Teacher '{}' not found. Using all teachers.", teacher_name);
            all_teacher_paths
        } else {
            filtered
        }
    } else {
        all_teacher_paths
    };

    // ٢. تحميل tokenizer الطالب
    let tokenizer = match ZumarTokenizer::load_or_train(
        "models/tokenizer/tokenizer.json",
        "data",
        50257,
    ) {
        Ok(t) => {
            println!("   ✅ Tokenizer loaded: {} tokens", t.vocab_size());
            t
        }
        Err(e) => {
            println!("❌ Tokenizer error: {}", e);
            return Ok(());
        }
    };

    // ٣. تحميل نصوص التدريب
    let training_data = data::TrainingData::load(args.get(2).map(|s| s.as_str()));
    let all_texts = training_data.texts.clone();
    println!("   📊 Training texts: {}", all_texts.len());

    // ٤. إنشاء مجلد zlog
    std::fs::create_dir_all("models/zlog")?;

    // ٥. استخراج Logits لكل معلم (أو للمعلم المطلوب فقط)
    for path in &teacher_paths {
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        if file_size > 700_000_000 {
            println!("   ⚠️  Skipping {} ({} MB)", 
                path.file_name().unwrap().to_string_lossy(),
                file_size / 1_048_576);
            continue;
        }

        let teacher_name = path.parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // ✅ التحقق مما إذا كان الملف موجوداً بالفعل (لتجنب إعادة الاستخراج)
        let output_path = format!("models/zlog/{}.zlog", teacher_name);
        if std::path::Path::new(&output_path).exists() && !force_extract(args) {
            println!("\n⏭️  Skipping '{}' (zlog already exists. Use --force to re-extract)", teacher_name);
            continue;
        }

        println!("\n🧠 Extracting from '{}'...", teacher_name);

        let teacher = match AutoTeacher::load(path.to_str().unwrap(), device) {
            Ok(t) => t,
            Err(e) => {
                println!("   ❌ Cannot load: {}", e);
                continue;
            }
        };

        let real_vocab_size = teacher.config.vocab_size;
        println!("   📚 Teacher vocab size: {}", real_vocab_size);

        let mut file = std::fs::File::create(&output_path)?;
        
        // كتابة header
        file.write_all(b"ZLOG")?;
        file.write_all(&(all_texts.len() as u32).to_le_bytes())?;
        file.write_all(&(real_vocab_size as u32).to_le_bytes())?;

        let mut entry_count = 0u32;
        let total = all_texts.len();

        for (i, text) in all_texts.iter().enumerate() {
            let tokens = teacher.tokenize(text);
            if tokens.is_empty() { continue; }

            let logits_raw = match teacher.predict_with_embeddings(&tokens) {
                Ok(l) => l,
                Err(_) => continue,
            };

            let logits = if logits_raw.len() > real_vocab_size {
                logits_raw[..real_vocab_size].to_vec()
            } else {
                logits_raw
            };

            file.write_all(&(logits.len() as u32).to_le_bytes())?;
            for &val in &logits {
                let f16_val = f16::from_f32(val);
                file.write_all(&f16_val.to_bits().to_le_bytes())?;
            }

            entry_count += 1;
            if (i + 1) % 10 == 0 || i == total - 1 {
                println!("   📝 {}/{} texts processed", i + 1, total);
            }
        }

        println!("   ✅ Saved: {} ({} entries)", output_path, entry_count);
    }

    println!("\n🎉 Signal extraction complete!");
    println!("   📂 Files saved in models/zlog/");
    println!("   🚀 Now run: cargo run -- distill 10 /data");
    Ok(())
}


/// تحليل وسائط سطر الأوامر لاستخراج قائمة المعلمين المطلوبين
fn parse_teacher_args(args: &[String]) -> Option<Vec<String>> {
    // البحث عن --teachers متبوعاً بقائمة المعلمين
    for i in 0..args.len() {
        if args[i] == "--teachers" && i + 1 < args.len() {
            let teachers_str = &args[i + 1];
            let teachers: Vec<String> = teachers_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !teachers.is_empty() {
                return Some(teachers);
            }
        }
    }
    None
}

fn match_teacher_name(teacher_folder: &str, pattern: &str) -> bool {
    let pattern_lower = pattern.to_lowercase();
    let folder_lower = teacher_folder.to_lowercase();
    
    // مطابقة تامة أو جزئية
    folder_lower == pattern_lower || 
    folder_lower.contains(&pattern_lower) ||
    pattern_lower.contains(&folder_lower)
}

/// تحليل وسائط سطر الأوامر لاستخراج معلم واحد محدد
fn parse_single_teacher_arg(args: &[String]) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == "--teacher" && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
        if args[i] == "-t" && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
    }
    None
}

/// التحقق من وجود خيار --force
fn force_extract(args: &[String]) -> bool {
    args.iter().any(|arg| arg == "--force" || arg == "-f")
}

/// تحليل حجم النموذج من سطر الأوامر
fn parse_model_size(args: &[String]) -> Option<f32> {
    for i in 0..args.len() {
        if args[i] == "--size" && i + 1 < args.len() {
            let size_str = &args[i + 1];
            let size = size_str
                .trim_end_matches(|c| c == 'B' || c == 'b')
                .parse::<f32>()
                .ok()?;
            return Some(size);
        }
    }
    None
}

/// تحليل أبعاد مخصصة من سطر الأوامر
fn parse_custom_dimensions(args: &[String]) -> Option<(usize, usize, usize)> {
    for i in 0..args.len() {
        if args[i] == "--dims" && i + 3 < args.len() {
            let hidden = args[i + 1].parse::<usize>().ok()?;
            let layers = args[i + 2].parse::<usize>().ok()?;
            let experts = args[i + 3].parse::<usize>().ok()?;
            return Some((hidden, layers, experts));
        }
    }
    None
}

/// الحصول على تكوين النموذج
fn get_model_config(args: &[String]) -> ModelConfig {
    // محاولة الأبعاد المخصصة أولاً
    if let Some((hidden, layers, experts)) = parse_custom_dimensions(args) {
        println!("🎯 Using custom dimensions: {}d, {}L, {} experts", hidden, layers, experts);
        return ModelConfig::from_dimensions(hidden, layers, experts);
    }
    
    // ثم الحجم بالبايت
    if let Some(size_b) = parse_model_size(args) {
        println!("🎯 Configuring model for {:.1}B parameters", size_b);
        return ModelConfig::from_params_billions(size_b);
    }
    
    // الافتراضي
    ModelConfig::default()
}

/// تحليل مجلد الإخراج من سطر الأوامر
fn parse_output_dir(args: &[String]) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
        if args[i] == "-o" && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
    }
    None
}

/// الحصول على مسار النموذج حسب الحجم المطلوب
fn get_model_path_for_size(size: f32) -> String {
    let size_name = format_model_size_name(size);
    format!("{}/{}", MODELS_BASE_DIR, size_name)
}

/// تحويل الحجم الرقمي إلى اسم مجلد (مثلاً 1.5 -> "1.5B")
fn format_model_size_name(size: f32) -> String {
    if size >= 1.0 {
        format!("{}B", size)
    } else {
        format!("{}M", (size * 1000.0) as usize)
    }
}

/// تحويل اسم المجلد إلى حجم رقمي (مثلاً "1.5B" -> 1.5)
fn parse_size_from_folder_name(name: &str) -> Option<f32> {
    if name.ends_with('B') {
        name.trim_end_matches('B').parse::<f32>().ok()
    } else if name.ends_with('M') {
        let mb = name.trim_end_matches('M').parse::<f32>().ok()?;
        Some(mb / 1000.0)
    } else {
        None
    }
}

/// عرض النماذج المتاحة في مجلد models/zumar/
fn list_available_models() -> Vec<(String, f32, PathBuf)> {
    let mut models = Vec::new();
    let base_path = PathBuf::from(MODELS_BASE_DIR);
    
    if !base_path.exists() {
        return models;
    }
    
    if let Ok(entries) = fs::read_dir(&base_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let folder_name = path.file_name().unwrap().to_string_lossy().to_string();
                if let Some(size) = parse_size_from_folder_name(&folder_name) {
                    let model_file = path.join("model.safetensors");
                    if model_file.exists() {
                        let size_mb = fs::metadata(&model_file)
                            .map(|m| m.len() as f64 / 1_048_576.0)
                            .unwrap_or(0.0);
                        models.push((folder_name, size, path));
                    }
                }
            }
        }
    }
    
    // ترتيب حسب الحجم
    models.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    models
}

/// عرض قائمة النماذج واختيار واحد
fn select_model_interactive() -> Option<String> {
    let models = list_available_models();
    
    if models.is_empty() {
        println!("\n\x1b[1;33m⚠️  No trained models found in {}/\x1b[0m", MODELS_BASE_DIR);
        println!("   Please train a model first:");
        println!("   cargo run -- distill 100 /data --size 1.5B\n");
        return None;
    }
    
    println!("\n\x1b[1;36m📦 Available Models:\x1b[0m");
    println!("   {}", "═".repeat(50));
    
    for (i, (name, size, path)) in models.iter().enumerate() {
        let model_file = path.join("model.safetensors");
        let size_mb = fs::metadata(&model_file)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0);
        let ewc_file = path.join("ewc_state.json");
        let has_checkpoint = ewc_file.exists();
        
        println!("   \x1b[1;32m[{}]\x1b[0m {} ({:.1}B) - {:.0}MB {}",
            i + 1,
            name,
            size,
            size_mb,
            if has_checkpoint { "✓" } else { "" }
        );
    }
    println!("   {}", "═".repeat(50));
    print!("\n   \x1b[1;33mSelect model (1-{}): \x1b[0m", models.len());
    std::io::stdout().flush().unwrap();
    
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok()?;
    let idx = input.trim().parse::<usize>().ok()?;
    
    if idx >= 1 && idx <= models.len() {
        Some(models[idx - 1].0.clone())
    } else {
        None
    }
}

/// تحليل حجم النموذج من سطر الأوامر (مع دعم القائمة التفاعلية)
fn parse_model_size_interactive(args: &[String]) -> Option<f32> {
    // البحث عن --size في سطر الأوامر
    for i in 0..args.len() {
        if args[i] == "--size" && i + 1 < args.len() {
            let size_str = &args[i + 1];
            if size_str == "list" || size_str == "?" {
                // عرض النماذج المتاحة
                let models = list_available_models();
                if models.is_empty() {
                    println!("\n\x1b[1;33m⚠️  No trained models found. Available sizes: 0.08B, 0.4B, 1.5B, 7B, 13B, 70B\x1b[0m");
                    return None;
                }
                println!("\n\x1b[1;36m📦 Available trained models:\x1b[0m");
                for (i, (name, size, _)) in models.iter().enumerate() {
                    println!("   {}. {} ({:.1}B)", i + 1, name, size);
                }
                return None;
            }
            let size = size_str
                .trim_end_matches(|c| c == 'B' || c == 'b')
                .parse::<f32>()
                .ok()?;
            return Some(size);
        }
    }
    None
}

/// الحصول على تكوين النموذج مع تحديد مسار الحفظ
fn get_model_config_with_path(args: &[String]) -> (ModelConfig, String) {
    let mut output_dir = MODELS_BASE_DIR.to_string();
    let mut size: Option<f32> = None;
    
    // محاولة الأبعاد المخصصة أولاً
    if let Some((hidden, layers, experts)) = parse_custom_dimensions(args) {
        let size_b = (hidden * layers * experts) as f32 / 100_000_000.0; // تقدير تقريبي
        output_dir = format!("{}/custom_{}d_{}L", MODELS_BASE_DIR, hidden, layers);
        let config = ModelConfig::from_dimensions(hidden, layers, experts);
        return (config, output_dir);
    }
    
    // ثم الحجم
    if let Some(size_b) = parse_model_size_interactive(args) {
        size = Some(size_b);
        let size_name = format_model_size_name(size_b);
        output_dir = format!("{}/{}", MODELS_BASE_DIR, size_name);
        let config = ModelConfig::from_params_billions(size_b);
        return (config, output_dir);
    }
    
    // الافتراضي - عرض النماذج المتاحة إن وجدت
    let models = list_available_models();
    if !models.is_empty() && args.iter().all(|a| a != "--size") {
        println!("\n\x1b[1;33m⚠️  No --size specified. Using existing model...\x1b[0m");
        if let Some(selected) = select_model_interactive() {
            if let Some(size_val) = parse_size_from_folder_name(&selected) {
                output_dir = format!("{}/{}", MODELS_BASE_DIR, selected);
                let config = ModelConfig::from_params_billions(size_val);
                return (config, output_dir);
            }
        }
    }
    
    (ModelConfig::default(), format!("{}/80M", MODELS_BASE_DIR))
}

/// استخراج أبعاد النموذج من مسار المجلد
fn parse_model_dimensions_from_path(path: &str) -> Result<(usize, usize, usize, usize, usize, usize)> {
    // حاول قراءة أبعاد النموذج من اسم المجلد
    let folder_name = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    
    // الأبعاد الافتراضية
    let (mut hidden_size, mut num_layers, mut num_experts) = (512, 12, 6);
    let vocab_size = 50257;
    let top_k = 2;
    let n_heads = 16;
    
    // استخراج الحجم من اسم المجلد
    if folder_name.ends_with('B') || folder_name.ends_with('M') {
        if let Some(size_val) = parse_size_from_folder_name(folder_name) {
            let config = ModelConfig::from_params_billions(size_val);
            hidden_size = config.hidden_size;
            num_layers = config.num_layers;
            num_experts = config.num_experts;
            println!("📊 Detected model dimensions: {}d, {}L, {} experts", hidden_size, num_layers, num_experts);
        }
    } else if folder_name.starts_with("custom_") {
        // تنسيق مخصص: custom_2048d_32L
        let parts: Vec<&str> = folder_name.split('_').collect();
        if parts.len() >= 3 {
            if let Ok(h) = parts[1].trim_end_matches('d').parse::<usize>() {
                hidden_size = h;
            }
            if let Ok(l) = parts[2].trim_end_matches('L').parse::<usize>() {
                num_layers = l;
            }
        }
        println!("📊 Custom dimensions: {}d, {}L", hidden_size, num_layers);
    }
    
    Ok((vocab_size, hidden_size, num_layers, num_experts, top_k, n_heads))
}