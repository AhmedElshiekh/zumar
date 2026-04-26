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
    println!("  train <epochs>       - Self-training on built-in data");
    println!("  distill <epochs>     - True distillation from ALL safetensors in teacher/");
    println!("  chat                 - Chat mode (default)");
    println!("\nExamples:");
    println!("  cargo run -p core --release -- distill 50");
    println!("  cargo run -p core --release -- train 10");
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
            
            // جمع كل ملفات safetensors
            let mut teacher_files: Vec<std::path::PathBuf> = Vec::new();
            if let Ok(entries) = std::fs::read_dir(teacher_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map_or(false, |e| e == "safetensors") {
                        teacher_files.push(path);
                    }
                }
            }
            
            // ترتيب: الأكبر أولاً
            teacher_files.sort_by(|a, b| {
                let sa = std::fs::metadata(a).map(|m| m.len()).unwrap_or(0);
                let sb = std::fs::metadata(b).map(|m| m.len()).unwrap_or(0);
                sb.cmp(&sa)
            });
            
            if teacher_files.is_empty() {
                println!("\x1b[1;31m❌ No safetensors found in models/teacher/\x1b[0m");
                println!("\x1b[1;33m   Place .safetensors models there\x1b[0m");
                return Ok(());
            }
            
            println!("\x1b[1;32m📂 Found {} teacher model(s):\x1b[0m", teacher_files.len());
            for f in &teacher_files {
                let size = std::fs::metadata(f)
                    .map(|m| m.len() as f64 / 1_048_576.0)
                    .unwrap_or(0.0);
                println!("   📄 {} ({:.1} MB)", f.file_name().unwrap().to_string_lossy(), size);
            }
            
            let epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
            
            // قطّر كل نموذج حتى ينجح واحد
            for (i, teacher_path) in teacher_files.iter().enumerate() {
                println!("\n{}", "=".repeat(60));
                println!("🧬 Model {}/{}: {}", 
                    i + 1, teacher_files.len(),
                    teacher_path.file_name().unwrap().to_string_lossy()
                );
                println!("{}", "=".repeat(60));
                
                let varmap = VarMap::new();
                let vs = candle_nn::VarBuilder::from_varmap(
                    &varmap, 
                    candle_core::DType::F32, 
                    &device
                );
                
                let mut model = ZumarModel::new(
                    vocab_size, hidden_size, num_layers,
                    num_experts, top_k, n_heads, vs.clone(),
                )?;
                
                let config = true_distill::DistillConfig {
                    epochs,
                    learning_rate: 0.001,
                    temperature: 3.0,
                };
                
                let distiller = true_distill::TrueDistiller::new(config, device.clone());
                let training_data = data::TrainingData::load(None);
                let all_texts = training_data.repeat(10);
                
                match distiller.distill(
                    &mut model,
                    &varmap,
                    teacher_path.to_str().unwrap(),
                    &all_texts,
                ) {
                    Ok(_) => {
                        let save_dir = std::path::Path::new("models/zumar-v1");
                        std::fs::create_dir_all(save_dir)?;
                        let save_path = save_dir.join("model.safetensors");
                        varmap.save(&save_path)?;
                        
                        let size_mb = std::fs::metadata(&save_path)
                            .map(|m| m.len() as f64 / 1_048_576.0)
                            .unwrap_or(0.0);
                        println!("\n\x1b[1;32m💾 Saved ({:.1} MB)\x1b[0m", size_mb);
                        println!("\x1b[1;32m✅ Model {} distillation complete!\x1b[0m", i + 1);
                        println!("\x1b[1;36m🚀 Run: cargo run -p core --release\x1b[0m");
                        break;  // نجحنا - توقف
                    }
                    Err(e) => {
                        println!("\x1b[1;31m❌ Failed: {}\x1b[0m", e);
                        if i < teacher_files.len() - 1 {
                            println!("\x1b[1;33m   ⏭  Trying next model...\x1b[0m");
                        }
                    }
                }
            }
        }
        
        // ==========================================
        // تدريب ذاتي
        // ==========================================
        "train" => {
            print_usage();
            println!("\n🎓 SELF-TRAINING MODE\n");
            
            let varmap = VarMap::new();
            let vs = candle_nn::VarBuilder::from_varmap(
                &varmap, 
                candle_core::DType::F32, 
                &device
            );
            
            let mut model = ZumarModel::new(
                vocab_size, hidden_size, num_layers,
                num_experts, top_k, n_heads, vs.clone(),
            )?;
            
            let epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            
            println!("   Config: {}d, {}L, {}h, {} experts", 
                hidden_size, num_layers, n_heads, num_experts);
            println!("   Epochs: {}\n", epochs);
            
            train::run_training(&mut model, &varmap, &device, None, epochs)?;
            
            let save_dir = std::path::Path::new("models/zumar-v1");
            std::fs::create_dir_all(save_dir)?;
            let save_path = save_dir.join("model.safetensors");
            varmap.save(&save_path)?;
            
            let size_mb = std::fs::metadata(&save_path)
                .map(|m| m.len() as f64 / 1_048_576.0)
                .unwrap_or(0.0);
            println!("\n💾 Saved ({:.1} MB)", size_mb);
            println!("🚀 Run: cargo run -p core --release");
        }
        
        // ==========================================
        // مساعدة
        // ==========================================
        "help" | "--help" | "-h" => {
            print_usage();
        }
        
        // ==========================================
        // محادثة (افتراضي)
        // ==========================================
        _ => {
            println!("\n💬 Chat Mode\n");
            
            let loader = loader::ZumarLoader::new("models/zumar-v1");
            let vb = loader.load_weights(&device)?;
            
            println!("🔧 Building model...");
            let mut model = ZumarModel::new(
                vocab_size, hidden_size, num_layers,
                num_experts, top_k, n_heads, vb.clone(),
            )?;
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
                
                let tokens: Vec<u32> = prompt.chars()
                    .map(|c| (c as u32 % 256) + 3)
                    .collect();
                
                let mut current = *tokens.last().unwrap_or(&1);
                let start = std::time::Instant::now();
                let mut generated = Vec::new();
                
                print!("\x1b[1;36mZumar>\x1b[0m ");
                io::stdout().flush().ok();
                
                for _ in 0..max_tokens {
                    let emb = match model.embed(current, &device) {
                        Ok(e) => e, Err(_) => break
                    };
                    let out = match model.forward(&emb) {
                        Ok(o) => o, Err(_) => break
                    };
                    let flat = match out.flatten_all() {
                        Ok(f) => f, Err(_) => break
                    };
                    let v = match flat.to_vec1::<f32>() {
                        Ok(vec) => vec, Err(_) => break
                    };
                    
                    // عقوبة التكرار
                    let mut logits = v.clone();
                    for &prev in &generated {
                        let idx = prev as usize;
                        if idx < logits.len() {
                            logits[idx] /= penalty;
                        }
                    }
                    
                    // Temperature
                    let scaled: Vec<f32> = logits.iter()
                        .map(|&x| x / temperature as f32)
                        .collect();
                    
                    // Softmax
                    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
                    let sum: f32 = exp.iter().sum();
                    let probs: Vec<f32> = exp.iter().map(|&v| v / (sum + 1e-9)).collect();
                    
                    // اختيار أعلى احتمال
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
                let tps = if elapsed.as_secs_f64() > 0.0 {
                    n as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                
                println!();
                println!("\x1b[90m──────────────────────────────────────\x1b[0m");
                println!("\x1b[90m📊 {} tokens in {:.1}s ({:.1} tok/s)\x1b[0m",
                    n, elapsed.as_secs_f64(), tps);
            }
            
            println!("\n\x1b[1;35m══════════════════════════════════════\x1b[0m");
            println!("\x1b[1;35m🛡️  ZUMAR SHUTTING DOWN\x1b[0m");
            println!("\x1b[1;35m══════════════════════════════════════\x1b[0m");
        }
    }
    Ok(())
}