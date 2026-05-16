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
    println!("  distill <epochs>     - True distillation (resumes from last save)");
    println!("  train <epochs>       - Self-training on built-in data");
    println!("  chat                 - Chat mode (default)");
    println!("  pack                 - Export to .zmr + .gguf");
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
            export_formats(&varmap, &device, vocab_size, hidden_size, num_layers, num_experts, top_k, n_heads)?;
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
        
        "help" | "--help" | "-h" => { print_usage(); }
        
        _ => {
            println!("\n💬 Chat Mode\n");
            
            // ✅ تحقق من وجود الأوزان أولاً
            let zmr_path = std::path::Path::new("models/zumar-v1/zumar-b1.58.zmr");
            let safetensors_path = std::path::Path::new("models/zumar-v1/model.safetensors");
            
            // ✅ إذا لم توجد أي أوزان، شغّل التقطير المدمج
            if !zmr_path.exists() && !safetensors_path.exists() {
                println!("\x1b[1;33m⚠️  No Zumar weights found.\x1b[0m");
                println!("\x1b[1;36m🔍 Searching for teacher model...\x1b[0m");
                
                distill_runner(
                  &args, &device, 
                  vocab_size, hidden_size, 
                  num_layers, num_experts, 
                  top_k, n_heads
                )?;
            }
            
            
            // ✅ تحميل الأوزان (الموجودة أو المقطرة حديثاً)
            let mut loader = loader::ZumarLoader::new("models/zumar-v1");
            let _ = loader.load_weights(&device)?;
            
            let (v, h, l, e) = if let Some(cfg) = loader.get_zmr_config() {
                (cfg.vocab_size, cfg.hidden_size, cfg.num_layers, cfg.num_experts)
            } else {
                (vocab_size, hidden_size, num_layers, num_experts)
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
            } else {
                let safetensors_path = std::path::Path::new("models/zumar-v1/model.safetensors");
                if safetensors_path.exists() {
                    println!("   📦 Using .safetensors (FP32)");
                    let vb = unsafe { 
                        candle_nn::VarBuilder::from_mmaped_safetensors(
                            &[safetensors_path], candle_core::DType::F32, &device
                        )? 
                    };
                    ZumarModel::new(v, h, l, e, top_k, n_heads, vb)?
                } else {
                    println!("\x1b[1;31m❌ No model found\x1b[0m");
                    return Ok(());
                }
            };
            
            println!("✅ Ready.\n");
            
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
                            // تحويل من FP16 إلى FP32 إذا لزم الأمر
                            let e = e.to_dtype(candle_core::DType::F32).unwrap_or(e);
                            e
                        },
                        Err(_) => break
                    };
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
                println!("\x1b[90m📊 {} tokens in {:.1}s ({:.1} tok/s)\x1b[0m", n, elapsed.as_secs_f64(), tps);
            }
            println!("\n\x1b[1;35m🛡️  ZUMAR SHUTTING DOWN\x1b[0m");
      }
    }
    Ok(())
}

fn export_formats(
    varmap: &candle_nn::VarMap,  // ← استخدم VarMap مباشرة
    device: &candle_core::Device,
    vocab_size: usize, hidden_size: usize, num_layers: usize,
    num_experts: usize, top_k: usize, n_heads: usize,
) -> Result<()> {
    
    let save_path = std::path::Path::new("models/zumar-v1").join("model.safetensors");
    if !save_path.exists() {
        println!("\x1b[1;31m❌ No model found. Train first.\x1b[0m");
        return Ok(());
    }
    
    // let vs = candle_nn::VarBuilder::from_varmap(varmap, candle_core::DType::F32, device);
    // let model = ZumarModel::new(vocab_size, hidden_size, num_layers, num_experts, 2, n_heads, vs)?;
     // بناء النموذج مع QLoRA مباشرة
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let mut model = ZumarModel::new_qlora(
        vocab_size, hidden_size, num_layers,
        num_experts, top_k, n_heads, vs,
        8,   // rank
        16.0 // alpha
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
        d.extend_from_slice(&(k.len() as u64).to_le_bytes()); d.extend_from_slice(k.as_bytes());
        d.extend_from_slice(&8u32.to_le_bytes()); d.extend_from_slice(&(v.len() as u64).to_le_bytes()); d.extend_from_slice(v.as_bytes());
    };
    let wmu = |d: &mut Vec<u8>, k: &str, v: u32| {
        d.extend_from_slice(&(k.len() as u64).to_le_bytes()); d.extend_from_slice(k.as_bytes());
        d.extend_from_slice(&4u32.to_le_bytes()); d.extend_from_slice(&v.to_le_bytes());
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
        let mut tdata = Vec::new(); tdata.extend_from_slice(&scale.to_le_bytes()); tdata.extend_from_slice(&packed);
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
        gguf_data.extend_from_slice(&(name.len() as u64).to_le_bytes()); gguf_data.extend_from_slice(name.as_bytes());
        gguf_data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims { gguf_data.extend_from_slice(&(d as u64).to_le_bytes()); }
        gguf_data.extend_from_slice(&dtype.to_le_bytes()); gguf_data.extend_from_slice(&offset.to_le_bytes());
        offset += data.len() as u64;
    }
    for (_, _, _, data) in &gguf_tensor_infos { gguf_data.extend_from_slice(data); }
    
    let zmr_path = std::path::Path::new("models/zumar-v1").join("zumar-b1.58.zmr");
    let gguf_path = std::path::Path::new("models/zumar-v1").join("zumar-b1.58.gguf");
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

    let mut teacher_paths: Vec<std::path::PathBuf> = Vec::new();
    collect_teacher_files(teacher_dir, &mut teacher_paths);

    if teacher_paths.is_empty() {
        println!("❌ No safetensors found in models/teacher/");
        return Ok(());
    }

    // ✅ ترتيب curriculum: الأصغر أولاً (الأبسط)
    teacher_paths.sort_by_key(|p| std::fs::metadata(p)
        .map(|m| m.len()).unwrap_or(0));

    println!("\n📂 Found {} teacher(s):", teacher_paths.len());
    for p in &teacher_paths {
        let mb = std::fs::metadata(p).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
        println!("   📄 {} ({:.1} MB)", p.file_name().unwrap().to_string_lossy(), mb);
    }

    // ── تحميل النموذج (مرة واحدة) ───────────────────────────
    let mut varmap = candle_nn::VarMap::new();
    let model_path = std::path::Path::new("models/zumar-v1/model.safetensors");
    if model_path.exists() {
        println!("\n♻️  Resuming from checkpoint...");
        varmap.load(model_path)?;
    }
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    // let mut model = ZumarModel::new(
    //     vocab_size, hidden_size, num_layers,
    //     num_experts, top_k, n_heads, vs,
    // )?;
    // // تفعيل QLoRA
    println!("   🧬 Activating QLoRA (NF4 + LoRA rank=8)...");
    // model.add_qlora(8, 16.0)?;
    // بناء النموذج مع QLoRA مباشرة
    let mut model = ZumarModel::new_qlora(
        vocab_size, hidden_size, num_layers,
        num_experts, top_k, n_heads, vs,
        8,   // rank
        16.0 // alpha
    )?;
    
    // ── بيانات التدريب ──────────────────────────────────────
    let training_data = data::TrainingData::load(args.get(4).map(|s| s.as_str()));
    let all_texts     = training_data.repeat(5);
    println!("   📊 Training samples: {}", all_texts.len());

    // ── VocabAligner ────────────────────────────────────────
    println!("\n🔗 Building vocabulary alignments...");
    let aligner = match true_distill::prepare_alignments_from_dir(
        &teacher_paths,
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
    
    for (path, alignment) in teacher_paths.iter().zip(aligner.into_iter()) {
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
        accum_steps: 64, //4,
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
    )?;

    // ── تصدير تلقائي ────────────────────────────────────────
    println!("\n📦 Exporting .zmr + .gguf...");
    export_formats(
        &varmap, &device,
        vocab_size, hidden_size, num_layers,
        num_experts, top_k, n_heads,
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
    vocab_size: usize,      // حجم الطالب (غير مستخدم هنا)
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

    let mut teacher_paths = Vec::new();
    collect_teacher_files(teacher_dir, &mut teacher_paths);
    teacher_paths.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));

    // ٢. تحميل tokenizer الطالب (لحساب الهاش فقط)
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

    // ٥. استخراج Logits لكل معلم
    for path in &teacher_paths {
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        if file_size > 700_000_000 {
            println!("   ⚠️  Skipping {} ({} MB) - too large",
                path.file_name().unwrap().to_string_lossy(),
                file_size / 1_048_576);
            continue;
        }

        let teacher_name = path.parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        println!("\n🧠 Extracting from '{}'...", teacher_name);

        let teacher = match AutoTeacher::load(path.to_str().unwrap(), device) {
            Ok(t) => t,
            Err(e) => {
                println!("   ❌ Cannot load: {}", e);
                continue;
            }
        };

        let real_vocab_size = teacher.config.vocab_size; // الحجم الحقيقي للمفردات
        println!("   📚 Teacher vocab size: {}", real_vocab_size);

        let output_path = format!("models/zlog/{}.zlog", teacher_name);
        let mut file = std::fs::File::create(&output_path)?;

        let mut entry_count = 0u32;
        let total = all_texts.len();

        for (i, text) in all_texts.iter().enumerate() {
            let tokens = teacher.tokenize(text);
            
            // if tokens.len() < 2 { continue; }
            if tokens.is_empty() { continue; }

            let logits_raw = match teacher.predict_with_embeddings(&tokens) {
                Ok(l) => l,
                Err(_) => continue,
            };

            // اقتصاص إلى الحجم الحقيقي للمفردات
            let logits = if logits_raw.len() > real_vocab_size {
                logits_raw[..real_vocab_size].to_vec()
            } else {
                logits_raw
            };

            // حساب hash (نفس طريقة التقطير)
            let clean_text = text.trim();
            let key_hash = {
                let mut hash: u64 = 0xcbf29ce484222325;
                for &b in clean_text.as_bytes() {
                    hash ^= b as u64;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
                hash
            };

            // كتابة: hash (8) + len (4) + logits f16
            file.write_all(&key_hash.to_le_bytes())?;
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

        println!("   ✅ Saved: {} ({} entries, vocab={})", output_path, entry_count, real_vocab_size);
    }

    println!("\n🎉 Signal extraction complete!");
    println!("   📂 Files saved in models/zlog/");
    println!("   🚀 Now run: cargo run -- distill 10 /data");

    Ok(())
}