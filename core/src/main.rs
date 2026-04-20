mod layers;
mod routing;
mod config;
mod tokenizer;
mod loader;
mod kv_cache;
mod kernels;

use candle_core::{Result, IndexOp, Device}; // حذفنا التوكنز غير المستخدمة لتجنب الـ Warnings
use std::io::{self, Write}; 
use crate::kv_cache::KVCache;

#[tokio::main]
async fn main() -> Result<()> {
    // تنسيق بصري احترافي للبداية
    print!("{}[2J", 27 as char); 
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
    println!("\x1b[1;36m--- 🌌 ZUMAR SOVEREIGN CORE ENGINE v2.2.0 ---\x1b[0m");
    println!("\x1b[90m--------------------------------------------------\x1b[0m");

    // 1. التوجيه الذكي للعتاد
    let router = routing::HardwareRouter::new();
    let device = router.route("Inference Task");

    // 2. تحميل الأوزان والقاموس
    let model_dir = "models/zumar-v1";
    let loader = loader::ZumarLoader::new(model_dir);
    let vb = loader.load_weights(&device)?;

    let zumar_tokenizer = if std::path::Path::new(&loader.get_tokenizer_path()).exists() {
        println!("✅ \x1b[32mTokenizer Status:\x1b[0m Loaded and Ready.");
        Some(tokenizer::ZumarTokenizer::new(&loader.get_tokenizer_path())?)
    } else {
        println!("⚠️ \x1b[33mWarning:\x1b[0m Tokenizer missing, raw mode active.");
        None
    };

    // 3. بناء الكتلة الهجين
    let hidden_size = 1024;
    let vocab_size = 50257;
    let model_block = layers::ZumarBlock::new(hidden_size, vocab_size, vb.clone())?; 

    println!("✅ \x1b[32mCore Status:\x1b[0m 100 Sovereign Layers Active");
    println!("\x1b[1;34m💡 Hint: Type your prompt or 'exit' to quit.\x1b[0m");
    println!("\x1b[90m--------------------------------------------------\x1b[0m");

    let mut _cache = KVCache::new();
    
    loop {
        print!("\n\x1b[1;32m┌───[ 👤 User ]\x1b[0m\n\x1b[1;32m└─> \x1b[0m");
        io::stdout().flush().ok();
        
        let mut user_input = String::new();
        if io::stdin().read_line(&mut user_input).is_err() { break; }
        let prompt = user_input.trim();

        if prompt == "exit" { 
            println!("\n\x1b[1;31mSession Terminated. Sovereignty Preserved.\x1b[0m");
            break; 
        }
        if prompt.is_empty() { continue; }

        print!("\x1b[1;36m┌───[ 🤖 Zumar ]\x1b[0m\n\x1b[1;36m└─> \x1b[0m");
        io::stdout().flush().ok();

        // حل أخطاء الـ Tokenizer: تمرير الـ Device واستقبال الـ Tensor
        let mut current_input = if let Some(ref tok) = zumar_tokenizer {
            // نمرر &device كمعامل ثانٍ كما طلب الـ Compiler
            let tokens_tensor = tok.encode(prompt, &device).unwrap_or_else(|_| {
                candle_core::Tensor::new(&[1u32], &device).unwrap()
            });
            
            // استخراج أول توكن من التينسور بشكل آمن
            let first_token = tokens_tensor.flatten_all()?.get(0)?.to_scalar::<u32>()?;
            model_block.embed(first_token, &device)?
        } else {
            model_block.embed(1, &device)?
        };

        // 4. دورة التوليد (منطق الـ 100 طبقة)
        for _token_step in 0..50 { 
            let mut hidden_states = current_input.clone();
            
            for _layer_idx in 0..100 {
                let residual = hidden_states.clone();
                hidden_states = model_block.forward_core(&hidden_states)?; 
                hidden_states = (hidden_states + residual)?;
            }

            let logits = model_block.project_head(&hidden_states)?;
            let last_step_logits = logits.i((0, logits.dim(1)? - 1))?;
            let next_token_id = last_step_logits.argmax(0)?.to_scalar::<u32>()?;
            
            if let Some(ref tok) = zumar_tokenizer {
                if let Ok(text) = tok.decode(&[next_token_id]) {
                    print!("{}", text); 
                }
            } else {
                print!("[{}] ", next_token_id);
            }
            io::stdout().flush().ok();

            current_input = model_block.embed(next_token_id, &device)?; 

            if next_token_id == 50256 { break; }
        }
        println!("\n\x1b[90m──────────────────────────────────────────────────\x1b[0m");
    }
    
    Ok(())
}
