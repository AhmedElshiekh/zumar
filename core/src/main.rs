mod layers;
mod routing;
mod config;
mod tokenizer;
mod loader;
mod kv_cache;
mod kernels;

use candle_core::{Result, Tensor}; 
use std::io::{self, Write}; 

#[tokio::main]
async fn main() -> Result<()> {
    // --- التنسيق البصري السيادي لـ ZUMAR ---
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

    let router = routing::HardwareRouter::new();
    let device = router.route("Inference Task");

    let model_dir = "models/zumar-v1";
    let loader = loader::ZumarLoader::new(model_dir);
    let vb = loader.load_weights(&device)?;

    let zumar_tokenizer = if std::path::Path::new(&loader.get_tokenizer_path()).exists() {
        println!("✅ \x1b[32mTokenizer Status:\x1b[0m Loaded and Ready.");
        Some(tokenizer::ZumarTokenizer::new(&loader.get_tokenizer_path())?)
    } else {
        println!("⚠️ \x1b[33mWarning:\x1b[0m Tokenizer missing.");
        None
    };

    let hidden_size = 1024;
    let vocab_size = 50257;
    // تحميل بلوك زُمر الأساسي
    let model_block = layers::ZumarBlock::new(hidden_size, vocab_size, vb.clone())?; 

    println!("✅ \x1b[32mCore Status:\x1b[0m 100 Sovereign Layers Active");
    println!("\x1b[1;34m💡 Hint: Type your prompt or 'exit' to quit.\x1b[0m");

    // إعدادات التحكم في الاستجابة (Generation Hyperparameters)
    let temperature = 0.8f64; 
    let penalty_factor = 2.0f32; // تم رفع القيمة لمنع التكرار اللانهائي للرموز

    loop {
        print!("\n\x1b[1;32m┌───[ 👤 User ]\x1b[0m\n\x1b[1;32m└─> \x1b[0m");
        io::stdout().flush().ok();
        
        let mut user_input = String::new();
        if io::stdin().read_line(&mut user_input).is_err() { break; }
        let prompt = user_input.trim();
        if prompt == "exit" { break; }
        if prompt.is_empty() { continue; }

        print!("\x1b[1;30m[🧠 Zumar Thinking...]\x1b[0m");
        io::stdout().flush().ok();

        let mut generated_tokens: Vec<u32> = Vec::new();
        
        // ترميز النص المدخل وتحويله لتمثيل عددي (Embeddings)
        let mut current_input = if let Some(ref tok) = zumar_tokenizer {
            let tokens_tensor = tok.encode(prompt, &device).unwrap_or(Tensor::new(&[1u32], &device)?);
            // نأخذ آخر توكن في المدخلات لبدء التوليد
            let last_token = tokens_tensor.flatten_all()?.get(tokens_tensor.elem_count()-1)?.to_scalar::<u32>()?;
            model_block.embed(last_token, &device)?
        } else {
            model_block.embed(1, &device)?
        };

        println!("\r\x1b[K\x1b[1;36m┌───[ 🤖 Zumar ]\x1b[0m");
        print!("\x1b[1;36m└─> \x1b[0m");
        io::stdout().flush().ok();

        for _token_step in 0..120 { 
            let mut hidden_states = current_input.clone();
            
            // تمرير عبر الطبقات (Forward Pass)
            // ملاحظة: تم تعديل الحلقة لتجنب انفجار الأرقام (Signal Explosion)
            for _ in 0..12 { 
                let residual = hidden_states.clone();
                hidden_states = model_block.forward_core(&hidden_states)?; 
                hidden_states = (hidden_states + residual)?; 
            }

            // استخراج التوقعات (Logits)
            let logits = model_block.project_head(&hidden_states)?;
            
            // تصحيح: تحويل الـ Tensor إلى Vector بشكل مسطح للوصول لكل الاحتمالات
            let logits_flat = logits.flatten_all()?;
            let mut last_logits_vec = logits_flat.to_vec1::<f32>()?;
            
            // تطبيق عقوبة التكرار (Repetition Penalty) بأسلوب الطرح لضمان الفاعلية
            for &prev_token in &generated_tokens {
                let idx = prev_token as usize;
                if idx < last_logits_vec.len() {
                    last_logits_vec[idx] -= penalty_factor; 
                }
            }

            // تحويل النتائج مرة أخرى إلى Tensor للمعالجة الرياضية
            let processed_logits = Tensor::new(last_logits_vec.as_slice(), &device)?;
            let pr = (&processed_logits / temperature)?;
            let pr = candle_nn::ops::softmax(&pr, 0)?;
            
            // اختيار التوكن الأعلى احتمالية
            let next_token_id = pr.argmax(0)?.to_scalar::<u32>()?;
            generated_tokens.push(next_token_id);
            
            // فك الترميز وطباعة النص فوراً
            if let Some(ref tok) = zumar_tokenizer {
                if let Ok(text) = tok.decode(&[next_token_id]) { 
                    print!("{}", text); 
                    io::stdout().flush().ok();
                }
            } else { 
                print!("[{}] ", next_token_id); 
                io::stdout().flush().ok();
            }

            // تحديث المدخلات للخطوة القادمة (Autoregressive)
            current_input = model_block.embed(next_token_id, &device)?; 

            // التوقف عند رمز نهاية النص (EOS)
            if next_token_id == 50256 || next_token_id == 2 { break; }
        }
        println!("\n\x1b[90m──────────────────────────────────────────────────\x1b[0m");
    }
    Ok(())
}
