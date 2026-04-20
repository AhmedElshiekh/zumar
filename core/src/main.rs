mod layers;
mod routing;
mod config;
mod tokenizer;
mod loader;
mod kv_cache;
mod kernels;

use candle_core::{Result, Tensor, IndexOp};
use tokio::sync::mpsc;
//use std::fs;
use std::io::Write;
use crate::kv_cache::KVCache; // استدعاء الكاش

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- 🌌 ZUMAR SOVEREIGN CORE ENGINE v2.2.0 ---");

    let router = routing::HardwareRouter::new();
    let device = router.route("Inference Task");

    let model_dir = "models/zumar-v1";
    let loader = loader::ZumarLoader::new(model_dir);
    let vb = loader.load_weights(&device)?;

    let zumar_tokenizer = if std::path::Path::new(&loader.get_tokenizer_path()).exists() {
        Some(tokenizer::ZumarTokenizer::new(&loader.get_tokenizer_path())?)
    } else { None };

    let (tx, mut rx) = mpsc::channel::<String>(32);
    let test_tx = tx.clone();
    tokio::spawn(async move { let _ = test_tx.send("Execute Generation Loop".to_string()).await; });
   
    if let Some(prompt) = rx.recv().await {
        println!("📩 Task Received: \"{}\"", prompt);
        
        let vocab_size = 32000;
        let model_block = layers::ZumarBlock::new(768, vocab_size, vb.clone())?; 
        
        // 1. إنشاء الكاش المخصص للمحرك
        let mut cache = KVCache::new();
        
        // المدخلات الأولية (Prompt Tokens)
        let mut current_input = Tensor::randn(0f32, 0.02f32, (1, 5, 768), &device)?;
        
        println!("⚙️ Engine Status: Running Recursive BitNet + KV-Cache Optimized...");
        println!("--------------------------------------------------");

        for _token_step in 0..12 {
            // التحقق: إذا كان هناك كاش، نأخذ آخر توكن فقط للمعالجة
            // وإذا لم يكن، نعالج الـ Prompt كاملاً لأول مرة
            let input_to_process = if _token_step == 0 {
                current_input.clone()
            } else {
                // نأخذ آخر توكن فقط [Batch, 1, Hidden_Dim] لتقليل الحسابات
                current_input.i((.., current_input.dim(1)? - 1..))?
            };

            let mut hidden_states = input_to_process;
            
            // --- [ حلقة الـ 100 طبقة ] ---
            for _ in 0..100 {
                let residual = hidden_states.clone();
                hidden_states = model_block.forward_core(&hidden_states)?; 
                hidden_states = ((hidden_states + residual)? / 2.0)?;
            }

            // --- [ تحديث الـ KV-Cache ] ---
            // هنا نقوم بتخزين الحالة الحالية لتستخدمها الطبقة القادمة دون إعادة الحساب
            // ملاحظة: في Mamba الحقيقي، الـ Cache يخزن الـ "States"، وهنا نطبق الفكرة على الـ Sequence
            let (k_cached, _v_cached) = cache.update(hidden_states.clone(), hidden_states.clone())?;

            let logits = model_block.project_head(&k_cached)?; 
            let last_step_logits = logits.i((0, logits.dim(1)? - 1))?;
            let next_token_id = last_step_logits.argmax(0)?.to_scalar::<u32>()?;
            
            // العرض النصي
            if let Some(ref tok) = zumar_tokenizer {
                match tok.decode(&[next_token_id]) {
                    Ok(text) => print!("{} ", text),
                    Err(_) => print!("[ID_{}] ", next_token_id),
                }
            } else { print!("[ID_{}] ", next_token_id); }

            std::io::stdout().flush().ok();

            // تجهيز التوكن القادم (Embedding وهمي حالياً)
            let next_input_vec = Tensor::randn(0f32, 0.02f32, (1, 1, 768), &device)?;
            current_input = Tensor::cat(&[&current_input, &next_input_vec], 1)?;
        }
        
        println!("\n--------------------------------------------------");
        println!("✅ Optimized Cycle Complete.");
    }
    Ok(())
}
