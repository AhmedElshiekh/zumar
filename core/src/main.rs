mod layers;
mod routing;
mod config;
mod tokenizer;
mod loader;
mod kv_cache;
mod kernels;

use candle_core::{Result, Tensor, IndexOp};
use candle_nn::Module; 
use tokio::sync::mpsc;
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- 🌌 ZUMAR SOVEREIGN CORE ENGINE v2.2.0 ---");

    let _config = config::ZumarConfig::default();
    let router = routing::HardwareRouter::new();
    let device = router.route("Inference Task");

    let model_dir = "models/zumar-v1";
    let _ = fs::create_dir_all(model_dir);
    let loader = loader::ZumarLoader::new(model_dir);

    let vb = loader.load_weights(device)?;

    // إعداد التوكينايزر باستخدام المسار من الـ loader لإزالة التحذيرات
    let tok_path = loader.get_tokenizer_path();
    let zumar_tokenizer = if std::path::Path::new(&tok_path).exists() {
        Some(tokenizer::ZumarTokenizer::new(&tok_path)?)
    } else {
        None
    };

    if kernels::is_kernel_available(device) {
        println!("🚀 Hardware Acceleration: Enabled (CUDA/Optimized Kernels)");
    }

    let (tx, mut rx) = mpsc::channel::<String>(32);
    let test_tx = tx.clone();

    tokio::spawn(async move {
        let _ = test_tx.send("Execute Generation Loop".to_string()).await;
    });
   
   
    if let Some(prompt) = rx.recv().await {
        println!("📩 Task Received: \"{}\"", prompt);
        
        if let Some(ref tok) = zumar_tokenizer {
            let _prompt_ids = tok.encode(&prompt, device)?;
        }
        
        let vocab_size = 32000;
        let model_block = layers::ZumarBlock::new(768, vocab_size, vb.clone())?; 
        
        // نبدأ بـ Tensor مدخلات وهمي (لأننا لم نقم ببناء الـ Embedding Layer بعد)
        let mut current_input = Tensor::randn(0f32, 1f32, (1, 5, 768), device)?;
        
        println!("⚙️ Engine Status: Running Auto-Regressive Loop...");
        println!("--------------------------------------------------");

        for _i in 0..12 {
            let logits = model_block.forward(&current_input)?;
            let last_step_logits = logits.i((0, logits.dim(1)? - 1))?;
            
            // استخراج الـ ID المترجم من النواة
            let next_token_id = last_step_logits.argmax(0)?.to_scalar::<u32>()?;
            
            // محاولة فك الشفرة إلى نص باستخدام التوكينايزر
            if let Some(ref tok) = zumar_tokenizer {
                match tok.decode(&[next_token_id]) {
                    Ok(text) => print!("{} ", text),
                    Err(_) => print!("[ID_{}] + ", next_token_id),
                }
            } else {
                print!("[ID_{}] ", next_token_id);
            }

            use std::io::Write;
            std::io::stdout().flush().ok();

            // تحديث السياق للخطوة القادمة
            let next_input_vec = Tensor::randn(0f32, 1f32, (1, 1, 768), device)?;
            current_input = Tensor::cat(&[&current_input, &next_input_vec], 1)?;
        }
        
        println!("\n--------------------------------------------------");
        println!("✅ Generation Cycle Complete.");
    }

    println!("--- 🚀 Zumar Engine is Standby ---");
    Ok(())
}
