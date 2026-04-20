mod layers;
mod routing;
mod config;
mod tokenizer;
mod loader;
mod kv_cache;
mod kernels;

use candle_core::{Result, Tensor, IndexOp, DType};
use std::io::Write;
use crate::kv_cache::KVCache;
use candle_nn::Module; 

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- 🌌 ZUMAR SOVEREIGN CORE ENGINE v2.2.0 ---");

    // 1. التوجيه الذكي للعتاد (CPU/GPU)
    let router = routing::HardwareRouter::new();
    let device = router.route("Inference Task");

    // 2. تحميل الأوزان والقاموس
    let model_dir = "models/zumar-v1";
    let loader = loader::ZumarLoader::new(model_dir);
    
    // تحميل الأوزان (سيقوم بتوليدها تلقائياً إذا لم تكن موجودة)
    let vb = loader.load_weights(&device)?;

    // تحميل الـ Tokenizer
    let zumar_tokenizer = if std::path::Path::new(&loader.get_tokenizer_path()).exists() {
        println!("📖 Tokenizer found, loading vocabulary...");
        Some(tokenizer::ZumarTokenizer::new(&loader.get_tokenizer_path())?)
    } else {
        println!("⚠️ Warning: Tokenizer not found, using raw IDs.");
        None
    };

    // 3. بناء الكتلة الهجينة (Hybrid: Mamba + Sparse MoE + 1-bit)
    let hidden_size = 1024;
    let vocab_size = 50257;

    let model_block = layers::ZumarBlock::new(
        hidden_size, 
        vocab_size, 
        vb.clone()
    )?; 

    // 4. إعداد دورة التوليد (Generation Loop)
    let mut _cache = KVCache::new();
    
    // نبدأ بتوكن البداية (ID 1) ونحوله إلى Embedding [1, 1, 1024]
    let mut current_input = model_block.embed(1, &device)?;

    println!("⚙️ Engine Status: Hybrid Mamba-MoE Core + 100 Layers Active");
    println!("--------------------------------------------------");

    for _token_step in 0..50 { 
        let mut hidden_states = current_input.clone();
        
        // --- [ حلقة الـ 100 طبقة السيادية ] ---
        for _layer_idx in 0..100 {
            let residual = hidden_states.clone();
            
            // نستخدم forward_core ليبقى البعد 1024 ولا يحدث Shape Mismatch
            hidden_states = model_block.forward_core(&hidden_states)?; 
            
            // جمع النتائج مع المدخلات (Residual Connection)
            hidden_states = (hidden_states + residual)?;
        }

        // --- [ التنبؤ بالتوكن القادم ] ---
        // تحويل المتجه النهائي من 1024 إلى احتمالات الكلمات 50257
        let logits = model_block.project_head(&hidden_states)?;
        
        // أخذ آخر "نبضة" (Token) في التسلسل
        let last_step_logits = logits.i((0, logits.dim(1)? - 1))?;
        
        // اختيار الكلمة الأعلى احتمالاً
        let next_token_id = last_step_logits.argmax(0)?.to_scalar::<u32>()?;
        
        // عرض النص الناتج
        if let Some(ref tok) = zumar_tokenizer {
            if let Ok(text) = tok.decode(&[next_token_id]) {
                print!("{}", text); 
            } else {
                print!("[ID_{}] ", next_token_id);
            }
        } else {
            print!("[ID_{}] ", next_token_id);
        }

        std::io::stdout().flush().ok();

        // تحديث المدخلات للخطوة القادمة (تحويل الـ ID المختار إلى Embedding جديد)
        current_input = model_block.embed(next_token_id, &device)?; 
    }
    
    println!("\n--------------------------------------------------");
    println!("✅ Sovereign Generation Complete.");
    Ok(())
}
