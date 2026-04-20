mod layers;
mod routing;
mod config;
mod tokenizer;
mod loader;
mod kv_cache;
mod kernels;

use candle_core::{Result, Tensor};
use candle_nn::Module; 
use tokio::sync::mpsc;
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- 🌌 ZUMAR SOVEREIGN CORE ENGINE v2.2.0 ---");

    // 1. إعدادات النظام وتوجيه العتاد (CPU/GPU)
    let _config = config::ZumarConfig::default();
    let router = routing::HardwareRouter::new();
    let device = router.route("Inference Task");

    // 2. تجهيز مسار الأوزان والـ Loader
    let model_dir = "models/zumar-v1";
    let _ = fs::create_dir_all(model_dir);
    let loader = loader::ZumarLoader::new(model_dir);

    // 3. تحميل الأوزان الحقيقية عبر VarBuilder
    // vb سيقوم بتوزيع المصفوفات على طبقات الموديل هرمياً
    let vb = loader.load_weights(device)?;

    if kernels::is_kernel_available(device) {
        println!("🚀 Hardware Acceleration: Enabled (CUDA/Optimized Kernels)");
    }

    // 4. إعداد التوكينايزر والتحقق منه
    let tokenizer_path_str = loader.get_tokenizer_path();
    let tokenizer_path = std::path::Path::new(&tokenizer_path_str);

    if tokenizer_path.exists() {
        let tok = tokenizer::ZumarTokenizer::new(&tokenizer_path_str)?;
        println!("📂 Tokenizer loaded and verified.");
        
        // اختبار أولي للترميز
        let _test_ids = tok.encode("Zumar is online", device);
    } else {
        println!("ℹ️ Tokenizer not found at {:?}, running in raw tensor mode.", tokenizer_path);
    }

    // 5. إنشاء قناة التواصل لمعالجة المهام (Async Channel)
    let (tx, mut rx) = mpsc::channel::<String>(32);
    let test_tx = tx.clone();

    tokio::spawn(async move {
        let _ = test_tx.send("Zumar, execute 1-bit SSM inference".to_string()).await;
    });

    // 6. محرك الاستدلال الأساسي (Inference Loop)
    if let Some(prompt) = rx.recv().await {
        println!("📩 Task Received: \"{}\"", prompt);

        // --- التعديل الجوهري هنا ---
        // نمرر الـ VarBuilder (vb) بدلاً من الـ device ليتوافق مع التوقيع الجديد للدالة
        let model_block = layers::ZumarBlock::new(768, 768, vb.clone())?; 
        
        // إنشاء Tensor مدخلات (Input) بأبعاد (Batch, Sequence, Features)
        // 10 هو طول السياق التجريبي، و 768 هو d_model
        let input_tensor = Tensor::randn(0f32, 1f32, (1, 10, 768), device)?;
        
        println!("⚙️ Processing through Mamba + BitNet Layers...");
        
        // البدء بعملية المعالجة (Forward Pass)
        let output = model_block.forward(&input_tensor)?;

        // 7. إدارة الذاكرة الوسيطة (KV Cache)
        let mut cache = kv_cache::KVCache::new();
        cache.update(input_tensor, output.clone())?;
        
        println!("✅ Inference Successful. Output shape: {:?}", output.shape());
    }

    println!("--- 🚀 Zumar Engine is Standby ---");
    Ok(())
}
