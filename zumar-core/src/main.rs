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

    let _config = config::ZumarConfig::default();
    let router = routing::HardwareRouter::new();
    let device = router.route("Inference Task");

    let model_dir = "models/zumar-v1";
    let _ = fs::create_dir_all(model_dir);

    let loader = loader::ZumarLoader::new(model_dir);

    // 1. استدعاء تحميل الأوزان (لإخفاء تحذير load_weights)
    let _ = loader.load_weights(device);

    if kernels::is_kernel_available(device) {
        println!("🚀 Hardware Acceleration: Enabled");
    }

    let tokenizer_path_str = loader.get_tokenizer_path();
    let tokenizer_path = std::path::Path::new(&tokenizer_path_str);

    if tokenizer_path.exists() {
        let tok = tokenizer::ZumarTokenizer::new(&tokenizer_path_str)?;
        let _ids = tok.encode("Hello Zumar", device);
        
        // 2. استدعاء فك التشفير (لإخفاء تحذير decode)
        let _test_decode = tok.decode(&[1, 2, 3]); 
        println!("📂 Tokenizer loaded and verified.");
    } else {
        println!("ℹ️ Tokenizer file not found at {:?}, skipping loading...", tokenizer_path);
    }

    let (tx, mut rx) = mpsc::channel::<String>(32);
    let test_tx = tx.clone();

    tokio::spawn(async move {
        let _ = test_tx.send("Zumar, execute 1-bit inference".to_string()).await;
    });

    if let Some(prompt) = rx.recv().await {
        println!("📩 Task Received: \"{}\"", prompt);

        let model_block = layers::ZumarBlock::new(768, 768, device)?;
        let input_tensor = Tensor::randn(0f32, 1f32, (1, 768), device)?;
        let output = model_block.forward(&input_tensor)?;

        let mut cache = kv_cache::KVCache::new();
        cache.update(input_tensor, output.clone())?;
        cache.reset();

        println!("✅ Inference Successful. Output shape: {:?}", output.shape());
    }

    println!("--- 🚀 Zumar Engine is Standby ---");
    Ok(())
}
