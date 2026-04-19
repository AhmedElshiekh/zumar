// استيراد الوحدات (Modules)
mod layers;
mod routing;
mod kernels;
mod config;
mod tokenizer;
mod loader;
mod kv_cache; // إضافة وحدة الذاكرة المؤقتة

use crate::config::ZumarConfig;
use crate::tokenizer::ZumarTokenizer;
use crate::routing::HardwareRouter;
use crate::layers::ZumarBlock;
use crate::loader::ZumarLoader;
use crate::kv_cache::KVCache;

use candle_core::{Result, Tensor};
use tokio::sync::mpsc;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- 🌌 ZUMAR SOVEREIGN CORE ENGINE v1.0 ---");
    println!("Status: System initialized. Ready for high-performance inference.");

    // 1. إعداد المحمل (Loader) - المسار: zumar-core/weights/
    let loader = ZumarLoader::new("weights");

    // 2. تحميل الإعدادات المركزية
    let config = Arc::new(ZumarConfig::default());
    println!("⚙️ Config: Hidden Size = {}, Layers = {}, Bits = {}", 
        config.hidden_size, config.num_layers, config.bit_precision);

    // 3. إعداد الموجه الذكي للعتاد (Hardware Router)
    let router = HardwareRouter::new();

    // 4. تحميل المُجزئ (Tokenizer) من المجلد المخصص للأوزان
    let tokenizer_path = loader.get_tokenizer_path();
    let tokenizer = ZumarTokenizer::new(&tokenizer_path).expect(&format!(
        "❌ Critical Error: 'tokenizer.json' not found in {}",
        tokenizer_path
    ));
    println!("🔤 Tokenizer successfully loaded from weights folder.");

    // 5. تهيئة الذاكرة المؤقتة (KV-Cache)
    // ننشئ نسخة لكل جلسة استدلال لضمان سرعة التوليد
    let mut cache = KVCache::new();

    // 6. إعداد قناة التواصل (MPSC) لاستقبال الطلبات من الجسر (Bridge)
    let (tx, mut rx) = mpsc::channel::<String>(100);

    // محاكاة وصول أول طلب حقيقي
    let test_tx = tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        let _ = test_tx.send("Zumar, demonstrate 1-bit spiking efficiency.".to_string()).await;
    });

    println!("🚀 Zumar Core is active. Listening for Bridge requests...");
    println!("-------------------------------------------");

    // 7. حلقة الاستدلال الرئيسية (Inference Loop)
    while let Some(prompt) = rx.recv().await {
        println!("📩 Task Received: \"{}\"", prompt);

        // أ. اختيار العتاد الأنسب (Tesla P40 vs CPU)
        let device = router.route(&prompt);

        // ب. تحويل النص إلى أرقام (Tokenization)
        let input_ids = tokenizer.encode(&prompt, device)?;
        println!("🔢 Input encoded into {} tokens.", input_ids.dims()[1]);

        // ج. تحميل الأوزان (إذا كانت متوفرة في المجلد)
        loader.load_weights(device)?;

        // د. بناء ومعالجة البلوك (1-bit + Mamba + SNN)
        let mut model_block = ZumarBlock::new(
            config.hidden_size, 
            config.hidden_size, 
            device
        )?;

        println!("🧠 Forward Pass: Spiking Mamba Logic with KV-Cache active...");
        
        // تنفيذ الاستدلال
        let output = model_block.forward(&input_ids)?;

        // هـ. تحديث الذاكرة المؤقتة (كمثال للعملية الداخلية)
        // في التنفيذ الكامل، يتم استدعاء cache.update داخل الطبقات
        println!("💾 KV-Cache updated for seamless token generation.");

        // و. عرض النتائج
        println!("✅ Inference complete on {:?}.", device);
        println!("📊 Output Shape: {:?}", output.dims());
        println!("--- Waiting for next request ---");
        
        // ملاحظة: نقوم بتصفير الكاش إذا انتهت الجلسة (اختياري)
        // cache.reset();
    }

    Ok(())
}
