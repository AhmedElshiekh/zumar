mod layers;
mod routing;
mod config;
mod tokenizer;
mod loader;
mod kv_cache;
mod kernels;

use crate::layers::ZumarModel;
use candle_core::{Result, Tensor};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<()> {
    // --- التنسيق البصري ---
    print!("\x1b[2J");
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
    println!("\x1b[1;36m--- 🌌 ZUMAR SOVEREIGN CORE ENGINE v3.0.0 ---\x1b[0m");
    println!("\x1b[90m--------------------------------------------------\x1b[0m");

    // --- إعدادات النموذج ---
    let hidden_size: usize = 1024;
    let vocab_size: usize = 50257;
    let num_layers: usize = 12;
    let num_experts: usize = 8;
    let top_k: usize = 2;
    let n_heads: usize = 16;

    // --- اختيار الجهاز ---
    let router = routing::HardwareRouter::new();
    let device = router.route("Inference Task");
    println!("🖥️  Running on: {:?}", device);

    // --- تحميل الأوزان ---
    let model_dir = "models/zumar-v1";
    let loader = loader::ZumarLoader::new(model_dir);
    let vb = loader.load_weights(&device)?;

    // --- تحميل المحلل اللغوي ---
    let zumar_tokenizer = {
        let tokenizer_path = loader.get_tokenizer_path();
        if std::path::Path::new(&tokenizer_path).exists() {
            println!("✅ Tokenizer: Loaded.");
            Some(tokenizer::ZumarTokenizer::new(&tokenizer_path)?)
        } else {
            println!("⚠️  Tokenizer: Missing. Using raw token IDs.");
            None
        }
    };

    // --- بناء النموذج ---
    println!("🔧 Building ZumarModel with {} layers...", num_layers);
    let model = ZumarModel::new(
        vocab_size,
        hidden_size,
        num_layers,
        num_experts,
        top_k,
        n_heads,
        vb.clone(),
    )?;
    println!("✅ Model: Built successfully.");
    println!(
        "📊 Config: {} layers, {} experts (top-{}), {} heads",
        num_layers, num_experts, top_k, n_heads
    );

    // --- إعدادات التوليد ---
    let temperature: f64 = 0.8;
    let max_tokens: usize = 120;
    let repetition_penalty: f32 = 1.5;
    let eos_token_id: u32 = 50256;

    println!("\x1b[1;34m💡 Hint: Type your prompt or 'exit' to quit.\x1b[0m");
    println!();

    // --- حلقة المحادثة ---
    loop {
        print!("\x1b[1;32m┌───[ 👤 User ]\x1b[0m\n\x1b[1;32m└─> \x1b[0m");
        io::stdout().flush().ok();

        let mut user_input = String::new();
        if io::stdin().read_line(&mut user_input).is_err() {
            break;
        }
        let prompt = user_input.trim();
        if prompt == "exit" {
            break;
        }
        if prompt.is_empty() {
            continue;
        }

        print!("\x1b[1;30m[🧠 Zumar Thinking...]\x1b[0m");
        io::stdout().flush().ok();

        // --- ترميز المدخلات ---
        let input_ids: Vec<u32> = if let Some(ref tok) = zumar_tokenizer {
            match tok.encode(prompt, &device) {
                Ok(tensor) => {
                    tensor.flatten_all()
                        .ok()
                        .and_then(|t| t.to_vec1::<u32>().ok())
                        .unwrap_or(vec![1])
                }
                Err(_) => vec![1],
            }
        } else {
            vec![1]
        };

        // --- توليد النص ---
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut current_token = *input_ids.last().unwrap_or(&1);

        print!("\r\x1b[K\x1b[1;36m┌───[ 🤖 Zumar ]\x1b[0m");
        print!("\n\x1b[1;36m└─> \x1b[0m");
        io::stdout().flush().ok();

        for _step in 0..max_tokens {
            // 1. تضمين الرمز الحالي - يُرجع [1, 1, hidden_size]
            let embedding = model.embed(current_token, &device)?;

            // 2. تمريرة أمامية عبر جميع الطبقات
            // forward يتوقع [b, s, h] ويعيد [b, s, h]
            let hidden_states = model.forward(&embedding)?;

            // 3. استخراج لوغاريتمات آخر رمز
            // hidden_states: [1, 1, vocab_size] بعد lm_head
            let logits_flat = hidden_states.flatten_all()?;
            let mut logits_vec = logits_flat.to_vec1::<f32>()?;

            // 4. تطبيق عقوبة التكرار
            for &prev_token in &generated_tokens {
                let idx = prev_token as usize;
                if idx < logits_vec.len() {
                    if logits_vec[idx] > 0.0 {
                        logits_vec[idx] /= repetition_penalty;
                    } else {
                        logits_vec[idx] *= repetition_penalty;
                    }
                }
            }

            // 5. تطبيق درجة الحرارة و softmax
            let processed_logits = Tensor::new(logits_vec.as_slice(), &device)?;
            let scaled = (&processed_logits / temperature)?;
            let probs = candle_nn::ops::softmax(&scaled, 0)?;

            // 6. اختيار الرمز التالي
            let next_token = probs.argmax(0)?.to_scalar::<u32>()?;

            // 7. تسجيل الرمز
            generated_tokens.push(next_token);
            current_token = next_token;

            // 8. فك الترميز وعرضه
            if let Some(ref tok) = zumar_tokenizer {
                if let Ok(text) = tok.decode(&[next_token]) {
                    print!("{}", text);
                    io::stdout().flush().ok();
                }
            } else {
                print!("[{}] ", next_token);
                io::stdout().flush().ok();
            }

            // 9. التوقف عند رمز النهاية
            if next_token == eos_token_id || next_token == 2 {
                break;
            }
        }

        println!();
        println!("\x1b[90m──────────────────────────────────────────────────\x1b[0m");
        println!(
            "\x1b[90mGenerated {} tokens\x1b[0m",
            generated_tokens.len()
        );
    }

    println!("\n\x1b[1;35m══════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[1;35m🛡️  ZUMAR SHUTTING DOWN - SOVEREIGNTY PRESERVED\x1b[0m");
    println!("\x1b[1;35m══════════════════════════════════════════════════\x1b[0m");
    Ok(())
}