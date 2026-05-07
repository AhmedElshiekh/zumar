// #[warn(unused_imports)]
#[allow(unused)]
#[allow(dead_code)]
use tokenizers::{
    Tokenizer,
    Trainer,
    models::bpe::{BPE, BpeTrainer, BpeTrainerBuilder},
    pre_tokenizers::whitespace::Whitespace,
    normalizers::NFKC,
    processors::template::TemplateProcessing,
    decoders::bpe::BPEDecoder,
    AddedToken,
};
use candle_core::{Device, Tensor, Result as CandleResult};
use std::path::{Path, PathBuf};
use std::collections::HashMap;

const PAD_TOKEN: &str = "<pad>";
const UNK_TOKEN: &str = "<unk>";
const BOS_TOKEN: &str = "<s>";
const EOS_TOKEN: &str = "</s>";
const MASK_TOKEN: &str = "<mask>";

pub struct ZumarTokenizer {
    tokenizer: Tokenizer,
}

impl ZumarTokenizer {
    pub fn new(path: &str) -> CandleResult<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| candle_core::Error::Msg(
                format!("Tokenizer load error: {}", e)
            ))?;
        Ok(Self { tokenizer })
    }

    pub fn load_or_train(
        tokenizer_path: &str,
        data_dir:       &str,
        vocab_size:     usize,
    ) -> CandleResult<Self> {
        if Path::new(tokenizer_path).exists() {
            println!("📖 Loading tokenizer from {}", tokenizer_path);
            Self::new(tokenizer_path)
        } else {
            println!("🔨 Tokenizer not found — training from {}", data_dir);
            let tok = Self::train_from_dir(data_dir, vocab_size, tokenizer_path)?;
            Ok(tok)
        }
    }
    
    pub fn train_from_dir(
        data_dir:    &str,
        vocab_size:  usize,
        output_path: &str,
    ) -> CandleResult<Self> {
        use tokenizers::models::bpe::{BPE, BpeTrainer};
        use tokenizers::models::ModelWrapper;
        use tokenizers::normalizers::NFKC;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDec;
        use tokenizers::processors::byte_level::ByteLevel as ByteLevelProc;
        use tokenizers::AddedToken;
        use std::io::{BufReader, BufRead, Read};
        use std::fs::File;
    
        println!("\n🔨 BUILDING BPE TOKENIZER (streaming)");
        println!("   Vocab size: {}", vocab_size);
    
        // 1. إعداد المدرب
        let mut trainer = BpeTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(1)
            .show_progress(true)
            .special_tokens(vec![
                AddedToken::from("<|endoftext|>", true),
                AddedToken::from("<|padding|>", true),
                AddedToken::from("<|unknown|>", true),
            ])
            .build();
    
        // 2. إنشاء مكرر كسول (lazy iterator) يقرأ الملفات سطراً سطراً
        //    دون تحميلها كلها في الذاكرة.
        let data_dir = data_dir.to_string();
        let path_list: Vec<_> = std::fs::read_dir(&data_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        
        let lazy_lines = move || {
            path_list
                .clone() // نستعمل نسخة لضمان استدعاء المكرر عدة مرات (feed قد يحتاج ذلك)
                .into_iter()
                .flat_map(|path| {
                    let file = File::open(&path).ok();
                    let lines: Vec<String> = file
                        .map(|f| BufReader::new(f).lines().filter_map(|l| l.ok()).collect())
                        .unwrap_or_default();
                    lines.into_iter()
                })
                .chain(
                    std::iter::once("The quick brown fox jumps over the lazy dog.".to_string())
                        .filter(move |_| path_list.is_empty())
                )
        };
    
        // 3. تغذية النصوص (باستخدام دالة تحول كل سطر إلى قائمة كلمات أولية)
        trainer
            .feed(lazy_lines(), |text: &str| {
                Ok(text
                    .bytes()
                    .map(|b| String::from_utf8_lossy(&[b]).into_owned())
                    .collect::<Vec<_>>())
            })
            .map_err(|e| candle_core::Error::Msg(format!("BPE feed failed: {}", e)))?;
    
        // 4. تدريب نموذج BPE
        let mut bpe = BPE::default();
        trainer
            .train(&mut bpe)
            .map_err(|e| candle_core::Error::Msg(format!("BPE training failed: {}", e)))?;
    
        // 5. بناء Tokenizer (باقي الخطوات كما هي)
        let mut tokenizer = tokenizers::Tokenizer::new(ModelWrapper::BPE(bpe));
        tokenizer.with_normalizer(Some(NFKC));
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_decoder(Some(ByteLevelDec::default()));
        tokenizer.with_post_processor(Some(ByteLevelProc::default()));
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|endoftext|>", true),
            AddedToken::from("<|padding|>", true),
            AddedToken::from("<|unknown|>", true),
        ]);
    
        let real_vocab = tokenizer.get_vocab_size(true);
        println!("   ✅ Tokenizer ready → vocab size: {}", real_vocab);
    
        if let Some(parent) = std::path::Path::new(output_path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        tokenizer.save(output_path, false)
            .map_err(|e| candle_core::Error::Msg(format!("Save failed: {}", e)))?;
    
        Ok(Self { tokenizer })
    }
        

    pub fn encode_ids(&self, text: &str) -> CandleResult<Vec<u32>> {
        let enc = self.tokenizer.encode(text, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    // إضافة الدالة المطلوبة
    pub fn encode_to_vec(&self, text: &str) -> CandleResult<Vec<u32>> {
        self.encode_ids(text)
    }

    pub fn encode(&self, text: &str, device: &Device) -> CandleResult<Tensor> {
        let ids = self.encode_ids(text)?;
        Tensor::new(ids.as_slice(), device)
    }

    pub fn decode(&self, ids: &[u32]) -> CandleResult<String> {
        self.tokenizer.decode(ids, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn get_vocab_map(&self) -> HashMap<String, u32> {
        let mut map = HashMap::new();
        let vocab = self.tokenizer.get_vocab(true);
        for (word, id) in vocab.iter() {
            map.insert(word.clone(), *id);
        }
        map
    }

    fn quick_test(tokenizer: &Tokenizer) {
        let test_texts = [
            "Hello world",
            "The quick brown fox",
        ];
        println!("\n   🧪 Quick test:");
        for text in &test_texts {
            if let Ok(enc) = tokenizer.encode(*text, false) {
                let ids    = enc.get_ids();
                let tokens = enc.get_tokens();
                println!("     \"{}\" → {:?} ({} tokens)", text, tokens, ids.len());
            }
        }
    }
}

fn collect_text_files(dir: &str) -> CandleResult<Vec<PathBuf>> {
    // ... (نفس الدالة السابقة لديك، لا تغيير)
    // سأضع نسخة مبسطة
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "txt" || ext == "md" || ext == "text" {
                        files.push(path);
                    }
                }
            }
        }
    }
    Ok(files)
}