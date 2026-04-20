use tokenizers::Tokenizer;
use candle_core::{Device, Tensor, Result};

pub struct ZumarTokenizer {
    tokenizer: Tokenizer,
}

impl ZumarTokenizer {
    pub fn new(path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer Load Error: {}", e)))?;
        Ok(Self { tokenizer })
    }

  /// يحول النص إلى Tensor من الـ IDs
    #[allow(dead_code)]
    pub fn encode(&self, text: &str, device: &Device) -> Result<Tensor> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        let ids = encoding.get_ids();
        Tensor::new(ids, device)
    }
    
    
  /// يحول رقم (ID) واحد أو مجموعة أرقام إلى نص
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}
