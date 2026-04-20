use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZumarConfig {
    pub vocab_size: usize,      // حجم القاموس (عدد الكلمات التي يعرفها)
    pub hidden_size: usize,     // حجم الأبعاد الداخلية (مثل 768 أو 4096)
    pub num_layers: usize,      // عدد طبقات الـ 1-bit Mamba
    pub num_heads: usize,       // عدد رؤوس الـ Attention
    pub d_state: usize,         // حالة الـ SSM في معماري مامبا
    pub bit_precision: u8,      // نوع التكميم (1 لـ BitNet)
    pub use_flash_attn: bool,   // تفعيل Flash-Attention 3
}

impl Default for ZumarConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            d_state: 16,
            bit_precision: 1,
            use_flash_attn: true,
        }
    }
}
