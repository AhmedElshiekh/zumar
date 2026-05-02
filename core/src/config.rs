use serde::{Deserialize, Serialize};

/// ✅ ZumarConfig — مصدر الحقيقة الوحيد لجميع أبعاد النموذج
///
/// الإصلاح: توحيد القيم مع main.rs
///   قبل:  vocab_size=32000, hidden_size=768, num_heads=12
///   بعد:  vocab_size=50257, hidden_size=1024, num_heads=16
///
/// الاستخدام في main.rs:
///   let cfg = ZumarConfig::default();
///   let mut model = ZumarModel::new(cfg.vocab_size, cfg.hidden_size, ...);
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZumarConfig {
    pub vocab_size:    usize,   // حجم القاموس — يتوافق مع GPT-2 tokenizer
    pub hidden_size:   usize,   // أبعاد التمثيل الداخلي
    pub num_layers:    usize,   // عدد الطبقات
    pub num_heads:     usize,   // عدد رؤوس الـ Attention
    pub kv_heads:      usize,   // عدد رؤوس K,V (MQA: 1، GQA: أقل من num_heads)
    pub num_experts:   usize,   // عدد الخبراء في MoE
    pub top_k:         usize,   // عدد الخبراء المُفعَّلين لكل token
    pub d_state:       usize,   // حجم حالة SSM في Mamba
    pub d_conv:        usize,   // حجم نافذة Conv1d في Mamba
    pub expand:        usize,   // معامل التوسع في Mamba (d_inner = hidden * expand)
    pub max_seq_len:   usize,   // الحد الأقصى لطول التسلسل (KV Cache)
    pub bit_precision: u8,      // دقة التكميم (2 لـ BitNet b1.58)
    pub use_flash_attn: bool,   // تفعيل Flash Attention
}

impl Default for ZumarConfig {
    fn default() -> Self {
        Self {
            // ✅ متوافق مع main.rs
            vocab_size:    50257,  // GPT-2 vocabulary
            hidden_size:   1024,
            num_layers:    12,
            num_heads:     16,
            kv_heads:      1,      // MQA: رأس واحد لـ K,V
            num_experts:   8,
            top_k:         2,
            // Mamba config
            d_state:       16,
            d_conv:        4,
            expand:        2,
            // Runtime
            max_seq_len:   2048,
            bit_precision: 2,      // BitNet b1.58 = 2-bit
            use_flash_attn: true,
        }
    }
}

impl ZumarConfig {
    /// حجم d_inner في Mamba
    pub fn d_inner(&self) -> usize {
        self.hidden_size * self.expand
    }

    /// حجم كل رأس attention
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// عدد المعاملات التقريبي (مليون)
    pub fn approx_params_m(&self) -> f64 {
        let attn    = 4 * self.hidden_size * self.hidden_size;
        let moe     = self.num_experts * self.hidden_size * self.hidden_size;
        let mamba   = 4 * self.hidden_size * self.d_inner();
        let embed   = self.vocab_size * self.hidden_size;
        let per_layer = attn + moe + mamba;
        let total   = embed + per_layer * self.num_layers + embed;
        total as f64 / 1_000_000.0
    }

    /// حفظ إلى JSON
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// تحميل من JSON
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// تحميل أو استخدام القيم الافتراضية
    pub fn load_or_default(path: &str) -> Self {
        Self::load(path).unwrap_or_default()
    }
}
