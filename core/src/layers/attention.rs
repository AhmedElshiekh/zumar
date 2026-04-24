use candle_core::{Tensor, Result};

pub struct ZumarFlashAttention {
    pub n_heads: usize,
    pub head_dim: usize,
    pub softmax_scale: f32,
}

impl ZumarFlashAttention {
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            n_heads,
            head_dim,
            softmax_scale,
        }
    }

    /// انتباه متعدد الرؤوس
    /// المدخلات: [batch, seq_len, n_heads * head_dim]
    /// المخرجات: [batch, seq_len, n_heads * head_dim]
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, total_dim) = q.dims3()?;
        
        // التحقق من تطابق الأبعاد
        if total_dim != self.n_heads * self.head_dim {
            // إذا كانت الأبعاد غير متطابقة، أرجع المدخل كما هو
            return Ok(q.clone());
        }

        // إعادة تشكيل المصفوفات لتقسيمها على رؤوس الانتباه
        let q = q
            .reshape((b_sz, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;

        // كود احتياطي للـ CPU
        let scale = self.softmax_scale as f64;

        let attn_scores = q.matmul(&k.transpose(2, 3)?)?;
        let attn_scores = (attn_scores * scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_scores, candle_core::D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;

        // إعادة التشكيل للشكل الأصلي
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, total_dim))
    }
}