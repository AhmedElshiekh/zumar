use candle_core::{Tensor, Result, Device};
use candle_nn::{Module, VarMap};

pub struct ZumarFlashAttention {
    n_heads: usize,
    head_dim: usize,
    softmax_scale: f16,
}

impl ZumarFlashAttention {
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        let softmax_scale = 1.0 / (head_dim as f16).sqrt();
        Self {
            n_heads,
            head_dim,
            softmax_scale,
        }
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = q.dims3()?;
        
        // إعادة تشكيل المصفوفات لتقسيمها على رؤوس الانتباه (Heads)
        let q = q.reshape((b_sz, seq_len, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.n_heads, self.head_dim))?.transpose(1, 2)?;

        // تنفيذ Flash-Attention إذا كان العتاد يدعم CUDA
        #[cfg(feature = "cuda")]
        if q.device().is_cuda() {
            // استدعاء المسرع الخاص بـ Flash-Attention 3
            return candle_transformers::ops::flash_attn(&q, &k, &v, self.softmax_scale, false);
        }

        // كود احتياطي (Fallback) في حال عدم وجود CUDA
        let att = (q.matmul(&k.t()?)? * self.softmax_scale as f64)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        att.matmul(&v)
    }
}
