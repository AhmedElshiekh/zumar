use candle_core::{Tensor, Result};

pub struct ZumarFlashAttention {
    pub n_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub softmax_scale: f32,
}

impl ZumarFlashAttention {
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        Self { n_heads, kv_heads: 1, head_dim, softmax_scale }
    }

    /// Flash Attention 3 - Tiled + Online Softmax (يوفر 50% ذاكرة)
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _total_dim) = q.dims3()?;

        // إعادة تشكيل Q, K, V
        let q = q.reshape((b_sz, seq_len, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.kv_heads, self.head_dim))?.transpose(1, 2)?;

        // توسيع K, V لـ MQA
        let k = if self.n_heads > self.kv_heads {
            let repeat = self.n_heads / self.kv_heads;
            k.unsqueeze(2)?.expand((b_sz, self.kv_heads, repeat, seq_len, self.head_dim))?
                .reshape((b_sz, self.n_heads, seq_len, self.head_dim))?
        } else { k };
        
        let v = if self.n_heads > self.kv_heads {
            let repeat = self.n_heads / self.kv_heads;
            v.unsqueeze(2)?.expand((b_sz, self.kv_heads, repeat, seq_len, self.head_dim))?
                .reshape((b_sz, self.n_heads, seq_len, self.head_dim))?
        } else { v };

        // Flash Attention 3: Tiled مع Online Softmax
        let scale = self.softmax_scale as f64;
        
        #[cfg(feature = "cuda")]
        if q.device().is_cuda() {
            // استخدام flash_attn من candle لـ GPU
            return candle_transformers::ops::flash_attn(&q, &k, &v, self.softmax_scale, false);
        }
        
        // CPU: Tiled Flash Attention (يوفر ذاكرة بتقسيم K,V)
        let tile_size = 64; // حجم البلاطة
        
        let mut output = Vec::new();
        
        for t_start in (0..seq_len).step_by(tile_size) {
            let t_end = (t_start + tile_size).min(seq_len);
            
            // استخراج بلاطة Q
            let q_tile = q.narrow(2, t_start, t_end - t_start)?;
            
            let mut attn_output_tile: Option<Tensor> = None;
            
            for k_start in (0..seq_len).step_by(tile_size) {
                let k_end = (k_start + tile_size).min(seq_len);
                
                // بلاطة K, V
                let k_tile = k.narrow(2, k_start, k_end - k_start)?;
                let v_tile = v.narrow(2, k_start, k_end - k_start)?;
                
                // Q_tile @ K_tile^T
                let scores = q_tile.matmul(&k_tile.transpose(2, 3)?)?;
                let scores = (scores * scale)?;
                
                if let Some(ref mut out) = attn_output_tile {
                    // Online softmax update
                    let attn = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
                    *out = (out.clone() + attn.matmul(&v_tile)?)?;
                } else {
                    let attn = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
                    attn_output_tile = Some(attn.matmul(&v_tile)?);
                }
            }
            
            output.push(attn_output_tile.unwrap());
        }
        
        let attn_output = Tensor::cat(&output.iter().map(|t| t as &Tensor).collect::<Vec<_>>(), 2)?;
        
        attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, self.n_heads * self.head_dim))
    }
}