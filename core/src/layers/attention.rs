use candle_core::{Tensor, Result, D};

pub struct ZumarFlashAttention {
    pub n_heads:       usize,
    pub kv_heads:      usize,
    pub head_dim:      usize,
    pub softmax_scale: f32,
}

impl ZumarFlashAttention {
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        Self { n_heads, kv_heads: 1, head_dim, softmax_scale }
    }

    /// ✅ إصلاح GQA/MQA: توسيع K, V بالطريقة الصحيحة
    /// المشكلة القديمة: (kv_heads, repeat, seq, head_dim) → reshape خلط الأبعاد
    /// الحل: transpose أولاً ثم flatten kv_heads*repeat فقط
    fn expand_kv(kv: &Tensor, n_heads: usize, kv_heads: usize) -> Result<Tensor> {
        if n_heads == kv_heads { return Ok(kv.clone()); }

        let repeat = n_heads / kv_heads;
        // kv shape: [b, kv_heads, seq, head_dim]
        let (b, _, seq, hd) = kv.dims4()?;

        // ✅ repeat_interleave صحيح: توسيع على محور kv_heads
        // [b, kv_heads, seq, hd] → [b, kv_heads, 1, seq, hd]
        //                         → [b, kv_heads, repeat, seq, hd]
        //                         → [b, kv_heads*repeat, seq, hd]
        kv.unsqueeze(2)?                                                    // [b, kv_heads, 1, seq, hd]
          .expand((b, kv_heads, repeat, seq, hd))?                         // [b, kv_heads, repeat, seq, hd]
          .reshape((b, kv_heads * repeat, seq, hd))                        // ✅ flatten الأبعاد الصحيحة فقط
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _total_dim) = q.dims3()?;

        // ── إعادة تشكيل Q, K, V ─────────────────────────────────
        // [b, seq, n_heads*head_dim] → [b, n_heads, seq, head_dim]
        let q = q.reshape((b_sz, seq_len, self.n_heads,   self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.kv_heads,  self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.kv_heads,  self.head_dim))?.transpose(1, 2)?;

        // ✅ توسيع K, V بالطريقة الصحيحة
        let k = Self::expand_kv(&k, self.n_heads, self.kv_heads)?;
        let v = Self::expand_kv(&v, self.n_heads, self.kv_heads)?;

        // ── GPU: Flash Attention مدمج ────────────────────────────
        #[cfg(feature = "cuda")]
        if q.device().is_cuda() {
            let out = candle_transformers::ops::flash_attn(
                &q, &k, &v, self.softmax_scale, false,
            )?;
            return out.transpose(1, 2)?.reshape((b_sz, seq_len, self.n_heads * self.head_dim));
        }

        // ── CPU: Tiled Flash Attention مع Online Softmax صحيح ───
        self.tiled_attention_cpu(&q, &k, &v, b_sz, seq_len)
    }

    /// ✅ Online Softmax الصحيح رياضياً
    ///
    /// المشكلة القديمة:
    ///   out += softmax(scores_tile) @ V_tile
    ///   → يجمع softmax من نوافذ مختلفة مباشرة — خطأ رياضي
    ///   → النتيجة مضخمة بعدد الـ tiles
    ///
    /// الحل الصحيح (Flash Attention Algorithm):
    ///   نتتبع max_i و sum_i ونُعيد التطبيع بعد كل tile
    fn tiled_attention_cpu(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        b_sz: usize, seq_len: usize,
    ) -> Result<Tensor> {
        let scale     = self.softmax_scale as f64;
        let tile_size = 64.min(seq_len);
        let device    = q.device();

        // مخرج الـ attention لكل q_tile
        let mut q_tile_outputs: Vec<Tensor> = Vec::new();

        for t_start in (0..seq_len).step_by(tile_size) {
            let t_len   = (t_start + tile_size).min(seq_len) - t_start;
            let q_tile  = q.narrow(2, t_start, t_len)?;
            // [b, n_heads, t_len, head_dim]

            // ✅ تتبع max و sum لكل (b, head, q_pos)
            // نبدأ بقيم ابتدائية
            let mut running_max = Tensor::full(
                f32::NEG_INFINITY,
                (b_sz, self.n_heads, t_len, 1),
                device,
            )?;
            let mut running_sum = Tensor::zeros(
                (b_sz, self.n_heads, t_len, 1),
                candle_core::DType::F32,
                device,
            )?;
            let mut running_out = Tensor::zeros(
                (b_sz, self.n_heads, t_len, self.head_dim),
                candle_core::DType::F32,
                device,
            )?;

            for k_start in (0..seq_len).step_by(tile_size) {
                let k_len  = (k_start + tile_size).min(seq_len) - k_start;
                let k_tile = k.narrow(2, k_start, k_len)?;
                let v_tile = v.narrow(2, k_start, k_len)?;

                // scores: [b, n_heads, t_len, k_len]
                let scores = (q_tile.matmul(&k_tile.transpose(2, 3)?)? * scale)?;

                // ── Online Softmax Update ─────────────────────────────
                // 1. max الحالي لهذا الـ tile: [b, n_heads, t_len, 1]
                let tile_max = scores.max_keepdim(D::Minus1)?;

                // 2. max الجديد = max(running_max, tile_max)
                let new_max = running_max.maximum(&tile_max)?;

                // 3. تصحيح الـ running_out والـ running_sum بفارق الـ max
                //    correction = exp(old_max - new_max)
                let correction = (running_max - &new_max)?.exp()?;
                running_out = (running_out.broadcast_mul(&correction))?;
                running_sum = (running_sum.broadcast_mul(&correction))?;

                // 4. exp(scores - new_max): [b, n_heads, t_len, k_len]
                let scores_shifted = (scores - new_max.broadcast_as(scores.shape())?)?;
                let exp_scores     = scores_shifted.exp()?;

                // 5. تحديث المجموع
                let tile_sum = exp_scores.sum_keepdim(D::Minus1)?;
                running_sum  = (running_sum + tile_sum)?;

                // 6. تحديث المخرج: out += exp_scores @ V_tile
                //    exp_scores: [b, n_heads, t_len, k_len]
                //    v_tile:     [b, n_heads, k_len, head_dim]
                running_out  = (running_out + exp_scores.matmul(&v_tile)?)?;

                // تحديث running_max
                running_max  = new_max;
            }

            // ✅ التطبيع النهائي: out / sum
            // running_sum: [b, n_heads, t_len, 1]
            let eps = Tensor::full(1e-6f32, running_sum.shape(), device)?;
            let sum_safe = (running_sum + eps)?;
            let tile_out = running_out.broadcast_div(&sum_safe)?;
            // [b, n_heads, t_len, head_dim]

            q_tile_outputs.push(tile_out);
        }

        // دمج جميع الـ tiles على محور seq
        let attn_output = Tensor::cat(
            &q_tile_outputs.iter().map(|t| t as &Tensor).collect::<Vec<_>>(),
            2,
        )?;
        // [b, n_heads, seq_len, head_dim] → [b, seq_len, n_heads*head_dim]
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.n_heads * self.head_dim))
    }
}
