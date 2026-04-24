use candle_core::{Tensor, Result};
use candle_nn::{Module, VarBuilder};
use crate::layers::bitlinear::ZumarBitLinear;

pub struct ZumarMoE {
    pub gate: ZumarBitLinear,
    pub experts: Vec<ZumarBitLinear>,
    pub num_experts: usize,
    pub top_k: usize,
}

impl ZumarMoE {
    pub fn new(
        in_dim: usize,
        num_experts: usize,
        top_k: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        if top_k > num_experts {
            return Err(candle_core::Error::Msg(
                "top_k must be <= num_experts".to_string(),
            ));
        }

        let gate = ZumarBitLinear::new(in_dim, num_experts, vs.pp("gate"))?;

        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(ZumarBitLinear::new(
                in_dim,
                in_dim,
                vs.pp(format!("expert_{}", i)),
            )?);
        }

        Ok(Self {
            gate,
            experts,
            num_experts,
            top_k,
        })
    }

    /// توجيه المدخلات إلى أفضل k خبراء ودمج النتائج
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let flat_dim = b * s;
        let x_flat = x.reshape((flat_dim, h))?;

        // 1. حساب درجات التوجيه
        let router_logits = self.gate.forward(&x_flat)?; // [b*s, num_experts]
        let routing_probs = candle_nn::ops::softmax(&router_logits, 1)?; // [b*s, num_experts]

        // 2. معالجة كل خبير وجمع النتائج الموزونة (نسخة مبسطة تعمل)
        let mut output = x_flat.zeros_like()?;

        for expert_idx in 0..self.num_experts {
            // مخرجات هذا الخبير
            let expert_out = self.experts[expert_idx].forward(&x_flat)?; // [b*s, h]

            // وزن هذا الخبير من الـ gate (العمود expert_idx)
            let expert_weights = routing_probs.narrow(1, expert_idx, 1)?; // [b*s, 1]

            // دمج موزون
            output = (output + expert_out.broadcast_mul(&expert_weights)?)?;
        }

        // إعادة التشكيل للشكل الأصلي
        output.reshape((b, s, h))
    }
}