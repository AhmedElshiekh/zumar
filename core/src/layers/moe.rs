use candle_core::{Tensor, Result};
use candle_nn::{Module, VarBuilder, Linear};

pub struct ZumarMoE {
    pub gate: Linear,
    pub experts: Vec<Linear>,
    pub num_experts: usize,
    pub top_k: usize,
}

impl ZumarMoE {
    pub fn new(in_dim: usize, num_experts: usize, top_k: usize, vs: VarBuilder) -> Result<Self> {
        let gate = candle_nn::linear(in_dim, num_experts, vs.pp("gate"))?;
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(candle_nn::linear(in_dim, in_dim, vs.pp(format!("expert_{}", i)))?);
        }
        Ok(Self { gate, experts, num_experts, top_k })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let flat_dim = b * s;
        let x_flat = x.reshape((flat_dim, h))?;

        // 1. Gate: [flat_dim, num_experts]
        let router_logits = self.gate.forward(&x_flat)?;
        
        // 2. Softmax على كل صف
        let routing_probs = candle_nn::ops::softmax(&router_logits, 1)?;
        
        // 3. لكل خبير، نأخذ مخرجاته موزونة باحتماله من الـ gate
        let mut output = x_flat.zeros_like()?;
        
        for expert_idx in 0..self.num_experts {
            // مخرجات هذا الخبير: [flat_dim, h]
            let expert_out = self.experts[expert_idx].forward(&x_flat)?;
            
            // وزن هذا الخبير من الـ gate: [flat_dim, 1]
            let expert_weight = routing_probs.narrow(1, expert_idx, 1)?;
            
            // دمج موزون: output += expert_out * weight
            output = (output + expert_out.broadcast_mul(&expert_weight)?)?;
        }

        // إعادة التشكيل للشكل الأصلي
        output.reshape((b, s, h))
    }
}