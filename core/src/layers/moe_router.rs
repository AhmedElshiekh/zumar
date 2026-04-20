use candle_core::{Tensor, Result, DType};
use candle_nn::{Linear, Module};

pub struct SovereignRouter {
    gate: Linear,
    num_experts: usize,
    top_k: usize,
}

impl SovereignRouter {
    pub fn new(dim: usize, num_experts: usize, top_k: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        let gate = candle_nn::linear(dim, num_experts, vb.pp("gate"))?;
        Ok(Self { gate, num_experts, top_k })
    }

    pub fn route(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // حساب درجات الأهمية لكل خبير
        let logits = self.gate.forward(x)?;
        
        // اختيار أفضل K خبراء (Top-k)
        let (weights, indices) = logits.topk(self.top_k, candle_core::D::Minus1, true)?;
        
        // تطبيق Softmax لتطبيع الأوزان
        let weights = candle_nn::ops::softmax(&weights, candle_core::D::Minus1)?;
        
        Ok((weights, indices))
    }
}
