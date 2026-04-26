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
    pub fn new(in_dim: usize, num_experts: usize, top_k: usize, vs: VarBuilder) -> Result<Self> {
        let gate = ZumarBitLinear::new(in_dim, num_experts, vs.pp("gate"))?;
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(ZumarBitLinear::new(in_dim, in_dim, vs.pp(format!("expert_{}", i)))?);
        }
        Ok(Self { gate, experts, num_experts, top_k })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let flat_dim = b * s;
        let x_flat = x.reshape((flat_dim, h))?;

        let router_logits = self.gate.forward(&x_flat)?;
        let routing_probs = candle_nn::ops::softmax(&router_logits, 1)?;

        let mut output = x_flat.zeros_like()?;
        
        for expert_idx in 0..self.num_experts {
            let expert_out = self.experts[expert_idx].forward(&x_flat)?;
            let expert_weight = routing_probs.narrow(1, expert_idx, 1)?;
            output = (output + expert_out.broadcast_mul(&expert_weight)?)?;
        }

        output.reshape((b, s, h))
    }
}