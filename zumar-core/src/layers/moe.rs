use candle_core::{Tensor, Result};
use candle_nn::{Module, VarBuilder}; 
use crate::layers::bitlinear::ZumarBitLinear;

pub struct ZumarMoE {
    pub gate: ZumarBitLinear,
    pub experts: Vec<ZumarBitLinear>,
    pub _k: usize, 
}

impl ZumarMoE {
    pub fn new(in_dim: usize, num_experts: usize, k: usize, vs: VarBuilder) -> Result<Self> {
        let gate = ZumarBitLinear::new(in_dim, num_experts, vs.pp("gate"))?;
        
        let mut experts = Vec::new();
        for i in 0..num_experts {
            experts.push(ZumarBitLinear::new(in_dim, in_dim, vs.pp(format!("expert_{}", i)))?);
        }

        Ok(Self { 
            gate, 
            experts, 
            _k: k 
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _logits = self.gate.forward(x)?;
        // حالياً نمرر البيانات عبر الخبير الأول
        self.experts[0].forward(x)
    }
}
