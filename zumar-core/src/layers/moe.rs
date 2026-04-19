use candle_core::{Tensor, Result, Device};
use candle_nn::Module; // تم إضافة هذا السطر لحل خطأ E0599
use crate::layers::bitlinear::ZumarBitLinear;

pub struct ZumarMoE {
    pub gate: ZumarBitLinear,
    pub experts: Vec<ZumarBitLinear>,
    pub _k: usize,
}

impl ZumarMoE {
    pub fn new(in_dim: usize, num_experts: usize, k: usize, device: &Device) -> Result<Self> {
        let gate = ZumarBitLinear::new(in_dim, num_experts, device)?;
        let mut experts = Vec::new();
        for _ in 0..num_experts {
            experts.push(ZumarBitLinear::new(in_dim, in_dim, device)?);
        }

        Ok(Self { 
            gate, 
            experts, 
            _k: k 
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // الآن سيتعرف المترجم على دالة forward لأن Module أصبح في النطاق
        let _logits = self.gate.forward(x)?;
        
        // استخدام أول خبير كمسار استدلالي أساسي
        let output = self.experts[0].forward(x)?;
        
        Ok(output)
    }
}
