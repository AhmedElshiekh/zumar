// تعريف الموديلات الفرعية لضمان تنظيم الكود
pub mod bitlinear;
pub mod moe;
pub mod mamba;

use crate::layers::bitlinear::ZumarBitLinear;
use crate::layers::moe::ZumarMoE;
use crate::layers::mamba::{ZumarMambaBlock, ZumarMambaConfig};
use candle_core::{Tensor, Result};
use candle_nn::{Module, VarBuilder};

/// ZumarBlock: الوحدة الهيكلية الأساسية للهجين (Hybrid) في زُمَر.
pub struct ZumarBlock {
    pub pre_norm: ZumarBitLinear,
    pub mamba_layer: ZumarMambaBlock, 
    pub moe_layer: ZumarMoE,          
    pub post_norm: ZumarBitLinear,
}

impl ZumarBlock {
    /// إنشاء بلوك جديد وتوزيع الأوزان عبر VarBuilder.
    pub fn new(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Self> {
        // 1. طبقة التثبيت الأولي (أوزان مخصصة للـ pre_norm)
        let pre_norm = ZumarBitLinear::new(in_dim, in_dim, vs.pp("pre_norm"))?;
        
        // 2. إعداد طبقة Mamba
        let mamba_cfg = ZumarMambaConfig {
            d_model: in_dim,
            d_state: 16,
            d_conv: 4,
            expand: 2,
        };
        // نمرر vs.pp("mamba") لتأخذ أوزانها المستقلة
        let mamba_layer = ZumarMambaBlock::new(&mamba_cfg, vs.pp("mamba"))?;

        // 3. طبقة الخبراء المتشعبة (Sparse Mixture of Experts)
        // نمرر vs.pp("moe") لتوزيع الأوزان على الخبراء داخلياً
        let moe_layer = ZumarMoE::new(in_dim, 8, 2, vs.pp("moe"))?;
        
        // 4. طبقة التثبيت النهائي
        let post_norm = ZumarBitLinear::new(in_dim, out_dim, vs.pp("post_norm"))?;

        Ok(Self {
            pre_norm,
            mamba_layer,
            moe_layer,
            post_norm,
        })
    }
}

impl Module for ZumarBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Pre-processing
        let x = self.pre_norm.forward(x)?;

        // 2. Mamba Selective Scan
        let x = self.mamba_layer.forward(&x)?;

        // 3. Sparse MoE
        let x = self.moe_layer.forward(&x)?;

        // 4. Post-processing
        self.post_norm.forward(&x)
    }
}
