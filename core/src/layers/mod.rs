pub mod bitlinear;
pub mod moe;
pub mod mamba;

use crate::layers::bitlinear::ZumarBitLinear;
use crate::layers::moe::ZumarMoE;
use crate::layers::mamba::{ZumarMambaBlock, ZumarMambaConfig};
use candle_core::{Tensor, Result};
use candle_nn::{Module, VarBuilder};

pub struct ZumarBlock {
    pub pre_norm: ZumarBitLinear,
    pub mamba_layer: ZumarMambaBlock, 
    pub moe_layer: ZumarMoE,          
    pub post_norm: ZumarBitLinear,
    pub lm_head: ZumarBitLinear, // الطبقة المسؤولة عن احتمالات الكلمات
}

impl ZumarBlock {
    pub fn new(in_dim: usize, vocab_size: usize, vs: VarBuilder) -> Result<Self> {
        let pre_norm = ZumarBitLinear::new(in_dim, in_dim, vs.pp("pre_norm"))?;
        
        let mamba_cfg = ZumarMambaConfig {
            d_model: in_dim,
            d_state: 16,
            d_conv: 4,
            expand: 2,
        };
        let mamba_layer = ZumarMambaBlock::new(&mamba_cfg, vs.pp("mamba"))?;
        let moe_layer = ZumarMoE::new(in_dim, 8, 2, vs.pp("moe"))?;
        
        // تحويل من d_model إلى d_model (التثبيت النهائي)
        let post_norm = ZumarBitLinear::new(in_dim, in_dim, vs.pp("post_norm"))?;
        
        // تحويل من d_model (768) إلى حجم القاموس (Vocab Size)
        let lm_head = ZumarBitLinear::new(in_dim, vocab_size, vs.pp("lm_head"))?;

        Ok(Self {
            pre_norm,
            mamba_layer,
            moe_layer,
            post_norm,
            lm_head,
        })
    }
}

impl Module for ZumarBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pre_norm.forward(x)?;
        let x = self.mamba_layer.forward(&x)?;
        let x = self.moe_layer.forward(&x)?;
        let x = self.post_norm.forward(&x)?;
        
        // المخرج هنا سيكون [Batch, Seq, Vocab_Size]
        self.lm_head.forward(&x)
    }
}
