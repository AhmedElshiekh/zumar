pub mod bitlinear;
pub mod moe;
pub mod mamba;

use crate::layers::bitlinear::ZumarBitLinear;
use crate::layers::moe::ZumarMoE;
use crate::layers::mamba::{ZumarMambaBlock, ZumarMambaConfig};
use candle_core::{Tensor, Result, Device};
use candle_nn::{Module, VarBuilder, Embedding, LayerNorm};

pub struct ZumarBlock {
    pub embedding: Embedding,
    pub pre_norm: LayerNorm,
    pub mamba_layer: ZumarMambaBlock, 
    pub moe_layer: ZumarMoE,          
    pub post_norm: LayerNorm,
    pub lm_head: ZumarBitLinear, 
}

impl ZumarBlock {
    pub fn new(in_dim: usize, vocab_size: usize, vs: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(vocab_size, in_dim, vs.pp("model.embed_tokens"))?;
        let pre_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("model.norm"))?;
        
        let mamba_cfg = ZumarMambaConfig {
            d_model: in_dim,
            d_state: 16,
            d_conv: 4,
            expand: 2,
        };
        
        // استخدام طبقة واحدة كمثال (أو يمكن تعديلها لتكرار الطبقات)
        let mamba_layer = ZumarMambaBlock::new(&mamba_cfg, vs.pp("model.layers.0.self_attn"))?;
        let moe_layer = ZumarMoE::new(in_dim,8, 2, vs.pp("model.layers.0.mlp"))?;
        let post_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("model.layers.0.post_attention_layernorm"))?;
        
        let lm_head = ZumarBitLinear::new(in_dim, vocab_size, vs.pp("model.embed_tokens"))?;

        Ok(Self {
            embedding,
            pre_norm,
            mamba_layer,
            moe_layer,
            post_norm,
            lm_head,
        })
    }

    pub fn embed(&self, token_id: u32, device: &Device) -> Result<Tensor> {
        let input_id = Tensor::new(&[token_id], device)?;
        self.embedding.forward(&input_id)?.unsqueeze(0)
    }

    /// المعالجة الجوهرية مع إضافة الـ Residual Connections
    pub fn forward_core(&self, x: &Tensor) -> Result<Tensor> {
        // 1. المسار الأول: Mamba
        let x_norm = self.pre_norm.forward(x)?;
        let mamba_out = self.mamba_layer.forward(&x_norm)?;
        let x = (mamba_out + x)?; // Residual 1

        // 2. المسار الثاني: MoE
        let x_norm_2 = self.post_norm.forward(&x)?;
        let moe_out = self.moe_layer.forward(&x_norm_2)?;
        let x = (moe_out + x)?; // Residual 2

        Ok(x)
    }

    pub fn project_head(&self, x: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(x)
    }
}
