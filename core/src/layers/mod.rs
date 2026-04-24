pub mod bitlinear;
pub mod moe;
pub mod mamba;
pub mod attention;

use crate::layers::bitlinear::ZumarBitLinear;
use crate::layers::moe::ZumarMoE;
use crate::layers::mamba::{ZumarMambaBlock, ZumarMambaConfig};
use crate::layers::attention::ZumarFlashAttention;
use candle_core::{Tensor, Result, Device};
use candle_nn::{Module, VarBuilder, Embedding, LayerNorm};

pub struct ZumarBlock {
    pub pre_norm: LayerNorm,
    pub mamba_layer: ZumarMambaBlock,
    pub attention: ZumarFlashAttention,
    pub moe_layer: ZumarMoE,
    pub post_norm: LayerNorm,
}

pub struct ZumarModel {
    pub embedding: Embedding,
    pub layers: Vec<ZumarBlock>,
    pub final_norm: LayerNorm,
    pub lm_head: ZumarBitLinear,
}

impl ZumarBlock {
    pub fn new(
        in_dim: usize, num_experts: usize, top_k: usize, n_heads: usize, vs: VarBuilder,
    ) -> Result<Self> {
        let pre_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("input_layernorm"))?;
        let mamba_cfg = ZumarMambaConfig { d_model: in_dim, d_state: 16, d_conv: 4, expand: 2 };
        let mamba_layer = ZumarMambaBlock::new(&mamba_cfg, vs.pp("self_attn"))?;
        let head_dim = in_dim / n_heads;
        let attention = ZumarFlashAttention::new(n_heads, head_dim);
        let moe_layer = ZumarMoE::new(in_dim, num_experts, top_k, vs.pp("mlp"))?;
        let post_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("post_attention_layernorm"))?;
        Ok(Self { pre_norm, mamba_layer, attention, moe_layer, post_norm })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let normed = self.pre_norm.forward(x)?;
        let mamba_out = self.mamba_layer.forward(&normed)?;
        let x = (residual + mamba_out)?;
        
        let residual_2 = x.clone();
        let normed_2 = self.post_norm.forward(&x)?;
        let moe_out = self.moe_layer.forward(&normed_2)?;
        let x = (residual_2 + moe_out)?;
        
        Ok(x)
    }
}

impl ZumarModel {
    pub fn new(
        vocab_size: usize, in_dim: usize, num_layers: usize,
        num_experts: usize, top_k: usize, n_heads: usize, vs: VarBuilder,
    ) -> Result<Self> {
        let embedding = candle_nn::embedding(vocab_size, in_dim, vs.pp("model.embed_tokens"))?;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_vs = vs.pp(format!("model.layers.{}", i));
            layers.push(ZumarBlock::new(in_dim, num_experts, top_k, n_heads, layer_vs)?);
        }
        let final_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("model.norm"))?;
        let lm_head = ZumarBitLinear::new(in_dim, vocab_size, vs.pp("lm_head"))?;
        Ok(Self { embedding, layers, final_norm, lm_head })
    }

    pub fn embed(&self, token_id: u32, device: &Device) -> Result<Tensor> {
        let input_id = Tensor::new(&[token_id], device)?;
        let emb = self.embedding.forward(&input_id)?;
        emb.unsqueeze(0)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if x.rank() == 2 { x.unsqueeze(0)? } else { x.clone() };
        let mut hidden_states = x;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states)?;
        }
        hidden_states = self.final_norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states)
    }
}