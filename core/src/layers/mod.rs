pub mod bitlinear;
pub mod moe;
pub mod mamba;
pub mod attention;
pub mod snn;

use crate::layers::bitlinear::ZumarBitLinear;
use crate::layers::moe::ZumarMoE;
use crate::layers::attention::ZumarFlashAttention;
use candle_core::{Tensor, Result, Device};
use candle_nn::{Module, VarBuilder, Embedding, LayerNorm};

pub struct ZumarBlock {
    pub pre_norm: LayerNorm,
    pub q_proj: ZumarBitLinear,
    pub k_proj: ZumarBitLinear,
    pub v_proj: ZumarBitLinear,
    pub o_proj: ZumarBitLinear,
    pub attention: ZumarFlashAttention,
    pub moe: ZumarMoE,
    pub post_norm: LayerNorm,
}

pub struct ZumarModel {
    pub embedding: Embedding,
    pub layers: Vec<ZumarBlock>,
    pub final_norm: LayerNorm,
    pub lm_head: ZumarBitLinear,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl ZumarBlock {
    pub fn new(
        in_dim: usize, num_experts: usize, top_k: usize, n_heads: usize, vs: VarBuilder,
    ) -> Result<Self> {
        let pre_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("input_layernorm"))?;
        let q_proj = ZumarBitLinear::new(in_dim, in_dim, vs.pp("self_attn.q_proj"))?;
        let k_proj = ZumarBitLinear::new(in_dim, in_dim, vs.pp("self_attn.k_proj"))?;
        let v_proj = ZumarBitLinear::new(in_dim, in_dim, vs.pp("self_attn.v_proj"))?;
        let o_proj = ZumarBitLinear::new(in_dim, in_dim, vs.pp("self_attn.o_proj"))?;
        let attention = ZumarFlashAttention::new(n_heads, in_dim / n_heads);
        let moe = ZumarMoE::new(in_dim, num_experts, top_k, vs.pp("mlp"))?;
        let post_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("post_attention_layernorm"))?;
        Ok(Self { pre_norm, q_proj, k_proj, v_proj, o_proj, attention, moe, post_norm })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let normed = self.pre_norm.forward(x)?;
        let q = self.q_proj.forward(&normed)?;
        let k = self.k_proj.forward(&normed)?;
        let v = self.v_proj.forward(&normed)?;
        let attn_out = self.attention.forward(&q, &k, &v)?;
        let attn_out = self.o_proj.forward(&attn_out)?;
        let x = (residual + attn_out)?;
        let residual_2 = x.clone();
        let normed_2 = self.post_norm.forward(&x)?;
        let moe_out = self.moe.forward(&normed_2)?;
        residual_2 + moe_out
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
        Ok(Self { embedding, layers, final_norm, lm_head, hidden_size: in_dim, vocab_size })
    }

    pub fn embed(&self, token_id: u32, device: &Device) -> Result<Tensor> {
        let input_id = Tensor::new(&[token_id], device)?;
        let emb = self.embedding.forward(&input_id)?;
        emb.unsqueeze(0)
    }

    pub fn embed_tokens(&self, token_ids: &[u32], device: &Device) -> Result<Tensor> {
        let input = Tensor::new(token_ids, device)?;
        let emb = self.embedding.forward(&input)?;
        emb.unsqueeze(0)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if x.rank() == 2 { x.unsqueeze(0)? } else { x.clone() };
        let mut hidden_states = x;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        hidden_states = self.final_norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states)
    }
}