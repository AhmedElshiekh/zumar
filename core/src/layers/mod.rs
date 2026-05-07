pub mod bitlinear;
pub mod moe;
pub mod mamba;
pub mod attention;
pub mod snn;
pub mod vocab_aligner;
pub mod ewc;

use crate::layers::bitlinear::ZumarBitLinear;
use crate::layers::moe::ZumarMoE;
use crate::layers::attention::ZumarFlashAttention;
use candle_core::{Tensor, Result, Device, DType};
use candle_nn::{Module, VarBuilder, Embedding, LayerNorm};
use std::collections::HashMap;

pub struct ZumarBlock {
    pub pre_norm:  LayerNorm,
    pub q_proj:    ZumarBitLinear,
    pub k_proj:    ZumarBitLinear,
    pub v_proj:    ZumarBitLinear,
    pub o_proj:    ZumarBitLinear,
    pub attention: ZumarFlashAttention,
    pub moe:       ZumarMoE,
    pub post_norm: LayerNorm,
}

pub struct ZumarModel {
    pub embedding:  Embedding,
    pub layers:     Vec<ZumarBlock>,
    pub final_norm: LayerNorm,
    pub lm_head:    ZumarBitLinear,
    pub hidden_size: usize,
    pub vocab_size:  usize,
}

#[derive(Clone)]
pub struct PackedBlockRef {
    pub data:  Vec<u8>,
    pub scale: f32,
}

impl PackedBlockRef {
    pub fn to_bitlinear(&self, shape: (usize, usize), device: &Device) -> Result<ZumarBitLinear> {
        ZumarBitLinear::from_packed_block(&self.data, self.scale, shape, device)
    }
}

impl ZumarBlock {
    pub fn new(
        in_dim: usize, num_experts: usize, top_k: usize,
        n_heads: usize, vs: VarBuilder,
    ) -> Result<Self> {
        let head_dim = in_dim / n_heads;
        let kv_dim   = head_dim;

        let pre_norm  = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("input_layernorm"))?;
        let q_proj    = ZumarBitLinear::new(in_dim, in_dim,   vs.pp("self_attn.q_proj"))?;
        let k_proj    = ZumarBitLinear::new(in_dim, kv_dim,   vs.pp("self_attn.k_proj"))?;
        let v_proj    = ZumarBitLinear::new(in_dim, kv_dim,   vs.pp("self_attn.v_proj"))?;
        let o_proj    = ZumarBitLinear::new(in_dim, in_dim,   vs.pp("self_attn.o_proj"))?;
        let attention = ZumarFlashAttention::new(n_heads, head_dim);
        let moe       = ZumarMoE::new(in_dim, num_experts, top_k, vs.pp("mlp"))?;
        let post_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("post_attention_layernorm"))?;

        Ok(Self { pre_norm, q_proj, k_proj, v_proj, o_proj, attention, moe, post_norm })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let normed   = self.pre_norm.forward(x)?;
        let q        = self.q_proj.forward(&normed)?;
        let k        = self.k_proj.forward(&normed)?;
        let v        = self.v_proj.forward(&normed)?;
        let attn_out = self.attention.forward(&q, &k, &v)?;
        let attn_out = self.o_proj.forward(&attn_out)?;
        let x        = (residual + attn_out)?;
        let residual_2 = x.clone();
        let normed_2   = self.post_norm.forward(&x)?;
        let moe_out    = self.moe.forward(&normed_2)?;
        residual_2 + moe_out
    }

    pub fn forward_checkpointed(&mut self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        let attn_out = {
            let normed = self.pre_norm.forward(x)?;
            let q      = self.q_proj.forward(&normed)?;
            let k      = self.k_proj.forward(&normed)?;
            let v      = self.v_proj.forward(&normed)?;
            let attn   = self.attention.forward(&q, &k, &v)?;
            let out    = self.o_proj.forward(&attn)?;
            drop(normed); drop(q); drop(k); drop(v); drop(attn);
            out
        };

        let x          = (residual + attn_out)?;
        let residual_2 = x.clone();
        let moe_out    = {
            let normed = self.post_norm.forward(&x)?;
            let out    = self.moe.forward(&normed)?;
            drop(normed);
            out
        };

        residual_2 + moe_out
    }

    pub fn from_packed_blocks(
        in_dim: usize, num_experts: usize, n_heads: usize,
        layer_blocks: &[PackedBlockRef], device: &Device,
    ) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vs     = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, device);
        let head_dim = in_dim / n_heads;

        let pre_norm  = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("input_layernorm"))?;
        let post_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("post_attention_layernorm"))?;
        let attention = ZumarFlashAttention::new(n_heads, head_dim);

        let q_proj = layer_blocks[0].to_bitlinear((in_dim,    in_dim),   device)?;
        let k_proj = layer_blocks[1].to_bitlinear((head_dim,  in_dim),   device)?;
        let v_proj = layer_blocks[2].to_bitlinear((head_dim,  in_dim),   device)?;
        let o_proj = layer_blocks[3].to_bitlinear((in_dim,    in_dim),   device)?;

        let gate           = layer_blocks[4].to_bitlinear((num_experts, in_dim), device)?;
        let packed_experts = layer_blocks[5..5 + num_experts].to_vec();

        let moe = ZumarMoE {
            gate,
            experts: Vec::new(),
            packed_experts,
            cached_experts: HashMap::new(),
            num_experts,
            top_k: 2,
            in_dim,
            device: device.clone(),
        };

        Ok(Self { pre_norm, q_proj, k_proj, v_proj, o_proj, attention, moe, post_norm })
    }
}

impl ZumarModel {
    pub fn new(
        vocab_size: usize, in_dim: usize, num_layers: usize,
        num_experts: usize, top_k: usize, n_heads: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        let embedding = candle_nn::embedding(vocab_size, in_dim, vs.pp("model.embed_tokens"))?;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(ZumarBlock::new(
                in_dim, num_experts, top_k, n_heads,
                vs.pp(format!("model.layers.{}", i)),
            )?);
        }
        let final_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("model.norm"))?;
        let lm_head    = ZumarBitLinear::new(in_dim, vocab_size, vs.pp("lm_head"))?;
        Ok(Self { embedding, layers, final_norm, lm_head, hidden_size: in_dim, vocab_size })
    }

    pub fn embed(&self, token_id: u32, device: &Device) -> Result<Tensor> {
        let input_id = Tensor::new(&[token_id], device)?;
        let emb      = self.embedding.forward(&input_id)?;
        emb.unsqueeze(0)
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let x = if x.rank() == 2 { x.unsqueeze(0)? } else { x.clone() };
        let mut h = x;
        for layer in &mut self.layers {
            h = layer.forward_checkpointed(&h)?;
        }
        h = self.final_norm.forward(&h)?;
        self.lm_head.forward(&h)
    }

    pub fn from_packed_blocks(
        vocab_size: usize, in_dim: usize, num_layers: usize,
        num_experts: usize, n_heads: usize,
        blocks: &[PackedBlockRef],
        embedding: Option<&PackedBlockRef>,
        device: &Device,
    ) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vs     = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, device);

        let embedding_layer = if let Some(emb) = embedding {
            load_packed_embedding(&emb.data, emb.scale, vocab_size, in_dim, device)?
        } else {
            candle_nn::embedding(vocab_size, in_dim, vs.pp("model.embed_tokens"))?
        };

        let final_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("model.norm"))?;
        let lm_head    = blocks.last()
            .ok_or_else(|| candle_core::Error::Msg("No blocks for lm_head".to_string()))?
            .to_bitlinear((vocab_size, in_dim), device)?;

        let blocks_per_layer = 4 + 1 + num_experts;
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let start = 1 + i * blocks_per_layer;
            let end   = start + blocks_per_layer;
            if end <= blocks.len() {
                layers.push(ZumarBlock::from_packed_blocks(
                    in_dim, num_experts, n_heads,
                    &blocks[start..end], device,
                )?);
            }
        }

        Ok(Self { embedding: embedding_layer, layers, final_norm, lm_head, hidden_size: in_dim, vocab_size })
    }
}

/// ✅ إصلاح: فك الضغط إلى F32 مباشرة (كان F16 → dtype mismatch مع بقية النموذج)
fn load_packed_embedding(
    data: &[u8],
    scale: f32,
    vocab_size: usize,
    hidden_size: usize,
    device: &Device,
) -> Result<Embedding> {
    // نفس DECODE_MAP الموحد من kernels/mod.rs
    // 0b00→0, 0b01→0, 0b10→+1, 0b11→-1
    const MAP: [f32; 4] = [0.0f32, 0.0f32, 1.0f32, -1.0f32];

    let total = vocab_size * hidden_size;
    let mut weights = Vec::with_capacity(total);

    for &byte in data {
        for bit in 0..4 {
            if weights.len() >= total { break; }
            let bits = (byte >> (bit * 2)) & 0b11;
            weights.push(MAP[bits as usize] * scale);
        }
    }
    weights.truncate(total);

    // ✅ F32 وليس F16 — يتوافق مع ZumarBitLinear وبقية الطبقات
    let tensor = Tensor::from_vec(weights, (vocab_size, hidden_size), device)?;
    Ok(Embedding::new(tensor, hidden_size))
}
