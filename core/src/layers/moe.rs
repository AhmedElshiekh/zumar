use candle_core::{Tensor, Result, Device};
use candle_nn::{Module, VarBuilder};
use crate::layers::bitlinear::ZumarBitLinear;
use crate::layers::PackedBlockRef;
use std::collections::HashMap;

pub struct ZumarMoE {
    pub gate: ZumarBitLinear,
    pub experts: Vec<ZumarBitLinear>,
    pub packed_experts: Vec<PackedBlockRef>,
    pub cached_experts: HashMap<usize, ZumarBitLinear>,
    pub num_experts: usize,
    pub top_k: usize,
    pub in_dim: usize,
    pub device: Device,
}

impl ZumarMoE {
    pub fn new(in_dim: usize, num_experts: usize, top_k: usize, vs: VarBuilder) -> Result<Self> {
        let gate = ZumarBitLinear::new(in_dim, num_experts, vs.pp("gate"))?;
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(ZumarBitLinear::new(in_dim, in_dim, vs.pp(format!("expert_{}", i)))?);
        }
        Ok(Self {
            gate, experts,
            packed_experts: Vec::new(),
            cached_experts: HashMap::new(),
            num_experts, top_k, in_dim,
            device: vs.device().clone(),
        })
    }

    fn get_expert(&mut self, idx: usize) -> Result<&ZumarBitLinear> {
        if !self.cached_experts.contains_key(&idx) {
            if idx < self.packed_experts.len() {
                let p = &self.packed_experts[idx];
                let expert = ZumarBitLinear::from_packed_block(
                    &p.data, p.scale, (self.in_dim, self.in_dim), &self.device,
                )?;
                self.cached_experts.insert(idx, expert);
            } else if idx < self.experts.len() {
                return Ok(&self.experts[idx]);
            }
        }
        self.cached_experts.get(&idx)
            .or_else(|| self.experts.get(idx))
            .ok_or_else(|| candle_core::Error::Msg(format!("Expert {} not found", idx)))
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let flat_dim = b * s;
        let x_flat = x.reshape((flat_dim, h))?;

        let router_logits = self.gate.forward(&x_flat)?;
        let routing_probs = candle_nn::ops::softmax(&router_logits, 1)?;

        let mut output = x_flat.zeros_like()?;
        
        for idx in 0..self.top_k.min(self.num_experts) {
            let expert = self.get_expert(idx)?;
            let expert_out = expert.forward(&x_flat)?;
            let w = routing_probs.narrow(1, idx, 1)?;
            output = (output + expert_out.broadcast_mul(&w)?)?;
        }

        output.reshape((b, s, h))
    }
}