use candle_core::{Tensor, Result, Device};
use candle_nn::{Module, VarBuilder};
use crate::layers::bitlinear::ZumarBitLinear;
use crate::layers::PackedBlockRef;
use std::collections::HashMap;

pub struct ZumarMoE {
    pub gate:           ZumarBitLinear,
    pub experts:        Vec<ZumarBitLinear>,
    pub packed_experts: Vec<PackedBlockRef>,
    pub cached_experts: HashMap<usize, ZumarBitLinear>,
    pub num_experts:    usize,
    pub top_k:          usize,
    pub in_dim:         usize,
    pub device:         Device,
}

impl ZumarMoE {
    pub fn new(
        in_dim: usize, num_experts: usize, top_k: usize, vs: VarBuilder,
    ) -> Result<Self> {
        let gate = ZumarBitLinear::new(in_dim, num_experts, vs.pp("gate"))?;
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(ZumarBitLinear::new(
                in_dim, in_dim, vs.pp(format!("expert_{}", i)),
            )?);
        }
        Ok(Self {
            gate, experts,
            packed_experts: Vec::new(),
            cached_experts: HashMap::new(),
            num_experts, top_k, in_dim,
            device: vs.device().clone(),
        })
    }

    /// تحميل خبير بالكسل (lazy) — من packed أو من experts المحمّلة مسبقاً
    fn get_expert(&mut self, idx: usize) -> Result<&ZumarBitLinear> {
        if !self.cached_experts.contains_key(&idx) {
            if idx < self.packed_experts.len() {
                let p      = &self.packed_experts[idx];
                let expert = ZumarBitLinear::from_packed_block(
                    &p.data, p.scale, (self.in_dim, self.in_dim), &self.device,
                )?;
                self.cached_experts.insert(idx, expert);
            } else if idx < self.experts.len() {
                // الخبير محمّل مسبقاً — لا داعي للـ cache
                return Ok(&self.experts[idx]);
            }
        }
        self.cached_experts.get(&idx)
            .or_else(|| self.experts.get(idx))
            .ok_or_else(|| candle_core::Error::Msg(format!("Expert {} not found", idx)))
    }

    /// ✅ استخراج top-k indices من tensor بدون topk مدمج في candle
    fn topk_indices(probs: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        // ترتيب تنازلي
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(k).map(|(i, _)| *i).collect()
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let flat_dim  = b * s;
        let x_flat    = x.reshape((flat_dim, h))?;

        // ── حساب احتمالات التوجيه ───────────────────────────────
        let router_logits = self.gate.forward(&x_flat)?;           // [flat, num_experts]
        let routing_probs = candle_nn::ops::softmax(&router_logits, 1)?;

        // ── التجميع عبر كل token ────────────────────────────────
        let mut output = x_flat.zeros_like()?;

        // نُحوّل routing_probs إلى Vec لاستخراج الـ top-k
        let probs_vec = routing_probs.to_vec2::<f32>()?;  // [flat, num_experts]

        for token_idx in 0..flat_dim {
            let token_probs = &probs_vec[token_idx];

            // ✅ إصلاح: استخراج top-k indices الحقيقية من الـ gate
            let top_indices = Self::topk_indices(token_probs, self.top_k.min(self.num_experts));

            // استخراج token واحد [1, h]
            let token = x_flat.narrow(0, token_idx, 1)?;
            let mut token_out = token.zeros_like()?;

            for &expert_idx in &top_indices {
                let weight = token_probs[expert_idx];
                if weight < 1e-6 { continue; }  // تخطي الخبراء ذوي الوزن الضئيل

                let expert     = self.get_expert(expert_idx)?;
                let expert_out = expert.forward(&token)?;  // [1, h]

                // تطبيق وزن الخبير
                let w_tensor   = Tensor::new(&[weight], &self.device)?
                    .reshape((1, 1))?;
                token_out = (token_out + expert_out.broadcast_mul(&w_tensor)?)?;
            }

            // ✅ تطبيع الوزن: قسمة على مجموع أوزان الخبراء المختارين
            let weight_sum: f32 = top_indices.iter()
                .map(|&i| token_probs[i])
                .sum();

            if weight_sum > 1e-6 {
                let norm = Tensor::new(&[weight_sum], &self.device)?
                    .reshape((1, 1))?;
                token_out = token_out.broadcast_div(&norm)?;
            }

            // وضع نتيجة الـ token في المخرجات
            let mask = {
                let mut m = vec![0.0f32; flat_dim];
                m[token_idx] = 1.0;
                Tensor::new(m.as_slice(), &self.device)?.reshape((flat_dim, 1))?
            };
            output = (output + token_out.expand((flat_dim, h))?.broadcast_mul(&mask)?)?;
        }

        output.reshape((b, s, h))
    }

    /// forward_selective: للاستخدام مع ZumarHybridBlock
    pub fn forward_selective(
        &mut self,
        x:       &Tensor,
        weights: &Tensor,
        indices: &Tensor,
    ) -> Result<Tensor> {
        // نفس منطق forward لكن الـ indices تأتي من الخارج
        let (b, s, h) = x.dims3()?;
        let flat_dim  = b * s;
        let x_flat    = x.reshape((flat_dim, h))?;
        let mut output = x_flat.zeros_like()?;

        let indices_vec = indices.to_vec2::<u32>()?;  // [flat, top_k]
        let weights_vec = weights.to_vec2::<f32>()?;  // [flat, top_k]

        for token_idx in 0..flat_dim {
            let token = x_flat.narrow(0, token_idx, 1)?;
            let mut token_out  = token.zeros_like()?;
            let mut weight_sum = 0.0f32;

            for k in 0..self.top_k.min(indices_vec[token_idx].len()) {
                let expert_idx = indices_vec[token_idx][k] as usize;
                let weight     = weights_vec[token_idx][k];
                if weight < 1e-6 { continue; }

                let expert     = self.get_expert(expert_idx)?;
                let expert_out = expert.forward(&token)?;
                let w_tensor   = Tensor::new(&[weight], &self.device)?.reshape((1, 1))?;
                token_out  = (token_out + expert_out.broadcast_mul(&w_tensor)?)?;
                weight_sum += weight;
            }

            if weight_sum > 1e-6 {
                let norm = Tensor::new(&[weight_sum], &self.device)?.reshape((1, 1))?;
                token_out = token_out.broadcast_div(&norm)?;
            }

            let mask = {
                let mut m = vec![0.0f32; flat_dim];
                m[token_idx] = 1.0;
                Tensor::new(m.as_slice(), &self.device)?.reshape((flat_dim, 1))?
            };
            output = (output + token_out.expand((flat_dim, h))?.broadcast_mul(&mask)?)?;
        }

        output.reshape((b, s, h))
    }
}
