// ================================================================
// zumar_block.rs — ZumarHybridBlock
//
// التغييرات:
//   - استبدال MambaLayer/SparseMoE/SovereignRouter غير المعرّفة
//     بالأنواع الحقيقية الموجودة في المشروع
//   - توحيد مع ZumarBlock الموجود في layers/mod.rs
//   - إضافة new() لبناء الـ block
// ================================================================

use candle_core::{Tensor, Result};
use candle_nn::{LayerNorm, VarBuilder, Module};
use crate::layers::mamba::ZumarMambaBlock;
use crate::layers::moe::ZumarMoE;
use crate::layers::moe_router::SovereignRouter;

/// Hybrid Block: Mamba (SSM) + MoE + SovereignRouter
///
/// التدفق:
///   x → LayerNorm → Mamba → SovereignRouter → MoE → + x (residual)
pub struct ZumarHybridBlock {
    norm:         LayerNorm,
    mamba:        ZumarMambaBlock,
    router:       SovereignRouter,
    moe:          ZumarMoE,
}

impl ZumarHybridBlock {
    pub fn new(
        d_model:     usize,
        d_state:     usize,
        d_conv:      usize,
        expand:      usize,
        num_experts: usize,
        top_k:       usize,
        vs:          VarBuilder,
    ) -> Result<Self> {
        use crate::layers::mamba::ZumarMambaConfig;

        let norm  = candle_nn::layer_norm(d_model, 1e-5, vs.pp("norm"))?;
        let cfg   = ZumarMambaConfig { d_model, d_state, d_conv, expand };
        let mamba = ZumarMambaBlock::new(&cfg, vs.pp("mamba"))?;
        let router = SovereignRouter::new(d_model, num_experts, top_k, vs.pp("router"))?;
        let moe   = ZumarMoE::new(d_model, num_experts, top_k, vs.pp("moe"))?;

        Ok(Self { norm, mamba, router, moe })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        // 1. تطبيع المدخل
        let x_normed = self.norm.forward(x)?;

        // 2. مسار Mamba: SSM لفهم السياق الطويل
        let x_mamba = self.mamba.forward(&x_normed)?;

        // 3. توجيه عبر SovereignRouter
        let (weights, indices) = self.router.route(&x_mamba)?;

        // 4. معالجة الخبراء
        let x_experts = self.moe.forward_selective(&x_mamba, &weights, &indices)?;

        // 5. Residual Connection
        x_experts + x
    }
}
