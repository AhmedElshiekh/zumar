pub struct ZumarHybridBlock {
    mamba_layer: MambaLayer,       // معالجة السياق الخطي (SSM)
    moe_layer: SparseMoE,          // معالجة المعرفة الضخمة (Experts)
    router: SovereignRouter,       // الموجه الذكي
    norm: candle_nn::LayerNorm,
}

impl ZumarHybridBlock {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. مسار Mamba: لفهم السياق الطويل جداً (1-bit Optimized)
        let x_mamba = self.mamba_layer.forward(&self.norm.forward(x)?)?;
        
        // 2. مسار MoE: استرجاع المعلومات من الخبراء (Top-k Routing)
        let (weights, indices) = self.router.route(&x_mamba)?;
        let x_experts = self.moe_layer.forward_selective(&x_mamba, &weights, &indices)?;
        
        // 3. دمج المخرجات (Residual Connection)
        x_experts.add(&x)
    }
}
