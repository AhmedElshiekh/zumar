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
        // 1. تحميل الـ Embedding (تحويل Token ID إلى متجه 1024)
        let embedding = candle_nn::embedding(vocab_size, in_dim, vs.pp("model.embed_tokens"))?;

        // 2. تحميل الـ Norm الرئيسي
        let pre_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("model.norm"))?;
        
        let mamba_cfg = ZumarMambaConfig {
            d_model: in_dim,
            d_state: 16,
            d_conv: 4,
            expand: 2,
        };
        
        // 3. تحميل الطبقات الهجينة (Mamba + MoE)
        let mamba_layer = ZumarMambaBlock::new(&mamba_cfg, vs.pp("model.layers.0.self_attn"))?;
        let moe_layer = ZumarMoE::new(in_dim, 8, 2, vs.pp("model.layers.0.mlp"))?;
        
        // 4. تحميل الـ Norm النهائي للطبقة الأخيرة
        let post_norm = candle_nn::layer_norm(in_dim, 1e-5, vs.pp("model.layers.29.post_attention_layernorm"))?;
        
        // 5. الرأس النهائي: يحول الـ 1024 إلى احتمالات لـ 50257 كلمة
        // ملاحظة: نستخدم أوزان embed_tokens لتقليل حجم النموذج (Weight Tying)
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

    /// يحول التوكن إلى Embedding أبعاده [1, 1, 1024]
    pub fn embed(&self, token_id: u32, device: &Device) -> Result<Tensor> {
        let input_id = Tensor::new(&[token_id], device)?;
        self.embedding.forward(&input_id)?.unsqueeze(0)
    }

    /// يعالج المتجه داخل الطبقات المخفية (يبقى البعد 1024)
    pub fn forward_core(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pre_norm.forward(x)?;
        let x = self.mamba_layer.forward(&x)?;
        self.moe_layer.forward(&x)
    }

    /// يحول المتجه من 1024 إلى 50257 (حجم القاموس)
    pub fn project_head(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.post_norm.forward(x)?;
        self.lm_head.forward(&x)
    }
}

impl Module for ZumarBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // نمرر البيانات عبر القلب ثم الرأس بالترتيب
        // تأكد من عدم وجود عملية جمع (+) هنا بين x القديمة والجديدة
        let x_hidden = self.forward_core(x)?;
        self.project_head(&x_hidden)
    }
}
