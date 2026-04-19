// تعريف الموديلات الفرعية (Sub-modules) لضمان تنظيم الكود
pub mod bitlinear;
pub mod moe;
pub mod mamba;

use crate::layers::bitlinear::ZumarBitLinear;
use crate::layers::moe::ZumarMoE;
use crate::layers::mamba::{ZumarMambaBlock, ZumarMambaConfig}; // استيراد المكونات الجديدة
use candle_core::{Tensor, Result, Device, DType};
use candle_nn::{Module, VarBuilder};

/// ZumarBlock: الوحدة الهيكلية الأساسية للهجين (Hybrid) في زُمَر.
/// تجمع بين معالجة السياق (Mamba SSM) وكفاءة الخبراء (Sparse MoE).
pub struct ZumarBlock {
    pub pre_norm: ZumarBitLinear,
    pub mamba_layer: ZumarMambaBlock, // طبقة معالجة السياق الطويل
    pub moe_layer: ZumarMoE,          // طبقة الذكاء المتشعب
    pub post_norm: ZumarBitLinear,
}

impl ZumarBlock {
    /// إنشاء بلوك جديد مع دمج طبقة Mamba و MoE.
    pub fn new(in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        // 1. طبقة التثبيت الأولي
        let pre_norm = ZumarBitLinear::new(in_dim, in_dim, device)?;
        
        // 2. إعداد طبقة Mamba (Task 1.2)
        // نستخدم VarBuilder لإنشاء مصفوفات الحالة (A, D, etc.)
        let vs_mamba = VarBuilder::zeros(DType::F32, device);
        let mamba_cfg = ZumarMambaConfig {
            d_model: in_dim,
            d_state: 16,
            d_conv: 4,
            expand: 2,
        };
        let mamba_layer = ZumarMambaBlock::new(&mamba_cfg, vs_mamba.pp("mamba"))?;

        // 3. طبقة الخبراء المتشعبة (Sparse Mixture of Experts)
        let moe_layer = ZumarMoE::new(in_dim, 8, 2, device)?;
        
        // 4. طبقة التثبيت النهائي وتغيير الأبعاد
        let post_norm = ZumarBitLinear::new(in_dim, out_dim, device)?;

        Ok(Self {
            pre_norm,
            mamba_layer,
            moe_layer,
            post_norm,
        })
    }
}

impl Module for ZumarBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Pre-processing
        let x = self.pre_norm.forward(x)?;

        // 2. Mamba Selective Scan (معالجة تسلسل البيانات والسياق)
        // هنا يكمن سر التفوق في التعامل مع النصوص الطويلة
        let x = self.mamba_layer.forward(&x)?;

        // 3. Sparse MoE (تفعيل الخبراء المطلوبين فقط)
        let x = self.moe_layer.forward(&x)?;

        // 4. Post-processing
        self.post_norm.forward(&x)
    }
}
