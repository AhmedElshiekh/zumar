// تعريف الموديلات الفرعية (Sub-modules) لضمان تنظيم الكود
pub mod bitlinear;
pub mod moe;

use crate::layers::bitlinear::ZumarBitLinear;
use crate::layers::moe::ZumarMoE;
use candle_core::{Tensor, Result, Device};
use candle_nn::Module; // ضروري لتعريف الـ Trait بشكل موحد

/// ZumarBlock: الوحدة الهيكلية الأساسية في مشروع زُمَر.
/// تجمع بين كفاءة الـ 1-bit وسرعة الـ Sparse MoE.
pub struct ZumarBlock {
    pub pre_norm: ZumarBitLinear,
    pub moe_layer: ZumarMoE,
    pub post_norm: ZumarBitLinear,
}

impl ZumarBlock {
    /// إنشاء بلوك جديد مع تحديد الأبعاد والعتاد المستخدم.
    /// يتم ضبط عدد الخبراء (Experts) افتراضياً على 8 (يُفعل منها 2 فقط).
    pub fn new(in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        // طبقة التثبيت الأولي (Identity/Norm substitute in 1-bit logic)
        let pre_norm = ZumarBitLinear::new(in_dim, in_dim, device)?;
        
        // طبقة الخبراء المتشعبة (Sparse Mixture of Experts)
        // عدد الخبراء: 8، عدد المفعلين (k): 2
        let moe_layer = ZumarMoE::new(in_dim, 8, 2, device)?;
        
        // طبقة التثبيت النهائي وتغيير الأبعاد إذا لزم الأمر
        let post_norm = ZumarBitLinear::new(in_dim, out_dim, device)?;

        Ok(Self {
            pre_norm,
            moe_layer,
            post_norm,
        })
    }
}

/// تنفيذ الـ Module Trait للسماح باستخدام البلوك داخل أي نموذج Candle
impl Module for ZumarBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. معالجة المدخلات عبر الطبقة الخطية الأولى (Pre-processing)
        let x = self.pre_norm.forward(x)?;

        // 2. التمرير عبر "موجه الخبراء" (The Gating Mechanism)
        // هنا يتم توفير الطاقة الحسابية عبر تفعيل 25% فقط من الشبكة
        let x = self.moe_layer.forward(&x)?;

        // 3. إنتاج المخرجات النهائية (Post-processing)
        self.post_norm.forward(&x)
    }
}
