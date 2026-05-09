use candle_core::{Tensor, Result, Device, DType};
use candle_nn::{Module, VarBuilder};

/// LoRA Layer — تكيّف منخفض الرتبة
pub struct LoRALinear {
    /// الوزن الأصلي (مُجمّد)
    weight: Tensor,
    /// bias أصلي (مُجمّد)
    bias: Option<Tensor>,
    /// مصفوفة A (r × in_dim)
    lora_a: Tensor,
    /// مصفوفة B (out_dim × r)
    lora_b: Tensor,
    /// معامل القياس α/r
    scaling: f64,
    /// هل LoRA مفعّل؟
    active: bool,
}

impl LoRALinear {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        rank: usize,
        alpha: f64,
        vs: VarBuilder,
    ) -> Result<Self> {
        let in_dim = weight.dim(1)?;
        let out_dim = weight.dim(0)?;
        let device = vs.device();

        // A: (r, in_dim) - تهيئة عشوائية
        let lora_a = vs.get_with_hints(
            (rank, in_dim),
            "lora_A",
            candle_nn::Init::Randn { mean: 0.0, stdev: 1.0 / (in_dim as f64).sqrt() },
        )?;

        // B: (out_dim, r) - تهيئة صفرية (لتبدأ LoRA من الصفر)
        let lora_b = vs.get_with_hints(
            (out_dim, rank),
            "lora_B",
            candle_nn::Init::Const(0.0),
        )?;

        let scaling = alpha / rank as f64;

        Ok(Self {
            weight,
            bias,
            lora_a,
            lora_b,
            scaling,
            active: true,
        })
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// دمج LoRA في الوزن الأصلي (للاستدلال)
    pub fn merge(&self) -> Result<Tensor> {
        // W' = W + scaling * (B @ A)
        let delta = self.lora_b.matmul(&self.lora_a)?;  // (out, r) @ (r, in) → (out, in)
        Ok((&self.weight + (delta * self.scaling)?)?)
    }
}

impl Module for LoRALinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // الأصل: x @ W^T
        let mut out = x.matmul(&self.weight.t()?)?;

        // إضافة LoRA إن كان مفعّلاً
        if self.active {
            // x @ A^T @ B^T * scaling
            let lora_out = x
                .matmul(&self.lora_a.t()?)?  // (..., in) @ (in, r) → (..., r)
                .matmul(&self.lora_b.t()?)?  // (..., r) @ (r, out) → (..., out)
                * self.scaling;
            out = (out + lora_out)?;
        }

        if let Some(bias) = &self.bias {
            out = out.broadcast_add(bias)?;
        }

        Ok(out)
    }
}

/// مُجمّد LoRA: يحوّل طبقة ZumarBitLinear إلى LoRA
pub fn apply_lora_to_bitlinear(
    bitlinear: &crate::layers::bitlinear::ZumarBitLinear,
    rank: usize,
    alpha: f64,
    vs: VarBuilder,
) -> Result<LoRALinear> {
    LoRALinear::new(
        bitlinear.latent_weight.clone(),
        bitlinear.bias.clone(),
        rank,
        alpha,
        vs,
    )
}

/// تطبيق LoRA على طبقة مكممة (QLoRA)
pub fn apply_qlora_to_bitlinear(
    bitlinear: &crate::layers::bitlinear::ZumarBitLinear,
    rank: usize,
    alpha: f64,
    vs: VarBuilder,
) -> Result<LoRALinear> {
    // فك التكميم أولاً للحصول على الوزن الأصلي
    let weight = if bitlinear.quantize {
        bitlinear.dequantize_from_nf4()?
    } else {
        bitlinear.latent_weight.clone()
    };
    
    LoRALinear::new(
        weight,
        bitlinear.bias.clone(),
        rank,
        alpha,
        vs,
    )
}