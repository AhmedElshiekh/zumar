use candle_core::{Tensor, Result, DType};
use candle_nn::{Module, VarBuilder, Linear, Conv1d, Conv1dConfig};

fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.exp()? + 1.0f64)?.log()
}

pub struct ZumarMambaConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv:  usize,
    pub expand:  usize,
}

pub struct ZumarMambaBlock {
    pub in_proj:  Linear,
    pub conv1d:   Conv1d,       // ✅ Conv1d حقيقي بدلاً من حلقة O(L²)
    pub x_proj:   Linear,
    pub dt_proj:  Linear,
    pub out_proj: Linear,
    pub a_log:    Tensor,
    pub d:        Tensor,
    pub d_state:  usize,
    pub d_inner:  usize,
    pub d_conv:   usize,
}

impl ZumarMambaBlock {
    pub fn new(cfg: &ZumarMambaConfig, vs: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_model * cfg.expand;

        let in_proj  = candle_nn::linear(cfg.d_model, d_inner * 2, vs.pp("in_proj"))?;
        let x_proj   = candle_nn::linear(d_inner, cfg.d_state * 2 + d_inner, vs.pp("x_proj"))?;
        let dt_proj  = candle_nn::linear(d_inner, d_inner, vs.pp("dt_proj"))?;
        let out_proj = candle_nn::linear(d_inner, cfg.d_model, vs.pp("out_proj"))?;

        // ✅ Conv1d حقيقي من candle_nn
        // groups=d_inner لأن Mamba تستخدم depthwise convolution
        let conv_cfg = Conv1dConfig {
            padding: cfg.d_conv - 1,  // causal padding من اليسار
            groups:  d_inner,
            stride:  1,
            dilation: 1,
            cudnn_fwd_algo: None
        };
        let conv1d = candle_nn::conv1d(
            d_inner, d_inner, cfg.d_conv,
            conv_cfg,
            vs.pp("conv1d"),
        )?;

        let a_log = vs.get_with_hints(
            (cfg.d_state, d_inner), "a_log",
            candle_nn::Init::Const(0.0),
        )?;
        let d = vs.get_with_hints(
            d_inner, "d",
            candle_nn::Init::Const(1.0),
        )?;

        Ok(Self {
            in_proj, conv1d, x_proj, dt_proj, out_proj,
            a_log, d,
            d_state: cfg.d_state,
            d_inner,
            d_conv: cfg.d_conv,
        })
    }

    /// ✅ Conv1d صحيح باستخدام candle_nn
    /// المشكلة القديمة: حلقة مزدوجة O(L² × D) بطيئة جداً
    /// الحل: candle_nn::Conv1d تُنفّذ بعملية واحدة على CPU/GPU
    fn apply_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        // x: [b, seq, d_inner]
        // Conv1d في candle يتوقع [b, channels, seq]
        let (b, seq, d) = x.dims3()?;

        // [b, seq, d] → [b, d, seq]
        let x_t = x.transpose(1, 2)?;

        // تطبيق Conv1d — يتضمن الـ causal padding تلقائياً
        let out = self.conv1d.forward(&x_t)?;
        // out: [b, d, seq + padding] — نقطع الـ padding الزائد

        // قطع آخر (d_conv - 1) عمود لإزالة الـ non-causal padding
        let out_seq = out.dim(2)?;
        let trimmed = if out_seq > seq {
            out.narrow(2, 0, seq)?
        } else {
            out
        };

        // [b, d, seq] → [b, seq, d]
        trimmed.transpose(1, 2)
    }

    /// ✅ Selective Scan مع Causal Masking صحيح
    ///
    /// المشكلة القديمة:
    ///   b و c يُحسبان من كامل التسلسل مسبقاً —
    ///   يعني token[i] يرى b_t و c_t من token[i+1] (يرى المستقبل)
    ///
    /// الحل:
    ///   b_t و c_t تُستخرج في كل خطوة t من x_dbl[:, t, :]
    ///   وهي مشتقة من x_conv[:, t, :] فقط (بعد Conv1d السببية)
    ///   هذا يضمن أن كل token لا يرى إلا الماضي
    fn selective_scan(
        &self,
        x_conv: &Tensor,   // [b, seq, d_inner] — مخرج Conv1d
        delta:  &Tensor,   // [b, seq, d_inner]
        a:      &Tensor,   // [d_state, d_inner]
        d:      &Tensor,   // [d_inner]
    ) -> Result<Tensor> {
        let (batch, seqlen, _dim) = x_conv.dims3()?;
        let device = x_conv.device();

        // حالة SSM الابتدائية: صفر
        // h: [batch, d_state, d_inner]
        let mut h = Tensor::zeros(
            (batch, self.d_state, self.d_inner),
            DType::F32,
            device,
        )?;

        let mut outputs = Vec::with_capacity(seqlen);

        for t in 0..seqlen {
            // ✅ استخراج المُدخلات في الخطوة t فقط (causal)
            let x_t     = x_conv.narrow(1, t, 1)?.squeeze(1)?;  // [b, d_inner]
            let delta_t = delta.narrow(1, t, 1)?.squeeze(1)?;   // [b, d_inner]

            // x_proj على x_t الخاص بهذا الـ token
            let x_dbl_t = self.x_proj.forward(&x_t)?;           // [b, d_state*2 + d_inner]

            // ✅ b_t و c_t مشتقة من x_t فقط — لا رؤية للمستقبل
            let b_t = x_dbl_t.narrow(1, 0,              self.d_state)?; // [b, d_state]
            let c_t = x_dbl_t.narrow(1, self.d_state,   self.d_state)?; // [b, d_state]
            let u_t = x_dbl_t.narrow(1, self.d_state*2, self.d_inner)?; // [b, d_inner]

            // Discretization: delta_t → softplus لضمان القيم الموجبة
            // let delta_t = candle_nn::ops::softplus(&delta_t)?;           // [b, d_inner]
            let delta_t = softplus(&delta_t)?;

            // A_bar = exp(delta_t * a)
            // a:       [d_state, d_inner]
            // delta_t: [b, d_inner] → expand إلى [b, d_state, d_inner]
            let delta_exp = delta_t.unsqueeze(1)?.expand((batch, self.d_state, self.d_inner))?;
            let a_exp     = a.unsqueeze(0)?.expand((batch, self.d_state, self.d_inner))?;
            // let a_bar     = (a_exp * &delta_exp)?.neg()?.exp()?;   // [b, d_state, d_inner]
            let a_bar = a_exp.mul(&delta_exp)?.neg()?.exp()?;
            
            // B_bar = delta_t * b_t
            // b_t: [b, d_state] → [b, d_state, 1]
            let b_bar = b_t.unsqueeze(2)?.expand((batch, self.d_state, self.d_inner))?;
            // let b_bar = (&b_bar * &delta_exp)?;                    // [b, d_state, d_inner]
            let b_bar = b_bar.mul(&delta_exp)?;

            // u_t: [b, d_inner] → [b, 1, d_inner]
            let u_expanded = u_t.unsqueeze(1)?.expand((batch, self.d_state, self.d_inner))?;

            // h = A_bar * h + B_bar * u_t
            h = (a_bar.broadcast_mul(&h)? + b_bar.broadcast_mul(&u_expanded)?)?;

            // y_t = c_t @ h + d * u_t
            // c_t: [b, d_state] → [b, 1, d_state]
            let c_expanded = c_t.unsqueeze(1)?;                    // [b, 1, d_state]
            // h:   [b, d_state, d_inner]
            let ch = c_expanded.matmul(&h)?;                       // [b, 1, d_inner]
            let ch = ch.squeeze(1)?;                               // [b, d_inner]

            let d_expanded = d.unsqueeze(0)?.expand((batch, self.d_inner))?;
            // let du         = (&u_t * &d_expanded)?;                // [b, d_inner]
            let du         = u_t.mul(&d_expanded)?;
            let y_t        = (ch + du)?;                           // [b, d_inner]

            outputs.push(y_t.unsqueeze(1)?);                       // [b, 1, d_inner]
        }

        // [b, seq, d_inner]
        Tensor::cat(
            &outputs.iter().map(|t| t as &Tensor).collect::<Vec<_>>(),
            1,
        )
    }
}

impl Module for ZumarMambaBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_rank = x.rank();
        let x = if original_rank == 2 { x.unsqueeze(0)? } else { x.clone() };
        // x: [b, seq, d_model]

        // ── In Projection ────────────────────────────────────────
        let xz     = self.in_proj.forward(&x)?;             // [b, seq, d_inner*2]
        let chunks = xz.chunk(2, candle_core::D::Minus1)?;
        let x_path = &chunks[0];                            // [b, seq, d_inner]
        let z_path = &chunks[1];                            // [b, seq, d_inner]

        // ── ✅ Conv1d السببية (depthwise) ─────────────────────────
        let x_conv = self.apply_conv1d(x_path)?;           // [b, seq, d_inner]
        let x_conv = candle_nn::ops::silu(&x_conv)?;

        // ── dt Projection ────────────────────────────────────────
        let delta = self.dt_proj.forward(&x_conv)?;        // [b, seq, d_inner]

        // ── ✅ Selective Scan مع Causal Masking ───────────────────
        let a = self.a_log.neg()?.exp()?;                  // A من المعادلة الأصلية
        let y = self.selective_scan(&x_conv, &delta, &a, &self.d)?;
        // [b, seq, d_inner]

        // ── Gate (SiLU على z) ────────────────────────────────────
        let z_gated = candle_nn::ops::silu(z_path)?;
        let y_gated = (y * z_gated)?;

        // ── Out Projection ───────────────────────────────────────
        let output = self.out_proj.forward(&y_gated)?;    // [b, seq, d_model]

        if original_rank == 2 { output.squeeze(0) } else { Ok(output) }
    }
}
