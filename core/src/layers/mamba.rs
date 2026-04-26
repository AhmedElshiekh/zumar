use candle_core::{Tensor, Result, DType};
use candle_nn::{Module, VarBuilder, Linear};

pub struct ZumarMambaConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
}

pub struct ZumarMambaBlock {
    pub in_proj: Linear,
    pub conv1d_weight: Tensor,
    pub conv1d_bias: Tensor,
    pub x_proj: Linear,
    pub dt_proj: Linear,
    pub out_proj: Linear,
    pub a_log: Tensor,
    pub d: Tensor,
    pub d_state: usize,
    pub d_inner: usize,
    pub d_conv: usize,
}

impl ZumarMambaBlock {
    pub fn new(cfg: &ZumarMambaConfig, vs: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_model * cfg.expand;
        let _device = vs.device();

        let in_proj = candle_nn::linear(cfg.d_model, d_inner * 2, vs.pp("in_proj"))?;
        let x_proj = candle_nn::linear(d_inner, cfg.d_state * 2 + d_inner, vs.pp("x_proj"))?;
        let dt_proj = candle_nn::linear(d_inner, d_inner, vs.pp("dt_proj"))?;
        let out_proj = candle_nn::linear(d_inner, cfg.d_model, vs.pp("out_proj"))?;

        let a_log = vs.get_with_hints(
            (cfg.d_state, d_inner), "a_log", candle_nn::Init::Const(0.0),
        )?;
        let d = vs.get_with_hints(d_inner, "d", candle_nn::Init::Const(1.0))?;
        let conv1d_weight = vs.get_with_hints(
            (d_inner, 1, cfg.d_conv), "conv1d.weight",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 },
        )?;
        let conv1d_bias = vs.get_with_hints(
            d_inner, "conv1d.bias", candle_nn::Init::Const(0.0),
        )?;

        Ok(Self {
            in_proj, conv1d_weight, conv1d_bias, x_proj, dt_proj, out_proj,
            a_log, d, d_state: cfg.d_state, d_inner, d_conv: cfg.d_conv,
        })
    }

    fn selective_scan(
        &self,
        u: &Tensor,
        delta: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        d: &Tensor,
    ) -> Result<Tensor> {
        let (_batch, seqlen, dim) = u.dims3()?;
        let device = u.device();

        let delta = (delta.exp()? * 0.5f64)?;

        let a_expanded = a.unsqueeze(0)?.unsqueeze(0)?;
        let delta_expanded = delta.unsqueeze(2)?;
        let a_bar = a_expanded.broadcast_mul(&delta_expanded)?.exp()?;

        let b_expanded = b.unsqueeze(3)?;
        let b_bar = b_expanded.broadcast_mul(&delta_expanded)?;

        let mut h = Tensor::zeros((_batch, self.d_state, dim), DType::F32, device)?;
        let mut outputs = Vec::new();

        for t in 0..seqlen {
            let a_t = a_bar.get_on_dim(1, t)?;
            let b_t = b_bar.get_on_dim(1, t)?;
            let u_t = u.get_on_dim(1, t)?.unsqueeze(1)?;

            let ah = a_t.broadcast_mul(&h)?;
            let bu = b_t.broadcast_mul(&u_t)?;
            h = (ah + bu)?;

            let c_t = c.get_on_dim(1, t)?.unsqueeze(1)?;
            let ch = c_t.matmul(&h)?;
            let du = u_t.broadcast_mul(&d)?;
            let y = (ch + du)?;

            outputs.push(y);
        }

        Tensor::cat(&outputs.iter().map(|t| t as &Tensor).collect::<Vec<_>>(), 1)
    }

    fn silu(x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::silu(x)
    }

    fn simple_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, l, d) = x.dims3()?;
        let kernel_size = self.d_conv;

        if l < kernel_size {
            return Ok(x.clone());
        }

        let mut outputs = Vec::new();
        let kernel_flat = self.conv1d_weight.reshape((d, kernel_size))?;

        for t in 0..l {
            let start = if t >= kernel_size - 1 { t - (kernel_size - 1) } else { 0 };
            let window_size = t - start + 1;

            let mut result = Tensor::zeros((_b, 1, d), DType::F32, x.device())?;

            for w in 0..window_size {
                let idx = start + w;
                let slice = x.get_on_dim(1, idx)?.unsqueeze(1)?;
                let kernel_w = kernel_flat.get_on_dim(1, w)?;
                let weighted = slice.broadcast_mul(&kernel_w)?;
                result = (result + weighted)?;
            }

            let bias = self.conv1d_bias.unsqueeze(0)?.unsqueeze(0)?;
            result = (result + bias)?;
            outputs.push(result);
        }

        Tensor::cat(&outputs.iter().map(|t| t as &Tensor).collect::<Vec<_>>(), 1)
    }
}
impl Module for ZumarMambaBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_rank = x.rank();
        let x = if original_rank == 2 { x.unsqueeze(0)? } else { x.clone() };

        let xz = self.in_proj.forward(&x)?;
        let chunks = xz.chunk(2, candle_core::D::Minus1)?;
        let x_path = &chunks[0];
        let z_path = &chunks[1];

        let x_conv = self.simple_conv1d(x_path)?;
        let x_dbl = self.x_proj.forward(&x_conv)?;
        let dt = self.dt_proj.forward(&x_conv)?;

        let b = x_dbl.narrow(2, 0, self.d_state)?;
        let c = x_dbl.narrow(2, self.d_state, self.d_state)?;
        let u = x_dbl.narrow(2, self.d_state * 2, self.d_inner)?;

        let y = self.selective_scan(&u, &dt, &self.a_log, &b, &c, &self.d)?;

        let z_gated = Self::silu(z_path)?;
        let y_gated = y.broadcast_mul(&z_gated)?;
        let output = self.out_proj.forward(&y_gated)?;

        if original_rank == 2 { output.squeeze(0) } else { Ok(output) }
    }
}