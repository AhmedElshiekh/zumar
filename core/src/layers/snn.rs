use candle_core::{Tensor, Result, DType};

pub struct ZumarSpikingLayer {
    pub threshold: f32,
    pub tau: f32,
    pub v_mem: Option<Tensor>,
    pub time_steps: usize,
}

impl ZumarSpikingLayer {
    pub fn new(threshold: f32, tau: f32, time_steps: usize) -> Self {
        Self {
            threshold,
            tau: tau.clamp(0.1, 1.0),
            v_mem: None,
            time_steps: time_steps.max(1),
        }
    }

    pub fn reset(&mut self) {
        self.v_mem = None;
    }

    fn surrogate_gradient(v: &Tensor, threshold: f32) -> Result<Tensor> {
        let device = v.device();
        let thresh = Tensor::new(&[threshold], device)?;
        let abs_diff = (v - &thresh)?.abs()?;
        let width = Tensor::new(&[1.0f32], device)?;
        let sub = width.broadcast_sub(&abs_diff)?;
        sub.relu()
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();

        if self.v_mem.is_none() {
            self.v_mem = Some(Tensor::zeros_like(x)?);
        }

        let mut v_mem = self.v_mem.take().unwrap();
        let x_flat = x.flatten_all()?;
        let mut total_spikes = Tensor::zeros_like(&x_flat)?;

        let tau_tensor = Tensor::new(&[self.tau], device)?;
        let thresh_tensor = Tensor::new(&[self.threshold], device)?;

        for _step in 0..self.time_steps {
            // 1. تسرب
            v_mem = v_mem.broadcast_mul(&tau_tensor)?;

            // 2. تكامل
            v_mem = (&v_mem + &x_flat)?;

            // 3. فحص العتبة
            let above_thresh = v_mem.ge(&thresh_tensor)?;

            // 4. توليد النبضات
            let spikes = above_thresh.to_dtype(DType::F32)?;
            let surrogate = Self::surrogate_gradient(&v_mem, self.threshold)?;

            // دمج النبضة مع التدرج البديل
            // detach() ترجع Tensor مباشرة، لا تحتاج ?
            let diff = (&spikes - &surrogate)?;
            let detached_diff = diff.detach();
            let spike_output = (detached_diff + &surrogate)?;

            // 5. إعادة تعيين الجهد
            let reset_mask = ((&spikes * -1.0f64)? + 1.0f64)?;
            v_mem = v_mem.broadcast_mul(&reset_mask)?;

            total_spikes = (&total_spikes + &spike_output)?;
        }

        self.v_mem = Some(v_mem);

        let spikes_reshaped = total_spikes.reshape(x.shape())?;
        let time_val = (self.time_steps as f32).max(1.0);
        spikes_reshaped.broadcast_div(&Tensor::new(&[time_val], device)?)
    }
}