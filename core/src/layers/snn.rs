use candle_core::{Tensor, Result, DType};

/// طبقة Spiking Neural Network (LIF - Leaky Integrate-and-Fire)
/// 
/// هذه الطبقة تحاكي سلوك العصبونات البيولوجية:
/// - تتراكم الإشارات في جهد الغشاء
/// - عندما يتجاوز الجهد العتبة، تُرسل نبضة (spike)
/// - يتسرب الجهد مع مرور الوقت (leak)
pub struct ZumarSpikingLayer {
    /// عتبة إطلاق النبضة
    pub threshold: f32,
    /// معامل التسرب (0.0 = لا تسرب، 1.0 = تسرب كامل)
    pub leak_factor: f32,
    /// جهد الغشاء الحالي لكل عصبون
    pub v_mem: Option<Tensor>,
    /// عدد الخطوات الزمنية للمحاكاة
    pub time_steps: usize,
}

impl ZumarSpikingLayer {
    pub fn new(threshold: f32, leak_factor: f32, time_steps: usize) -> Self {
        Self {
            threshold,
            leak_factor: leak_factor.clamp(0.0, 1.0),
            v_mem: None,
            time_steps,
        }
    }

    /// إعادة تعيين حالة الغشاء
    pub fn reset(&mut self) {
        self.v_mem = None;
    }

    /// تمريرة أمامية نبضية
    /// 
    /// المدخلات: `x` - الإشارة العصبية
    /// المخرجات: نبضات (0 أو 1)
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        
        // تهيئة جهد الغشاء إذا لم يكن موجوداً
        if self.v_mem.is_none() {
            self.v_mem = Some(Tensor::zeros_like(x)?);
        }
        
        let mut total_spikes = Tensor::zeros_like(x)?;
        let mut v_mem = self.v_mem.take().unwrap();

        for _step in 0..self.time_steps {
            // 1. تسرب الجهد
            v_mem = (v_mem * self.leak_factor as f64)?;

            // 2. تراكم الإشارة المدخلة
            v_mem = (&v_mem + x)?;

            // 3. فحص العتبة وإطلاق النبضات
            let mask = v_mem.ge(self.threshold as f64)?;
            let spikes = mask.to_dtype(DType::F32)?;

            // 4. تجميع النبضات
            total_spikes = (&total_spikes + &spikes)?;

            // 5. إعادة تعيين الجهد للعصبونات التي أطلقت
            let reset_mask = spikes.ones_like()? - &spikes;
            v_mem = v_mem.broadcast_mul(&reset_mask)?;
        }

        // تخزين حالة الغشاء للخطوة التالية
        self.v_mem = Some(v_mem);

        // تطبيع النبضات بعدد الخطوات الزمنية
        total_spikes.broadcast_div(&Tensor::new(&[(self.time_steps as f32)], device)?)
    }
}