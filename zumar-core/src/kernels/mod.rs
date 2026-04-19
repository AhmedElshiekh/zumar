use candle_core::{cuda_backend::cudarc::driver::LaunchAsync, Device, Result, Tensor};

/// وظيفة لاستدعاء الـ CUDA Kernel الخاص بالـ 1-bit MatMul
pub fn launch_bitnet_kernel(
    input: &Tensor,
    weights: &Tensor,
    output_shape: (usize, usize),
) -> Result<Tensor> {
    let device = input.device();
    
    if let Device::Cuda(cuda_dev) = device {
        // 1. تحميل الـ Kernel المترجم (PTX)
        let ptx_content = include_str!("bitnet_kernel.ptx");
        cuda_dev.load_ptx(ptx_content.into(), "bitnet_module", &["fast_bit_linear_forward"])?;

        let func = cuda_dev.get_func("bitnet_module", "fast_bit_linear_forward").unwrap();

        // 2. إعداد أبعاد المصفوفات
        let (m, k) = input.dims2()?;
        let (n, _k_w) = weights.dims2()?;
        
        // 3. إنشاء Tensor فارغ للنتيجة
        let output = Tensor::zeros(output_shape, input.dtype(), device)?;

        // 4. إطلاق الـ Kernel على الـ GPU
        // نقوم بتحديد عدد الـ Blocks والـ Threads بناءً على حجم المصفوفة
        let cfg = LaunchConfig::for_num_elems((m * n) as u32);
        unsafe {
            func.launch(cfg, (input, weights, &output, m as i32, n as i32, k as i32))?;
        }

        Ok(output)
    } else {
        Err(candle_core::Error::Msg("CUDA device not found for bitnet kernel".into()))
    }
}
