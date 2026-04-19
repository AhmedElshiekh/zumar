use candle_core::Device;

pub enum WorkloadIntensity {
    Low,    // مهام بسيطة (تصحيح لغوي، ردود قصيرة)
    High,   // مهام ثقيلة (كتابة كود، تحليل ملفات ضخمة)
}

pub struct HardwareRouter {
    cpu_device: Device,
    gpu_device: Option<Device>,
}

impl HardwareRouter {
    pub fn new() -> Self {
        let gpu = if candle_core::utils::cuda_is_available() {
            Some(Device::new_cuda(0).unwrap())
        } else {
            None
        };

        Self {
            cpu_device: Device::Cpu,
            gpu_device: gpu,
        }
    }

    pub fn route(&self, prompt: &str) -> &Device {
        // منطق بسيط: إذا كان السؤال أطول من 500 حرف، استخدم الـ GPU
        if prompt.len() > 500 && self.gpu_device.is_some() {
            println!("⚡ Routing to GPU for heavy workload...");
            self.gpu_device.as_ref().unwrap()
        } else {
            println!("🍃 Routing to CPU for efficiency...");
            &self.cpu_device
        }
    }
}
