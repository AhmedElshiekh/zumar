fn main() {
    // إعداد بناء ملفات CUDA إذا كان العتاد يدعم ذلك
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=cudart");
        
        // استخدام nvcc لترجمة الـ kernel
        let status = std::process::Command::new("nvcc")
            .args(&["-ptx", "src/kernels/bitnet_kernel.cu", "-o", "src/kernels/bitnet_kernel.ptx"])
            .status()
            .unwrap();
        
        if !status.success() {
            panic!("Failed to compile CUDA kernel");
        }
    }
}
