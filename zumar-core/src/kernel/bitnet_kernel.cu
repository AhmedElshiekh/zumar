// CUDA Kernel لعمليات الجمع والطرح بناءً على إشارة الوزن (Sign-based)
extern "C" __global__ void fast_bit_linear_forward(
    const float* input,    // المدخلات
    const int8_t* weights, // أوزان 1-بت محزومة
    float* output,         // الناتج
    int M, int N, int K    // الأبعاد
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            // بدلاً من الضرب: نجمع أو نطرح بناءً على قيمة الوزن {-1, 1}
            // الوزن المخزن هنا هو الإشارة فقط
            if (weights[row * K + i] > 0) {
                sum += input[col * K + i];
            } else {
                sum -= input[col * K + i];
            }
        }
        output[row * N + col] = sum;
    }
}
