#include <cuda_runtime.h>

#define TILE_SIZE 16

extern "C" __global__ void fast_bit_linear_forward(
    const float* __restrict__ input,    
    const int8_t* __restrict__ weights, 
    float* __restrict__ output,         
    int M, int N, int K    
) {
    // استخدام Shared Memory لتقليل الوصول للذاكرة الخارجية
    __shared__ float tileInput[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t tileWeights[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // تحميل البيانات إلى الـ Shared Memory بشكل تعاوني
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tileInput[threadIdx.y][threadIdx.x] = input[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileInput[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tileWeights[threadIdx.y][threadIdx.x] = weights[col * K + t * TILE_SIZE + threadIdx.y];
        else
            tileWeights[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // الحساب الفعلي داخل الـ Tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            int8_t w = tileWeights[i][threadIdx.x];
            if (w > 0) sum += tileInput[threadIdx.y][i];
            else if (w < 0) sum -= tileInput[threadIdx.y][i];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}
