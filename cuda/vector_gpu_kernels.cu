#include <cuda_runtime.h>

__global__ void daxpy_kernel(int n, double alpha, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += alpha * x[i];
    }
}

__global__ void dot_kernel(int n, const double* x, const double* y, double* result) {
    // shared memory among threads in the same block
    extern __shared__ double cache[]; 
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // fill the cache with results from threads in the block
    double temp = 0.0;
    while (i < n) {
        temp += x[i] * y[i];
        i += blockDim.x * gridDim.x; // in case we have less threads than elements
    }

    cache[tid] = temp;
    __syncthreads();

    // reduce the cache, final result in cache[0]
    // s>>=1 equivalent to s = s/2
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    // add to global result
    if (tid == 0) {
        atomicAdd(result, cache[0]);
    }
}