#include <cuda_runtime.h>

__global__ void matvec_kernel(int rows, int cols, const double* matrix_data, const double* vector_data, double* result_data) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            sum += matrix_data[row * cols + col] * vector_data[col];
        }
        result_data[row] = sum;
    }
}