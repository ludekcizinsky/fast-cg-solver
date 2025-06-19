#include "matrix_gpu.hh"
#include "matrix_gpu_kernels.cu"
#include <cuda_runtime.h>


MatrixGPU::MatrixGPU(int rows, int cols, int blockSize) : rows_(rows), cols_(cols), blockSize_(blockSize) {
    cudaMalloc(&data_, rows_ * cols_ * sizeof(double));
}

MatrixGPU::~MatrixGPU() {
    cudaFree(data_);
}

void MatrixGPU::matvec(const VectorGPU& x, VectorGPU& y) const {
    int numBlocks = (rows_ + blockSize_ - 1) / blockSize_;
    matvec_kernel<<<numBlocks, blockSize_>>>(rows_, cols_, data_, x.data(), y.data());
}

MatrixGPU& MatrixGPU::operator=(const Matrix& other) {
    if (rows_ * cols_ != other.m() * other.n()) {
        if (data_) cudaFree(data_);
        rows_ = other.m();
        cols_ = other.n();
        cudaMalloc(&data_, rows_ * cols_ * sizeof(double));
    }

    cudaMemcpy(data_, other.data(), rows_ * cols_ * sizeof(double), cudaMemcpyHostToDevice);
    return *this;
}
