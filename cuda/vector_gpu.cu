#include "vector_gpu.hh"
#include "vector_gpu_kernels.cu"
#include <cuda_runtime.h>


VectorGPU::VectorGPU(int n, int blockSize) : n_(n), blockSize_(blockSize) {
    cudaMalloc(&data_, n_ * sizeof(double));
}

VectorGPU::~VectorGPU() {
    cudaFree(data_);
}

VectorGPU& VectorGPU::operator=(const VectorGPU& other) {
    if (this == &other) return *this;

    if (n_ != other.n_) {
        if (data_) cudaFree(data_);
        n_ = other.n_;
        cudaMalloc(&data_, n_ * sizeof(double));
    }

    cudaMemcpy(data_, other.data_, n_ * sizeof(double), cudaMemcpyDeviceToDevice);
    return *this;
}

VectorGPU& VectorGPU::operator=(const std::vector<double>& other) {
    if (n_ != static_cast<int>(other.size())) {
        if (data_) cudaFree(data_);
        n_ = static_cast<int>(other.size());
        cudaMalloc(&data_, n_ * sizeof(double));
    }
    cudaMemcpy(data_, other.data(), n_ * sizeof(double), cudaMemcpyHostToDevice);
    return *this;
}


void VectorGPU::add(const VectorGPU& other, double alpha) {
    int numBlocks = (n_ + blockSize_ - 1) / blockSize_;
    daxpy_kernel<<<numBlocks, blockSize_>>>(n_, alpha, other.data_, data_);
}

// dot product implemented with shared memory + atomicAdd
double VectorGPU::dot(const VectorGPU& other) const {
    int numBlocks = (n_ + blockSize_ - 1) / blockSize_;
    
    double* d_result;
    cudaMalloc(&d_result, sizeof(double));
    cudaMemset(d_result, 0, sizeof(double));

    dot_kernel<<<numBlocks, blockSize_, blockSize_ * sizeof(double)>>>(n_, data_, other.data_, d_result);

    double h_result;
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    return h_result;
}
