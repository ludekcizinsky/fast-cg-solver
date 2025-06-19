#pragma once
#include <cuda_runtime.h>
#include <vector>

class VectorGPU {
public:
    VectorGPU(int n, int blockSize);
    ~VectorGPU();

    VectorGPU& operator=(const VectorGPU& other);
    VectorGPU& operator=(const std::vector<double>& other);

    __host__ double* data() { return data_; }
    __host__ const double* data() const { return data_; }

    void add(const VectorGPU& other, double alpha = 1.0);
    double dot(const VectorGPU& other) const;

private:
    int n_;
    int blockSize_;
    double* data_;
};