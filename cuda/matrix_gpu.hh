#pragma once
#include "vector_gpu.hh"
#include "matrix.hh"

class MatrixGPU {
public:
    MatrixGPU(int rows, int cols, int blockSize);
    ~MatrixGPU();

    void matvec(const VectorGPU& x, VectorGPU& y) const;
    MatrixGPU& operator=(const Matrix& other);

private:
    int rows_;
    int cols_;
    int blockSize_;
    double* data_;
};