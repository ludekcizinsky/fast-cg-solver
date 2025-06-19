#include "cg.hh"
#include "vector_gpu.hh"
#include "matrix_gpu.hh"

#include <algorithm>
#include <cmath>
#include <iostream>

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;

/*
    cgsolver solves the linear equation A*x = b where A is
    of size m x n

Code based on MATLAB code (from wikipedia ;-)  ):

function x = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end

*/
void CGSolver::solve(std::vector<double> & x, int block_size) {

  MatrixGPU d_A(m_m, m_n, block_size);
  d_A = m_A;
  VectorGPU d_x(m_n, block_size);
  d_x = x;
  VectorGPU d_b(m_n, block_size);
  d_b = m_b;
  VectorGPU d_tmp(m_n, block_size);

  // r = b - A * x;
  VectorGPU d_Ap(m_n, block_size);
  d_A.matvec(d_x, d_Ap);

  VectorGPU d_r(m_n, block_size);
  d_r = d_b;
  d_r.add(d_Ap, -1.0);

  // p = r;
  VectorGPU d_p(m_n, block_size);
  d_p = d_r;

  // rsold = r' * r;
  double h_rsold = d_r.dot(d_r);

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k) {
    // Ap = A * p;
    d_A.matvec(d_p, d_Ap);

    // alpha = rsold / (p' * Ap);
    double h_alpha = h_rsold / std::max(d_p.dot(d_Ap), h_rsold * NEARZERO);

    // x = x + alpha * p;
    d_x.add(d_p, h_alpha);

    // r = r - alpha * Ap;
    d_r.add(d_Ap, -h_alpha);

    // rsnew = r' * r;
    double h_rsnew = d_r.dot(d_r);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(h_rsnew) < m_tolerance)
      break; // Convergence test

    // p = r + (rsnew / rsold) * p;
    double h_beta = h_rsnew / h_rsold;
    d_tmp = d_r;
    d_tmp.add(d_p, h_beta);
    d_p = d_tmp;

    // rsold = rsnew;
    h_rsold = h_rsnew;
    if (DEBUG) {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(h_rsold) << "\r" << std::flush;
    }
  }

  if (DEBUG) {
    d_A.matvec(d_x, d_r);
    d_r.add(d_b, -1.0);
    auto res = std::sqrt(d_r.dot(d_r)) /
               std::sqrt(d_b.dot(d_b));
    auto nx = std::sqrt(d_x.dot(d_x));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(h_rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }

}

void CGSolver::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}


/*
Initialization of the source term b
*/
void Solver::init_source_term(double h) {
  m_b.resize(m_n);

  for (int i = 0; i < m_n; i++) {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}
