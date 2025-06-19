#include "matrix_coo.hh"
#include <cblas.h>
#include <string>
#include <vector>
#include <mpi.h>

#ifndef __CG_MPI_HH__
#define __CG_MPI_HH__

class Solver
{
public:
  virtual void read_matrix(const std::string &filename) = 0;
  void init_source_term(double h);
  virtual void solve(std::vector<double> &x) = 0;

  inline int m() const
  {
    return m_m;
  }
  inline int n() const
  {
    return m_n;
  }

  void tolerance(double tolerance)
  {
    m_tolerance = tolerance;
  }

protected:
  int m_m{0};
  int m_n{0};
  std::vector<double> m_b;
  double m_tolerance{1e-10};
};

class CGSolverSparseMPI : public Solver
{
public:
  CGSolverSparseMPI();
  virtual void read_matrix(const std::string &filename);
  virtual void solve(std::vector<double> &x);

private:
  // Define MPI specific variables
  int rank;
  int size;
  int local_start; // starting index of the local portion of the matrix
  int local_size;  // number of rows handled by this process

  // Define the local portion of the matrix
  MatrixCOO m_A;
};

#endif /* __CG_MPI_HH__ */