#include "matrix.hh"
#include "matrix_gpu.hh"
#include <string>
#include <vector>

#ifndef __CG_HH__
#define __CG_HH__


class Solver {
public:
  virtual void read_matrix(const std::string & filename) = 0;
  void init_source_term(double h);
  virtual void solve(std::vector<double> & x, int block_size) = 0;

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  void tolerance(double tolerance) { m_tolerance = tolerance; }

protected:
  int m_m{0};
  int m_n{0};
  std::vector<double> m_b;
  double m_tolerance{1e-10};
};

class CGSolver : public Solver {
public:
  CGSolver() = default;
  virtual void read_matrix(const std::string & filename);
  virtual void solve(std::vector<double> & x, int block_size);

private:
  Matrix m_A;
};

#endif /* __CG_HH__ */
