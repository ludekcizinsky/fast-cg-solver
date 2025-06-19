#include "cg_mpi.hh"
#include <iostream>

int main(int argc, char *argv[])
{

    int psize, prank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    if (argc < 2)
    {
        if (prank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " [path to .mtx file]" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    CGSolverSparseMPI sparse_solver;
    sparse_solver.read_matrix(argv[1]);
    int n = sparse_solver.n();
    int m = sparse_solver.m();
    double h = 1. / n;
    sparse_solver.init_source_term(h);
    std::vector<double> x(n, 0.0);

    if (prank == 0)
    {
        std::cout << "Call CG sparse on matrix size " << m << " x " << n << ")" << std::endl;
    }

    double start_time = MPI_Wtime();
    sparse_solver.solve(x);
    double end_time = MPI_Wtime();

    if (prank == 0)
    {
        std::cout << "Solution time: " << end_time - start_time << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}