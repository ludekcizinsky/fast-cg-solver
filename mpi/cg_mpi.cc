#include "cg_mpi.hh"
#include <algorithm>
#include <cmath>
#include <iostream>

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;

void CGSolverSparseMPI::solve(std::vector<double> &x)
{
    std::vector<double> r(m_n);
    std::vector<double> p(m_n);
    std::vector<double> Ap(m_n);
    std::vector<double> local_Ap(m_n);
    std::vector<double> tmp(m_n);
    int converged = 0;

    // Ax = A * x;
    m_A.mat_vec(x, local_Ap);
    MPI_Reduce(local_Ap.data(), Ap.data(), m_n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // r = b - A * x;
    r = m_b;
    cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);

    // p = r;
    p = r;

    // Broadcast initial vector p to all processes from the root process
    MPI_Bcast(p.data(), m_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // rsold = r' * r
    double rsold = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

    // Start CG iterations
    int k = 0;
    for (; k < m_n; ++k)
    {

        // Ap = A * p;
        m_A.mat_vec(p, local_Ap);
        MPI_Reduce(local_Ap.data(), Ap.data(), m_n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            // alpha = rsold / (p' * Ap);
            double dot_p_Ap = cblas_ddot(m_n, p.data(), 1, Ap.data(), 1);
            double alpha = rsold / std::max(dot_p_Ap, rsold * NEARZERO);

            // x = x + alpha * p;
            cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);

            // r = r - alpha * Ap;
            cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);

            // rsnew = r' * r;
            double rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

            // Check convergence (root decides)
            if (std::sqrt(rsnew) < m_tolerance)
            {
                converged = 1;
            }

            double beta = rsnew / rsold;
            // p = r + (rsnew / rsold) * p;
            tmp = r;
            cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
            p = tmp;

            // Update rsold for the next iteration (root only)
            rsold = rsnew;

            if (DEBUG && rank == 0)
            {
                std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << "\r" << std::flush;
            }
        }

        // Broadcast updated vector p from root to all processes for next iteration
        MPI_Bcast(p.data(), m_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Check convergence across all processes
        MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (converged == 1)
        {
            break;
        }
    }

    if (DEBUG)
    {
        // Make sure all processes have the same x vector
        MPI_Bcast(x.data(), m_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Ax = A * x
        std::vector<double> local_r(m_n);
        m_A.mat_vec(x, local_r);
        MPI_Reduce(local_r.data(), r.data(), m_n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Compute the statistics
        if (rank == 0)
        {
            cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
            auto res =
                std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) / std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
            auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
            std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << ", ||x|| = " << nx
                      << ", ||Ax - b||/||b|| = " << res << std::endl;
        }
    }
}

/*
Obtain the rank and size of the MPI process during initialization
*/
CGSolverSparseMPI::CGSolverSparseMPI()
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}

/*
Read the matrix from the file and distribute the nnz entries across all processes
*/
void CGSolverSparseMPI::read_matrix(const std::string &filename)
{
    // Read A and inform others about its specs
    bool is_root = (rank == 0);
    if (is_root)
    {
        m_A.read(filename);
        m_m = m_A.m();
        m_n = m_A.n();
    }
    MPI_Bcast(&m_m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m_A.m_is_sym, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute A across all processes
    if (is_root)
    {

        int total_nnz = m_A.irn.size();

        if (DEBUG)
        {
            std::cout << "[ReadMatrix] Total non-zero entries: " << total_nnz << std::endl;
        }

        // Distribute non-zero entries directly
        // (Note: do this backwards to avoid overwriting of A at root, only at the end)
        for (int p = size - 1; p > -1; p--)
        {
            int start_nnz = (total_nnz * p) / size;
            int end_nnz = (total_nnz * (p + 1)) / size;

            // Create local data for each process
            std::vector<int> p_irn;
            std::vector<int> p_jcn;
            std::vector<double> p_a;

            for (int z = start_nnz; z < end_nnz; z++)
            {
                p_irn.push_back(m_A.irn[z]);
                p_jcn.push_back(m_A.jcn[z]);
                p_a.push_back(m_A.a[z]);
            }

            // non-root processes
            if (p > 0)
            {
                int nz = p_irn.size();
                MPI_Send(&nz, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

                if (nz > 0)
                {
                    MPI_Send(p_irn.data(), nz, MPI_INT, p, 1, MPI_COMM_WORLD);
                    MPI_Send(p_jcn.data(), nz, MPI_INT, p, 2, MPI_COMM_WORLD);
                    MPI_Send(p_a.data(), nz, MPI_DOUBLE, p, 3, MPI_COMM_WORLD);
                }
            }
            // root process
            else
            {
                m_A.irn = p_irn;
                m_A.jcn = p_jcn;
                m_A.a = p_a;
            }
        }
        if (DEBUG)
        {
            std::cout << "[ReadMatrix Rank " << rank << "] # of nnz entries: " << m_A.irn.size() << std::endl;
        }
    }
    else
    {
        // Receive sizes first
        int nz;
        MPI_Recv(&nz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Allocate local arrays with correct sizes
        m_A.irn.resize(nz);
        m_A.jcn.resize(nz);
        m_A.a.resize(nz);

        // Receive data
        if (nz > 0)
        {
            MPI_Recv(m_A.irn.data(), nz, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(m_A.jcn.data(), nz, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(m_A.a.data(), nz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (DEBUG)
        {
            std::cout << "[ReadMatrix Rank " << rank << "] # of nnz entries: " << m_A.irn.size() << std::endl;
        }
    }
}

/*
Initialization of the source term b
*/
void Solver::init_source_term(double h)
{
    m_b.resize(m_n);
    for (int i = 0; i < m_n; i++)
    {
        m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) * std::sin(10. * M_PI * i * h);
    }
}