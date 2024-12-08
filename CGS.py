import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from mpi4py import MPI


# Sequential Classical Gram-Schmidt (CGS) implementation for QR factorization.
def cgs(A):

    m, n = A.shape
    R = np.zeros((n, n))
    Q = np.zeros((m, n))
    R[0, 0] = np.linalg.norm(A[:, 0], 2)
    Q[:, 0] = A[:, 0] / R[0, 0]
    for j in range(1, n):
        R[:j, j] = Q[:, :j].T @ A[:, j]
        Q[:, j] = A[:, j] - Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]

    return Q, R


# Parallel Classical Gram-Schmidt (CGS) algorithm for QR factorization using MPI.
def parallel_CGS(W_local,comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_m, n = W_local.shape
    R = np.zeros((n, n))
    Q_local = np.zeros((local_m, n))

    local_col = W_local[:, 0]
    beta = local_col @ local_col 
    beta = comm.allreduce(beta, op=MPI.SUM)
    R[0, 0] = np.sqrt(beta)
    Q_local[:, 0] = local_col / R[0, 0]

    for j in range(1, n):
        local_r = Q_local[:, :j].T @ W_local[:, j] 
        r = np.zeros(j)
        comm.Allreduce(local_r, r, op=MPI.SUM)
        R[:j, j] = r
        Q_local[:, j] = W_local[:, j] - Q_local[:, :j] @ R[:j, j]
        beta = np.dot(Q_local[:, j], Q_local[:, j])
        beta = comm.allreduce(beta, op=MPI.SUM)
        R[j, j] = np.sqrt(beta)

        Q_local[:, j] /= R[j, j]

    return Q_local, R



