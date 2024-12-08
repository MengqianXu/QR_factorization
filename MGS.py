import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from mpi4py import MPI

#Modified Gram-Schmidt (MGS) implementation for QR factorization.
def mgs(A):
    m, n = A.shape
    R = np.zeros((n, n))
    Q = np.zeros_like(A)

    for j in range(n):
        R[j, j] = np.linalg.norm(A[:, j])
        Q[:, j] = A[:, j] / R[j, j]
        for i in range(j + 1, n):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            A[:, i] -= Q[:, j] * R[j, i]

    return Q, R


#Parallel Modified Gram-Schmidt (1D-MGS) implementation for QR factorization.
def parallel_MGS(W_local):
    """
    Q_local : Local part of the orthogonal matrix Q (matching distribution of A_local).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_m, n = W_local.shape
    R = np.zeros((n, n))  
    Q_local = W_local.copy()  

    for j in range(n):
        for i in range(j):
            rho_local = np.dot(Q_local[:, i], Q_local[:, j])
            rho = comm.allreduce(rho_local, op=MPI.SUM)  
            R[i, j] = rho  
            Q_local[:, j] -= Q_local[:, i] * R[i, j]

        beta_local = np.dot(Q_local[:, j], Q_local[:, j])
        beta = comm.allreduce(beta_local, op=MPI.SUM)  
        R[j, j] = np.sqrt(beta)  

        Q_local[:, j] /= R[j, j]

    return Q_local, R