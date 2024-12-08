import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from mpi4py import MPI
'''
def householder_qr(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for i in range(min(m, n)):
        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x) * (-1 if x[0] < 0 else 1)
        u = (x - e) / np.linalg.norm(x - e)
        H = np.eye(m - i) - 2 * np.outer(u, u)
        R[i:, i:] = np.dot(H, R[i:, i:])
        Q[:, i:] = np.dot(Q[:, i:], np.vstack([np.eye(i), H]))

    return Q, R
'''



def tsqr(W_local):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    Q_local, R_local = np.linalg.qr(W_local)

    step = 1
    while size > step:
        partner = rank ^ step
        if partner < size:
            if rank < partner:
                R_partner = np.empty_like(R_local)
                comm.Recv(R_partner, source=partner)
                R_combined = np.vstack([R_local, R_partner])
                _, R_local = np.linalg.qr(R_combined)
            else:
                comm.Send(R_local, dest=partner)
                break
        step *= 2

    R_final = comm.gather(R_local if rank == 0 else None, root=0)
    if rank == 0:
        R_final = np.vstack(R_final)
    else:
        R_final = None

    return Q_local, R_final




