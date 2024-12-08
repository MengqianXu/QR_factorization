import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from mpi4py import MPI
from metrics import distribute_matrix
from generate_matrix import generate_matrix
from TSQR import tsqr  


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Generate test matrices on rank 0
if rank == 0:
    np.random.seed(42)
    random_matrix = np.random.randn(50000, 600)
    hilbert_matrix = np.array([[1 / (i + j + 1) for j in range(600)] for i in range(50000)])
    parametric_matrix = generate_matrix(50000, 600)  # m = 50000, n = 600
else:
    random_matrix = hilbert_matrix = parametric_matrix = None


random_matrix = comm.bcast(random_matrix, root=0)
hilbert_matrix = comm.bcast(hilbert_matrix, root=0)
parametric_matrix = comm.bcast(parametric_matrix, root=0)


local_random_matrix = distribute_matrix(random_matrix, comm)
local_hilbert_matrix = distribute_matrix(hilbert_matrix, comm)
local_parametric_matrix = distribute_matrix(parametric_matrix, comm)

# Test TSQR on all local matrices
matrices = {
    'Random Matrix': local_random_matrix,
    'Hilbert Matrix': local_hilbert_matrix,
    'Parametric Matrix': local_parametric_matrix
}

for name, local_W in matrices.items():
    Q_local, R_final = tsqr(local_W)

    if rank == 0:
        print(f"\nResults for {name}:")
        print(f"Final R matrix (shape {R_final.shape}):")
        print(R_final)
