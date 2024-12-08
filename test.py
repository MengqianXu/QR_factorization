import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
# Set environment variables for thread control
import time
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from generate_matrix import generate_matrix
from metrics import compute_column_condition_numbers, compute_metrics, distribute_matrix, log_metrics, plot_metrics
from CGS import cgs, parallel_CGS
from MGS import mgs, parallel_MGS



def gather_result(Q_local, comm):
    Q_parts = comm.gather(Q_local, root=0)
    if comm.Get_rank() == 0:
        return np.vstack(Q_parts)
    return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Define matrix sizes to test
    matrix_sizes = [(500, 50), (1000, 100), (5000, 500), (10000, 600)]
    # Initialize lists for results
    seq_cgs_times, par_cgs_times = [], []
    seq_mgs_times, par_mgs_times = [], []
    size_labels = []
    
    losses_cgs_seq, losses_cgs_par = [], []
    losses_mgs_seq, losses_mgs_par = [], []
    
    conds_cgs_seq, conds_cgs_par = [], []
    conds_mgs_seq, conds_mgs_par = [], []
    
    conds_input = []

    for m, n in matrix_sizes:
        if rank == 0:
            print(f"\nTesting matrix of size {m}x{n}...")
            W = generate_matrix(m, n)
        else:
            W = None

        # Broadcast the matrix to all processes
        W = comm.bcast(W, root=0)
        local_W = distribute_matrix(W, comm)
        conds_input.append(compute_column_condition_numbers(local_W))

        # Sequential CGS
        if rank == 0:
            start_time = time.time()
            Q_cgs, R_cgs = cgs(W.copy())
            seq_cgs_time = time.time() - start_time
            print(f"Sequential CGS completed in {seq_cgs_time:.4f} seconds.")
            seq_cgs_times.append(seq_cgs_time)
            loss_cgs, cond_cgs = compute_metrics(Q_cgs)
            losses_cgs_seq.append(loss_cgs)
            conds_cgs_seq.append(cond_cgs)

        # Parallel CGS
        comm.Barrier()
        start_time = time.time()
        Q_local_cgs, R_cgs_parallel = parallel_CGS(local_W)
        par_cgs_time = time.time() - start_time
        
        Q_cgs_parallel = gather_result(Q_local_cgs, comm)
        if rank == 0:
            print(f"Parallel CGS completed in {par_cgs_time:.4f} seconds.")
            par_cgs_times.append(par_cgs_time)
            loss_cgs_par, cond_cgs_par = compute_metrics(Q_cgs_parallel)
            losses_cgs_par.append(loss_cgs_par)
            conds_cgs_par.append(cond_cgs_par)
        

        # Sequential MGS
        if rank == 0:
            start_time = time.time()
            Q_mgs, R_mgs = mgs(W.copy())
            seq_mgs_time = time.time() - start_time
            print(f"Sequential MGS completed in {seq_mgs_time:.4f} seconds.")
            seq_mgs_times.append(seq_mgs_time)
            loss_mgs, cond_mgs = compute_metrics(Q_mgs)
            losses_mgs_seq.append(loss_mgs)
            conds_mgs_seq.append(cond_mgs)

        # Parallel MGS
        comm.Barrier()
        start_time = time.time()
        Q_local_mgs, R_mgs_parallel = parallel_MGS(local_W)
        par_mgs_time = time.time() - start_time
       
        Q_mgs_parallel = gather_result(Q_local_mgs, comm)
        if rank == 0:
            print(f"Parallel MGS completed in {par_mgs_time:.4f} seconds.")
            par_mgs_times.append(par_mgs_time)
            loss_mgs_par, cond_mgs_par = compute_metrics(Q_mgs_parallel)
            losses_mgs_par.append(loss_mgs_par)
            conds_mgs_par.append(cond_mgs_par)

        

        if rank == 0:
            size_labels.append(f"{m}x{n}")

    # Plot timing results
    if rank == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(size_labels, seq_cgs_times, marker='o', label="Sequential CGS")
        plt.plot(size_labels, par_cgs_times, marker='o', label="Parallel CGS")
        plt.plot(size_labels, seq_mgs_times, marker='o', label="Sequential MGS")
        plt.plot(size_labels, par_mgs_times, marker='o', label="Parallel MGS")
        plt.title("Execution Time Comparison")
        plt.xlabel("Matrix Size (m x n)")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid()
        plt.show()
        
        sizes = [f"{m}x{n}" for m, n in matrix_sizes]
       
        # Loss of Orthogonality Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(sizes, losses_cgs_seq, marker='o', label="CGS Sequential")
        plt.plot(sizes, losses_cgs_par, marker='o', label="CGS Parallel")
        plt.plot(sizes, losses_mgs_seq, marker='o', label="MGS Sequential")
        plt.plot(sizes, losses_mgs_par, marker='o', label="MGS Parallel")
        
        plt.title("Loss of Orthogonality")
        plt.xlabel("Matrix Size")
        plt.ylabel(r"$\|I - Q^T Q\|_2$")
        plt.yscale('log') 
        plt.legend()
        plt.grid()

        
        # Condition Numbers Plot
        plt.subplot(1, 2, 2)
        input_cond_avg = [np.mean(conds) for conds in conds_input]
        plt.plot(sizes, input_cond_avg, marker='o', label="Input Matrix Avg Cond")
        plt.plot(sizes, conds_cgs_seq, marker='o', label="CGS Sequential")
        plt.plot(sizes, conds_cgs_par, marker='o', label="CGS Parallel")
        plt.plot(sizes, conds_mgs_seq , marker='o', label="MDS Sequential")
        plt.plot(sizes , conds_mgs_par , marker='o' , label="MDS Parallel")

        plt.title("Condition Numbers")
        plt.xlabel("Matrix Size")
        plt.ylabel("Condition Number")
        plt.yscale('log') 
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
        

if __name__ == "__main__":
    main()