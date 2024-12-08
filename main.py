import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from generate_matrix import generate_matrix
from CGS import cgs, parallel_CGS
from MGS import mgs, parallel_MGS
from metrics import compute_metrics, distribute_matrix,compute_column_condition_numbers, plot_results



def gather_result(Q_local, comm):
    Q_parts = comm.gather(Q_local, root=0)
    if comm.Get_rank() == 0:
        return np.vstack(Q_parts)
    return None



def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    matrix_sizes = [(500, 50), (1000, 100), (5000, 500),(10000, 600),(50000,600)]

    # Initialize metrics
    seq_times_cgs, par_times_cgs = [], []
    seq_times_mgs, par_times_mgs = [], []
    seq_losses_cgs, par_losses_cgs = [], []
    seq_losses_mgs, par_losses_mgs = [], []
    seq_conds_cgs, par_conds_cgs = [], []
    seq_conds_mgs, par_conds_mgs = [], []
    conds_input = []

    for m, n in matrix_sizes:
        if rank == 0:
            print(f"\nTesting matrix of size {m}x{n}...")
            A = generate_matrix(m, n)
        else:
            A = None

        # Distribute matrix
        A = comm.bcast(A, root=0)
        local_A = distribute_matrix(A, comm)
        if rank == 0:
            conds_input.append(compute_column_condition_numbers(local_A))

        # Sequential CGS
        if rank == 0:
            start_time = MPI.Wtime()
            Q_seq_cgs, R_seq_cgs = cgs(A.copy())
            seq_times_cgs.append(MPI.Wtime() - start_time)
            print(f"Sequential CGS completed in {MPI.Wtime() - start_time:.4f} seconds.")
            loss_seq_cgs, cond_seq_cgs = compute_metrics(Q_seq_cgs)
            seq_losses_cgs.append(loss_seq_cgs)
            seq_conds_cgs.append(cond_seq_cgs)
            #print(f"Sequential CGS: Loss = {loss_seq_cgs:.4e}, Condition Number = {cond_seq_cgs:.4e}")

        # Parallel CGS
        comm.Barrier()
        start_time = MPI.Wtime()
        Q_local_cgs, R_par_cgs = parallel_CGS(local_A,comm)
        comm.Barrier()
        
        Q_par_cgs = gather_result(Q_local_cgs, comm)
        if rank == 0:
            par_times_cgs.append(MPI.Wtime() - start_time)
            print(f"Parallel CGS completed in {MPI.Wtime() - start_time:.4f} seconds.")
            loss_par_cgs, cond_par_cgs = compute_metrics(Q_par_cgs)
            par_losses_cgs.append(loss_par_cgs)
            par_conds_cgs.append(cond_par_cgs)
            #print(f"Parallel CGS: Loss = {loss_par_cgs:.4e}, Condition Number = {cond_par_cgs:.4e}")

        # Sequential MGS
        if rank == 0:
            start_time = MPI.Wtime()
            Q_seq_mgs, R_seq_mgs = mgs(A.copy())
            seq_times_mgs.append(MPI.Wtime() - start_time)
            print(f"Sequential MGS completed in {MPI.Wtime() - start_time:.4f} seconds.")
            loss_seq_mgs, cond_seq_mgs = compute_metrics(Q_seq_mgs)
            seq_losses_mgs.append(loss_seq_mgs)
            seq_conds_mgs.append(cond_seq_mgs)
            #print(f"Sequential MGS: Loss = {loss_seq_mgs:.4e}, Condition Number = {cond_seq_mgs:.4e}")

        # Parallel MGS
        comm.Barrier()
        start_time = MPI.Wtime()
        Q_local_mgs, R_par_mgs = parallel_MGS(local_A)
        comm.Barrier()
        Q_par_mgs = gather_result(Q_local_mgs, comm)
        if rank == 0:
            par_times_mgs.append(MPI.Wtime() - start_time)
            print(f"Parallel MGS completed in {MPI.Wtime() - start_time:.4f} seconds.")
            loss_par_mgs, cond_par_mgs = compute_metrics(Q_par_mgs)
            par_losses_mgs.append(loss_par_mgs)
            par_conds_mgs.append(cond_par_mgs)
            #print(f"Parallel MGS: Loss = {loss_par_mgs:.4e}, Condition Number = {cond_par_mgs:.4e}")

    # Plot results
    matrix_labels = [f"{m}x{n}" for m, n in matrix_sizes]
    if rank == 0:
        plt.figure(figsize=(10,6))
        plt.subplot(2, 2, 1)
        plt.plot(matrix_labels, seq_times_cgs, marker='o', label="Sequential CGS")
        plt.plot(matrix_labels, par_times_cgs, marker='o', label="Parallel CGS")
        plt.plot(matrix_labels, seq_times_mgs, marker='o', label="Sequential MGS")
        plt.plot(matrix_labels, par_times_mgs, marker='o', label="Parallel MGS")
        plt.title("Execution Time Comparison")
        plt.xlabel("Matrix Size")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid()
        plt.show()

        
        if not (len(matrix_labels) == len(seq_times_cgs) == len(par_times_cgs) ==
                len(seq_losses_cgs) == len(par_losses_cgs) == len(seq_losses_mgs) == len(par_losses_mgs)):
            raise ValueError("Data lengths mismatch. Ensure all data lists have the same length.")

        plot_results(matrix_labels, seq_losses_cgs, par_losses_cgs, seq_losses_mgs, par_losses_mgs,
                     seq_conds_cgs, par_conds_cgs, seq_conds_mgs, par_conds_mgs, conds_input)


if __name__ == "__main__":
    main()


"""
Testing matrix of size 500x50...
    Sequential CGS completed in 0.0015 seconds.
    Parallel CGS completed in 0.0124 seconds.
    Sequential MGS completed in 0.0128 seconds.
    Parallel MGS completed in 0.1226 seconds.

Testing matrix of size 1000x100...
    Sequential CGS completed in 0.0053 seconds.
    Parallel CGS completed in 0.0159 seconds.
    Sequential MGS completed in 0.0625 seconds.
    Parallel MGS completed in 0.4535 seconds.

Testing matrix of size 5000x500...
    Sequential CGS completed in 0.5455 seconds.
    Parallel CGS completed in 0.3943 seconds.
    Sequential MGS completed in 14.3078 seconds.
    Parallel MGS completed in 11.3058 seconds.

Testing matrix of size 10000x600...
    Sequential CGS completed in 1.5557 seconds.
    Parallel CGS completed in 1.1682 seconds.
    Sequential MGS completed in 34.0299 seconds.
    Parallel MGS completed in 21.9234 seconds.

Testing matrix of size 50000x600...
    Sequential CGS completed in 11.3910 seconds.
    Parallel CGS completed in 9.4679 seconds.
    Sequential MGS completed in 212.4596 seconds.
    Parallel MGS completed in 127.4689 seconds.


"""