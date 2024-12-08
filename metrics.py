import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def compute_metrics(Q):
    """
    loss : Loss of orthogonality ||I - Q^T Q||_2.
    cond_num : Condition number of Q.
    """
    identity_approx = Q.T @ Q
    loss = np.linalg.norm(np.eye(identity_approx.shape[0]) - identity_approx, ord=2)
    cond_num = np.linalg.cond(Q)
    return loss, cond_num


def distribute_matrix(W, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    m, n = W.shape if rank == 0 else (None, None)
    m = comm.bcast(m, root=0)
    n = comm.bcast(n, root=0)

    local_rows = m // size
    if rank == size - 1:
        local_W = W[rank * local_rows:]
    else:
        local_W = W[rank * local_rows:(rank + 1) * local_rows]

    return local_W

def record_metrics(Q, A_col, iteration, losses, conds, input_conds):
    """
    The orthogonality loss and condition number are recorded at each iteration.
    """
    loss, cond = compute_metrics(Q)
    input_cond = np.linalg.cond(A_col)
    losses.append((iteration, loss))
    conds.append((iteration, cond))
    input_conds.append((iteration, input_cond))


def compute_column_condition_numbers(A):
    condition_numbers = []
    for i in range(A.shape[1]):
        col = A[:, i].reshape(-1, 1)
        cond_num = np.linalg.norm(col, 2) * np.linalg.norm(np.linalg.pinv(col), 2)
        condition_numbers.append(cond_num)
    return condition_numbers


def plot_results(matrix_labels, seq_losses_cgs, par_losses_cgs, seq_losses_mgs, par_losses_mgs,
                 seq_conds_cgs, par_conds_cgs, seq_conds_mgs, par_conds_mgs, conds_input):
  
    plt.figure(figsize=(10,8))
    plt.subplot(1, 2, 1)
    plt.plot(matrix_labels, [np.log10(l) for l in seq_losses_cgs], marker='o', label="CGS Seq Loss (log)")
    plt.plot(matrix_labels, [np.log10(l) for l in par_losses_cgs], marker='o', label="CGS Par Loss (log)")
    plt.plot(matrix_labels, [np.log10(l) for l in seq_losses_mgs], marker='o', label="MGS Seq Loss (log)")
    plt.plot(matrix_labels, [np.log10(l) for l in par_losses_mgs], marker='o', label="MGS Par Loss (log)")
    plt.title("Loss of Orthogonality Comparison")
    plt.xlabel("Matrix Size")
    plt.ylabel(r"log$_{10}$(||I - Q$^T$Q||$_2$)")
    plt.legend()
    plt.grid()

    
    plt.subplot(1, 2, 2)
    input_cond_avg = [np.mean(conds) for conds in conds_input]
    plt.plot(matrix_labels, input_cond_avg, marker='o', label="Input Matrix Avg Cond")
    plt.plot(matrix_labels, [np.log10(c) for c in seq_conds_cgs], marker='o', label="CGS Seq Cond (log)")
    plt.plot(matrix_labels, [np.log10(c) for c in par_conds_cgs], marker='o', label="CGS Par Cond (log)")
    plt.plot(matrix_labels, [np.log10(c) for c in seq_conds_mgs], marker='o', label="MGS Seq Cond (log)")
    plt.plot(matrix_labels, [np.log10(c) for c in par_conds_mgs], marker='o', label="MGS Par Cond (log)")
    plt.title("Condition Numbers Comparison")
    plt.xlabel("Matrix Size")
    plt.ylabel(r"log$_{10}$(Condition Number)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()



"""
def log_metrics(iteration, Q, W_local):
    loss, cond_Q = compute_metrics(Q)
    
    cond_input = np.linalg.cond(W_local)

    metrics = {
        'iteration': iteration,
        'loss': loss,
        'cond_Q': cond_Q,
        'cond_input': cond_input
    }
    return metrics

def plot_metrics(metrics_list):

    iterations = [m['iteration'] for m in metrics_list]
    losses = [m['loss'] for m in metrics_list]
    cond_Qs = [m['cond_Q'] for m in metrics_list]
    plt.figure(figsize=(12, 6))

   
    plt.subplot(1, 2, 1)
    plt.plot(iterations, losses, marker='o', label='Loss of Orthogonality')
    plt.title('Loss of Orthogonality Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|I - Q^T Q\|_2$')
    plt.yscale('log')  
    plt.grid()
    

    plt.subplot(1, 2, 2)
    plt.plot(iterations, cond_Qs, marker='o', label='Condition Number of Q')
    plt.title('Condition Number of Q Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Condition Number')
    plt.yscale('log') 
    plt.grid()

    plt.tight_layout()
    plt.show()

"""