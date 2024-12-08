# Parallel and Sequential Implementations of QR Factorization

This project implements and compares sequential and parallel versions of Classical Gram-Schmidt (CGS), Modified Gram-Schmidt (MGS), and Tall-Skinny QR (TSQR) algorithms for QR factorization. The implementations leverage MPI for parallelization and are tested on various matrices to evaluate their performance, numerical stability, and scalability.

---

## Features

1. **Algorithms Implemented**:
   - Classical Gram-Schmidt (CGS): Sequential and Parallel.
   - Modified Gram-Schmidt (MGS): Sequential and Parallel.
   - Tall-Skinny QR (TSQR): Parallel.

2. **Matrix Generation**:
   - Random matrix generation.
   - Parametric matrix based on the function:
     \[
     f(x, \mu) = \frac{\sin(10(\mu + x))}{\cos(100(\mu - x)) + 1.1}
     \]
   - Hilbert matrix for challenging numerical stability tests.

3. **Metrics Evaluated**:
   - Execution time.
   - Loss of orthogonality (\( \|I - Q^T Q\|_2 \)).
   - Condition number of the computed \( Q \) matrix.

4. **Visualization**:
   - Plots comparing runtime, orthogonality loss, and condition numbers for different algorithms and matrix sizes.

---

## Setup

### Dependencies

- Python 3.8+
- `numpy`
- `matplotlib`
- `mpi4py`
- MPI environment.

---

## Running the Project

### 1. Sequential and Parallel CGS/MGS Testing

- **File**: `main.py`
- **Description**: Tests sequential and parallel implementations of CGS and MGS on multiple matrix sizes, evaluates performance metrics, and generates visualizations.
- **Run Command**:
  ```bash
  mpirun -np <num_processes> python main.py
  ```
- **Outputs**:
  - Execution times for sequential and parallel implementations.
  - Loss of orthogonality and condition numbers.
  - Visualization plots saved as PNGs.

---

### 2. TSQR Testing

- **File**: `Test_TSQR.py`
- **Description**: Implements the TSQR algorithm for large matrices distributed across processes and outputs the final \( R \) matrix.
- **Run Command**:
  ```bash
  mpirun -np <num_processes> python Test_TSQR.py
  ```
- **Outputs**:
  - Final \( R \) matrix from TSQR.
  - Results for scalability and correctness.

---

### 3. Benchmarking on Test Matrices

- **Test Matrices**:
  - Random matrices.
  - Hilbert matrices.
  - Parametric matrices of size \( 50000 \times 600 \).

---

## Project Files

| **File**         | **Description**                                                                                           |
|-------------------|-----------------------------------------------------------------------------------------------------------|
| `main.py`         | Runs sequential and parallel CGS/MGS on various matrices, evaluates metrics, and generates visualizations. |
| `Test_TSQR.py`    | Implements and tests TSQR on distributed matrices.                                                        |
| `CGS.py`          | Contains the sequential and parallel implementations of Classical Gram-Schmidt.                           |
| `MGS.py`          | Contains the sequential and parallel implementations of Modified Gram-Schmidt.                            |
| `TSQR.py`         | Implements the TSQR algorithm for distributed QR factorization.                                           |
| `generate_matrix.py` | Functions for generating parametric matrices.                                                          |
| `metrics.py`      | Provides utility functions for computing metrics, distributing matrices, and plotting results.            |
| `loss_cond.png`   | Plot showing loss of orthogonality and condition numbers for different algorithms.                        |
| `Time.png`        | Plot comparing execution times of sequential and parallel CGS/MGS.                                        |

---

## Results

### Execution Times (Refer to `Time.png`)
- Parallel implementations significantly reduce computation time, especially for large matrices.
- CGS is faster than MGS for both sequential and parallel cases, but MGS demonstrates better numerical stability.

### Loss of Orthogonality and Condition Numbers (Refer to `loss_cond.png`)
- MGS maintains lower orthogonality loss and better-conditioned \( Q \) matrices than CGS.
- TSQR scales well for distributed matrices but is limited to final results due to its distributed nature.



