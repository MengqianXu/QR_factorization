import numpy as np

def generate_matrix(m, n):
    """
    Generate a matrix based on the given method.
    Parameters:
        m : Number of rows.
        n : Number of columns.
    Return:
        Generated matrix of size m x n.
    """
    x = np.linspace(0, 1, m)
    mu = np.linspace(0, 1, n)
    f = lambda x, mu: np.sin(10 * (mu + x)) / (np.cos(100 * (mu - x)) + 1.1)
    matrix = np.array([[f((i - 1) / (m - 1), (j - 1) / (n - 1)) for j in range(1, n + 1)] for i in range(1, m + 1)])
    return matrix