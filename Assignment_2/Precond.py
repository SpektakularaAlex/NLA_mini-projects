import numpy as np
from scipy.linalg import qr, svd, null_space
import matplotlib.pyplot as plt
from scipy import sparse as sp
from PIL import Image
from scipy.linalg import solve_triangular


def Incomp_Chol(A_full_problem_matrix):
    n = A_full_problem_matrix.shape[0]
    L = np.zeros_like(A_full_problem_matrix)
    
    for j in range(n):
        s = A_full_problem_matrix[j, j] - L[j, :j] @ L[j, :j]
        if s < 0:
            raise ValueError(f"breakdown at column {j}: pivot {s:.3e}, square-root of neg-number")
        L[j, j] = np.sqrt(s)
        for i in range(j+1, n):
            if A_full_problem_matrix[i, j] == 0:
                print(f"Incomplete restriction in effect, meaning that a_ij = 0")
                pass
            nominator = A_full_problem_matrix[i, j] - L[i, :j] @ L[j, :j]
            denom = L[j, j]
            L[i, j] = nominator/denom
        return sp.csr_matrix(L)
    
