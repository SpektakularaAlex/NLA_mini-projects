import numpy as np
from scipy.linalg import qr, svd, null_space
import matplotlib.pyplot as plt


def x(t):
    return np.exp(t)*np.sin(np.pi * t)


def build_A(s, t, beta, c):
    h = t[1] - t[0]
    A = h*c * np.exp(-beta * (s[:, None] - t[None, :])**2)

    return A


n = 40
m = 40
t = np.linspace(-1, 1, n)
s = np.linspace(-1, 1, m)
beta = 100
c = 1
lamb = 0.5


A = build_A(s, t, beta, c)

x_true = x(t)

# e = 1e-1*np.random.normal(0, 1, size=m)
# b = A@x_true + e


smooth_xtrue_ornot = A@x_true 




L = (np.eye(n) - np.eye(n, k=1))[:-1, :]



def gsvd(A, L, tol=None):
    """
    GSVD-like factorization of the matrix pair (A,L),

        A : m x n,   m >= n
        L : p x n,

    assuming the stacked matrix [A; L] has full column rank.

    Computes

        A = U @ C @ Winv
        L = V @ S @ Winv

    where

        U    : m x m       orthogonal
        V    : p x p       orthogonal
        C    : m x n       rectangular diagonal
        S    : p x n       rectangular diagonal
        W    : n x n       nonsingular
        Winv : n x n       inverse of W

    and

        C.T @ C + S.T @ S = I_n.

    Parameters
    ----------
    A : ndarray, shape (m,n)
    L : ndarray, shape (p,n)
    tol : float, optional
        Numerical tolerance.

    Returns
    -------
    U, V, C, S, W, Winv
    """

    A = np.asarray(A, dtype=float)
    L = np.asarray(L, dtype=float)

    m, n = A.shape
    p, n2 = L.shape

    if m < n:
        raise ValueError("A must satisfy m >= n.")

    if n2 != n:
        raise ValueError(
            "A and L must have the same number of columns."
        )

    # ---------------------------------------------------------
    # 1. QR factorization of the stacked matrix
    #
    #        [ A ; L ] = Q R
    #
    # where
    #
    #        Q : (m+p) x n
    #        R : n x n
    #
    # and Q has orthonormal columns.
    # ---------------------------------------------------------

    Z = np.vstack((A, L))

    Q, R = qr(Z, mode='economic')

    if np.linalg.matrix_rank(R) < n:
        raise ValueError(
            "The stacked matrix [A; L] must have full column rank."
        )

    Q1 = Q[:m, :]       # m x n
    Q2 = Q[m:, :]       # p x n

    # We have
    #
    #     Q1.T @ Q1 + Q2.T @ Q2 = I_n.

    # ---------------------------------------------------------
    # 2. Full SVD of Q2
    #
    #        Q2 = V S X^T
    #
    # where
    #
    #        V : p x p
    #        S : p x n
    #        X : n x n
    # ---------------------------------------------------------

    V, s, Xt = svd(Q2, full_matrices=True)

    X = Xt.T

    r = min(p, n)

    # ---------------------------------------------------------
    # 3. Construct the generalized sines and cosines.
    #
    # The SVD gives r = min(p,n) singular values.
    # If p < n, the remaining generalized sines are zero.
    #
    # Since
    #
    #   Q1.T Q1 + Q2.T Q2 = I,
    #
    # we have
    #
    #   (Q1 X).T (Q1 X) = I - S.T S.
    #
    # Thus
    #
    #        c_j^2 + s_j^2 = 1.
    # ---------------------------------------------------------

    s_full = np.zeros(n)
    s_full[:r] = s

    c = np.sqrt(np.maximum(0.0, 1.0 - s_full**2))

    # ---------------------------------------------------------
    # 4. Construct the first n columns of U from
    #
    #        Q1 X = U1 diag(c).
    #
    # For c_j > 0,
    #
    #        u_j = Q1 X e_j / c_j.
    #
    # If c_j = 0, the corresponding column is obtained by
    # orthogonal completion.
    # ---------------------------------------------------------

    Q1X = Q1 @ X

    if tol is None:
        tol = (
            100
            * np.finfo(float).eps
            * max(m, n, p)
        )

    nz = c > tol
    z = ~nz

    U1 = np.zeros((m, n))

    # Columns determined by Q1 X
    U1[:, nz] = Q1X[:, nz] / c[nz]

    # Complete the columns corresponding to c_j = 0
    if np.any(z):

        Uknown = U1[:, nz]

        if Uknown.shape[1] == 0:
            Ucomp = np.eye(m)
        else:
            Ucomp = null_space(Uknown.T)

        if Ucomp.shape[1] < np.sum(z):
            raise ValueError(
                "Could not construct the required columns of U."
            )

        U1[:, z] = Ucomp[:, :np.sum(z)]

    # ---------------------------------------------------------
    # 5. Complete U1 to a full m x m orthogonal matrix U.
    # ---------------------------------------------------------

    if m > n:

        Uperp = null_space(U1.T)

        if Uperp.shape[1] != m - n:
            raise ValueError(
                "Could not complete U to an orthogonal matrix."
            )

        U = np.hstack((U1, Uperp))

    else:
        U = U1

    # ---------------------------------------------------------
    # 6. Construct the rectangular matrix C
    #
    #           [ diag(c) ]
    #       C = [         ]     m x n.
    #           [    0    ]
    # ---------------------------------------------------------

    C = np.zeros((m, n))
    C[:n, :] = np.diag(c)

    # ---------------------------------------------------------
    # 7. Construct the rectangular matrix S
    #
    # S is p x n with singular values of Q2 on its diagonal.
    # ---------------------------------------------------------

    S = np.zeros((p, n))

    for j in range(r):
        S[j, j] = s[j]

    # ---------------------------------------------------------
    # 8. Construct W.
    #
    # We have
    #
    #        Q1 X = U C
    #
    # and
    #
    #        Q2 X = V S.
    #
    # Therefore
    #
    #        Q1 = U C X^T
    #        Q2 = V S X^T.
    #
    # Since
    #
    #        A = Q1 R
    #        L = Q2 R,
    #
    # we obtain
    #
    #        A = U C X^T R
    #        L = V S X^T R.
    #
    # Thus
    #
    #        Winv = W^{-1} = X^T R.
    # ---------------------------------------------------------

    Winv = Xt @ R

    W = np.linalg.solve(Winv, np.eye(n))

    return U, V, C, S, W, Winv




U, V, C, S, W, Winv = gsvd(A, L)


# print(f"C={C}")
# print(f"S={S}")


alpha_list = np.zeros(n)
alpha_list[:min(C.shape)] = np.diag(C)   

beta_list = np.zeros(n)
beta_list[:min(S.shape)] = np.diag(S)   

print(f"alpha_list={alpha_list}")
# print(f"beta_i_list={beta_list}")


# def x_reconstructed_reg(lamb):
#     y_list_reg = np.zeros(n)
#     for i in range(len(y_list_reg)):
#         y_i = ((alpha_list[i])/(alpha_list[i]**2 + (lamb**2)*beta_list[i]**2)) * U.T[i] @ b  
#         y_list_reg[i] = y_i

#     x_recon_reg = W@y_list_reg  
#     return x_recon_reg



def x_recon(lamb, delta):

    e = delta*np.random.normal(0, 1, size=m)
    b = A@x_true + e

    y_list_reg = np.zeros(n)
    for i in range(len(y_list_reg)):
        y_i = ((alpha_list[i])/(alpha_list[i]**2 + (lamb**2)*beta_list[i]**2)) * U.T[i] @ b  
        y_list_reg[i] = y_i

    x_recon_reg = W@y_list_reg  

    y_list_unreg = np.zeros(n)
    for i in range(len(y_list_unreg)):
        y_i = ((alpha_list[i])/(alpha_list[i]**2)) * U.T[i] @ b  
        y_list_unreg[i] = y_i

    x_recon_unreg = W@y_list_unreg

    return x_recon_reg, x_recon_unreg


# print(f"y_list_reg = {y_list_reg}")
# print(f"y_list_unreg = {y_list_unreg}")

x_recon_reg, x_recon_unreg = x_recon(lamb, 1e-1)

# x_recon_reg = x_reconstructed_reg(lamb)
# x_recon_unreg = W@y_list_unreg

# print(f"W={W}")

# print(f"x_recon_reg={x_recon_reg}")
print(f"x_recon_unreg={x_recon_unreg}")



def plot_5_1(x_true, smooth_xtrue_ornot, b, x_recon_reg, x_recon_unreg, t):
    plt.plot(t, x_true, color="red", label="x_true")
    plt.plot(t, smooth_xtrue_ornot, color="blue", label="b' (Observed signal without noice)")
    plt.plot(t, b, color="green", label="b (Noisy observed signal)")
    plt.plot(t, x_recon_reg, color="pink", label="reconstructed regularized signal")
    plt.plot(t, x_recon_unreg, color="orange", label="reconstructed unregularized signal")
    plt.grid(True)
    plt.legend()
    # plt.axis('equal')
    plt.xlabel("t")
    plt.ylabel("Signal")
    plt.title("Reconstructed, observed and true signal")
    plt.show()


# plot_5_1(x_true, smooth_xtrue_ornot, b, x_recon_reg, x_recon_unreg, t)



"<-------------------------------------------------- 5.2 -------------------------------------------------->"

def Phi(lamb, i):
    phi_i = ((alpha_list[i]**2 )/(alpha_list[i]**2 + (lamb**2)*beta_list[i]**2))
    return phi_i
# print(f"Phi täljare={alpha_list[i]**2}")
# print(f"Phi nämnare={alpha_list[0]**2 + (lamb**2)*beta_list[0]**2}")
phi_list = np.zeros(n)


def Phi_plotter(lamb):

    for j in range(len(lamb)):
        for i in range(n):
            phi_list[i] = Phi(lamb[j], i)
        plt.plot(t, phi_list, label=f"lambda={lamb[j]}")
    plt.xlabel("t")
    plt.ylabel("Dimensionless filter factor")
    plt.legend()
    plt.grid(True)
    plt.show()


lamb = np.linspace(1.62e-1, 1.62e-1*6, 6)

# Phi_plotter(lamb)




def Plot_recon_reg_sig_ov_Lamba(lamb_list):
    for j in range(len(lamb)):
        x_rec_reg, x_recon_unreg = x_recon(lamb[j], 1e-1)
        plt.plot(t, x_rec_reg, label=f"lambda={lamb[j]}")
    plt.plot(t, x_true, ".", label="x_true", color="black")
    plt.xlabel("t")
    plt.ylabel("Dimensionless filter factor")
    plt.legend()
    plt.grid(True)
    # plt.axis('equal')
    plt.show()

Plot_recon_reg_sig_ov_Lamba(lamb)



"<-------------------------------------------------- 5.4 -------------------------------------------------->"


