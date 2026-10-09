import numpy as np
from scipy.linalg import qr, svd, null_space
import matplotlib.pyplot as plt
from scipy import sparse as sp
from PIL import Image

image = Image.open("img-1.png").convert("L")  
new_size = 300
image = image.resize((new_size, new_size))

im = np.asarray(image, dtype=float) / 255.0  

# plt.imshow(im, cmap="gray")                
# plt.show()


epsilon = 0.5


# print(f"im_before = {im}")



def filter_epsilon(M, epsilon):
    
    # for i in range(len(im[0])):
    #     for j in range(len(im[0])): #Assuming square matrix
    #         a_ij = im[i][j]
    #         if np.abs(a_ij) < epsilon:
    #             im[i][j] = 0


    # print(f"im_after = {im}")
    M = M.copy()
    M[np.abs(M) < epsilon] = 0
    return M



# plt.imshow(im, cmap="gray")                
# plt.show()



def build_A_fail(d, beta, c):
    base = np.linspace(0, d, d+1)
    # T, S = np.meshgrid(base, base, indexing="ij")

    pts = np.zeros((d, d, 2))
    for i in range(d):
        for j in range(d):
            pts[i, j] = [base[i], base[j]]

    S = pts
    T = pts

    print(f"Shape S = {np.shape(S)}, Shape T = {np.shape(T)}")    

    h = T[1] - T[0]
    A = h*c * np.exp(-beta * (S[:, None] - T[None, :])**2)

    return A


def build_A(d, beta, c):
      t = np.linspace(0, 1, d)
      s = t
      h = t[1] - t[0]
      A = h * c * np.exp(-beta * (s[:, None] - t[None, :])**2)   # (d, d)
      return A


beta = 1000
c=1
lamb = 0.5
sigma = 1e-4
d = 300

A = build_A(d, beta, c)

plt.spy(A)
plt.show()

def count_nonzero_entries(twoD_matrix):
    
    non_zero_list = []
    for i in twoD_matrix:
        for j in i:
            if j != 0:
                non_zero_list.append(j)
    
    non_zero_amount = len(non_zero_list)
    return (non_zero_amount/90000)*100

print(f"Size A = {np.size(A)}, Siza im = {np.size(im)}")
print(f"Shape A = {np.shape(A)}, Shape im = {np.shape(im)}")

print(f"Non-zero percentage of A = {count_nonzero_entries(A)}")
print(f"nnz A = {(count_nonzero_entries(A)/100)*np.size(A)}")


e = sigma*np.random.normal(0, 1, size=(np.shape(im)))
b = A@im@(A.T) + e

# plt.imshow(b, cmap="gray")                
# plt.show()


b_filtered = filter_epsilon(b, 0.001)
# plt.imshow(b_filtered, cmap="gray")                
# plt.show()

A_trunc = filter_epsilon(A, 1e-8)
A_sp = sp.csr_matrix(A_trunc)



"<------- Now to construct L ------->"
L1 = (np.eye(d) - np.eye(d, k=1))[:-1, :]
# L2 = (np.eye(d) - np.eye(d, k=1))[:-1, :]

L1_trunc = filter_epsilon(L1, 1e-8)
# L2_trunc = filter_epsilon(L2, 1e-8)

L1_sp = sp.csr_matrix(L1_trunc)
L2_sp = L1_sp
# L2_sp = sparse.csr_matrix(L2_trunc)

# L_G = np.array( [np.kron(np.eye(d), L1) ], 
#                [np.kron(L2, np.eye(d)) ])

L_G_sp = sp.vstack([
    sp.kron(sp.eye(d), L1_sp),    
    sp.kron(L2_sp, sp.eye(d)),    
]).tocsr()

rows, cols = L_G_sp.shape
total = rows * cols

print(f"L_G shape = {rows, cols} and total size={total}")

print(f"Non-zero perzentage of L_G={(len(L_G_sp.data) / total)*100}")
print(f"nnz L_G = {len(L_G_sp.data)}")

plt.spy(L_G_sp)
plt.show()

# plt.spy(L_G[:900, :900], markersize=1)                 # top-left corner of the top block
# plt.show()

# plt.spy(L_G[89700:89700+900, :1200], markersize=1)     # start of the bottom block
# plt.show()



"<---------------- ptoblem 8 with CG ---------------->"

c = A_sp.T @ b


r0 = c #Since we try for x0 = 0 first
p0 = r0

x_old = 0
lamb = 1

r_m = r0
p_m = p0
for i in range(10):
    M_pm = A_sp.T@(A_sp@p_m) + lamb * L_G_sp.T @ (L_G_sp @ p_m)
    alph_m = (r_m.T @ r_m) / (p_m.T @ M_pm)

    x_new = x_old + alph_m @ p_m
    r_m1 = c - (A_sp.T@(A_sp@x_new) + lamb*L_G_sp.T @ (L_G_sp @ x_new))

    if np.norm(r_m1, 2)





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
