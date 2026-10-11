import numpy as np
from scipy.linalg import qr, svd, null_space
import matplotlib.pyplot as plt
from scipy import sparse as sp
from PIL import Image

image = Image.open("img-1.png").convert("L")
new_size = 300
image = image.resize((new_size, new_size))

im = np.asarray(image, dtype=float) / 255.0
org_im = im

plt.imshow(im, cmap="gray")
plt.title(f"Original Image")
plt.show()


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


beta = 10000
c = 1
lamb = 0.5
sigma = 1e-5
d = 300

A = build_A(d, beta, c)

# plt.spy(A)
# plt.show()


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

plt.imshow(b, cmap="gray")
plt.title(f"Observed blurry and noisy image b")
plt.show()


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

# plt.spy(L_G_sp)
# plt.show()

# plt.spy(L_G[:900, :900], markersize=1)                 # top-left corner of the top block
# plt.show()

# plt.spy(L_G[89700:89700+900, :1200], markersize=1)     # start of the bottom block
# plt.show()


"<---------------- ptoblem 8 with CG ---------------->"

def CG_mine(A_sparse, L_G_sparse, b_signal, lamb, x0, tol=1e-3, max_iter=10):
    # print(f"b_signal has shape:{np.shape(b_signal)}")
    # print(f"A_sparse has shape:{np.shape(A_sparse)}")
    # print(f"L_G_sparse has shape:{np.shape(L_G_sparse)}")

    # c = (A_sparse.T @ b_signal).ravel()
    c = (A_sparse.T @ b_signal @ A_sparse).ravel()  
    # print(f"c has shape:{np.shape(c)}")

    d = 300


    x_old = x0
    # x_old = x0.reshape(d, d)

    r_m = c # Since we try for x0 = 0 first
    p_m = r_m

    it_counter = 0

    # print(f"p_m shape: {np.shape(p_m)}")
    for i in range(max_iter):
        P_m = p_m.reshape(d, d)
        # M_pm = (A_sparse.T@(A_sparse@P_m)).ravel() + lamb * L_G_sparse.T @ (L_G_sparse @ p_m)
        M_pm = (A_sparse.T @ (A_sparse @ P_m @ A_sparse.T) @ A_sparse).ravel() + lamb * (L_G_sparse.T @ (L_G_sparse @ p_m)) 
        alph_m = (r_m.T @ r_m) / (p_m.T @ M_pm)

        x_new = x_old + alph_m * p_m
        r_m1 = r_m - alph_m*M_pm
        # r_m1 = c - (A_sparse.T@(A_sparse@x_new) + lamb*L_G_sparse.T @ (L_G_sparse @ x_new))

        if np.linalg.norm(r_m1) / np.linalg.norm(c) <= tol:
            it_counter += 1
            return x_new.reshape(d, d), True, it_counter, np.linalg.norm(r_m1) / np.linalg.norm(c)
        
        beta_m = (r_m1.T @ r_m1) / (r_m.T @ r_m)
        p_m = r_m1 + beta_m * p_m
        x_old = x_new
        r_m = r_m1

        it_counter += 1

    return x_new.reshape(d, d), False, it_counter, np.linalg.norm(r_m1) / np.linalg.norm(c)

# lam = 10**(-4)

# x_pic_recovered_maybe, TF, it_count, relative_res = CG_mine(A_sp, L_G_sp, b, lam, np.zeros(300*300), 1e-15, 1000)

# print(f"true or false: {TF}, it_count={it_count}")
# # print(f"Testing if I can get something from my CG, x_pic_recovered_maybe={x_pic_recovered_maybe}")

# print(f"shape x_pic_recovered_maybe={np.shape(x_pic_recovered_maybe)}")

# plt.imshow(x_pic_recovered_maybe, cmap="gray")
# plt.title(f"Reconstructed Image x")
# plt.show()

def CG_over_lam(lam_list, origingal_im):
    X_rel_error_list =[]
    Rel_res_list = []
    for i in lam_list:
        X, converged, its, rel_res = CG_mine(A, L_G_sp, b, i, np.zeros(d*d), 1e-6, 500)
        # X_min_im_nomred_list.append(np.linalg.norm(X - origingal_im) / np.linalg.norm(origingal_im))
        rel_err = np.linalg.norm(X - origingal_im) / np.linalg.norm(origingal_im)
        print(f"lam={i}  rel_err={rel_err:.4f}  converged={converged}  its={its}")
        X_rel_error_list.append(rel_err)
        Rel_res_list.append(rel_res)

        # plt.imshow(X, cmap="gray")
        # plt.title(f"Reconstructed Image x with lamb={i} and relative error={rel_err}")
        # plt.show()
    plt.plot(lam_list, X_rel_error_list)
    plt.xlabel("Lambda")
    plt.ylabel("Relative error")
    plt.title(f"Relative error over lambda")
    plt.show()
    
    plt.plot(lam_list, Rel_res_list)
    plt.xlabel("Lambda")
    plt.ylabel("Relative residual")
    plt.title(f"Relative residual over lambda")
    plt.show()

# lam_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0, 1, 10]
# lam_list = np.linspace(1e-8, 5e-6, 10)
lam_list = np.logspace(-14, -7, 10)
CG_over_lam(lam_list, org_im)