import numpy as np
from scipy.linalg import qr, svd, null_space
import matplotlib.pyplot as plt
from scipy import sparse as sp
from PIL import Image
from scipy.linalg import solve_triangular

image = Image.open("img-1.png").convert("L")
new_size = 300
image = image.resize((new_size, new_size))

im = np.asarray(image, dtype=float) / 255.0
org_im = im


def G_rot_real(a, b):
    r = np.sqrt(a**2 + b**2)
    c = a / r
    s = b / r
    return np.array([[c, s], [-s, c]])

def MINRES_mine(A_symmetric, vectorised_b, x0, tolerance=1e-3, max_it=100):


    r0 = vectorised_b - A_symmetric @ x0
    v0 = 0
    v1 = r0 / np.linalg.norm(r0)

    relres = 1
    m = 1

    beta_0 = 0

    v_m_minus = v0
    v_m = v1
    beta_m_minus = beta_0
    #The Lanczos process:
    G_m_2 = np.eye(2)       
    G_m_1 = np.eye(2)     
    while relres > tolerance and m <= max_it:
        w = A_symmetric @ v_m - beta_m_minus * v_m_minus
        alpha_m = v_m.T @ w
        w = w - alpha_m * v_m
        beta_m = np.linalg.norm(w)

        if beta_m == 0:
            print(f"Happy ending")
            break

        v_m_minus = v_m
        v_m = w / beta_m
        beta_m_minus = beta_m

        #Ok, this hopwfullt gives an orthonormal base we can use to find the solution to y using MINRES
        beta_m_stored_for_lanczos = beta_m
        
        
        # #G_m-2:
        # G_m_2 = G_rot_real(0, beta_m_minus)
        # vect_rot = G_m_2@np.array([0, beta_m_minus])
        # gamma_m_2 = vect_rot[0]
        # beta_m_minus_new = vect_rot[1]

        # #G_m-1:
        # G_m_1 = G_rot_real(beta_m_minus_new, alpha_m)
        # vect_rot = G_m_1@np.array([beta_m_minus_new, alpha_m])
        # beta_m_minus_new = vect_rot[0]
        # alpha_m_new = vect_rot[1]    



        # Apply the STORED G_{m-2} to rows (m-2, m-1) of the new column: (0, beta_{m-1})
        vect_rot = G_m_2 @ np.array([0.0, beta_m_minus])
        gamma_m_2 = vect_rot[0]
        beta_m_minus_new = vect_rot[1]

        # Apply the STORED G_{m-1} to rows (m-1, m): (beta_{m-1}, alpha_m)
        vect_rot = G_m_1 @ np.array([beta_m_minus_new, alpha_m])
        beta_m_minus_new = vect_rot[0]
        alpha_m_new = vect_rot[1]
        
        #G_m:
        G_m = G_rot_real(alpha_m_new, beta_m)
        vect_rot = G_m@np.array([alpha_m_new, beta_m])
        alpha_m_new = vect_rot[0]
        beta_m = vect_rot[1]

        G_m_2 = G_m_1        # G_{m-1} becomes next iteration's G_{m-2}
        G_m_1 = G_m 



        m += 1





def MINRES(A_symmetric, vectorised_b, x0, tolerance=1e-3, max_it=100):
    r0 = vectorised_b - A_symmetric @ x0
    rho0 = np.linalg.norm(r0)
    b_norm = np.linalg.norm(vectorised_b)

    v_m_minus = np.zeros_like(r0)
    v_m = r0 / rho0
    beta_m_minus = 0.0

    G_m_2 = np.eye(2)            # "no rotation yet"
    G_m_1 = np.eye(2)

    V = []                       # all Lanczos vectors (needed for x = x0 + V y)
    R = np.zeros((max_it, max_it))
    g = np.zeros(max_it + 1)
    g[0] = rho0                  # right-hand side rho0 * e1
    relres = rho0 / b_norm
    relres_hist = [relres]
    m = 1

    while relres > tolerance and m <= max_it:
        # --- Lanczos ---
        V.append(v_m)
        w = A_symmetric @ v_m - beta_m_minus * v_m_minus
        alpha_m = v_m @ w
        w = w - alpha_m * v_m
        beta_m = np.linalg.norm(w)

        # --- STORED rotations on the new column (0, beta_{m-1}, alpha_m) ---
        gamma_m_2, beta_m_minus_new = G_m_2 @ np.array([0.0, beta_m_minus])
        beta_m_minus_new, alpha_m_new = G_m_1 @ np.array([beta_m_minus_new, alpha_m])

        # --- the one NEW rotation: kills beta_m ---
        G_m = G_rot_real(alpha_m_new, beta_m)
        alpha_m_new = (G_m @ np.array([alpha_m_new, beta_m]))[0]

        # --- store column m of R (0-based column index j) ---
        j = m - 1
        R[j, j] = alpha_m_new
        if j >= 1: R[j-1, j] = beta_m_minus_new
        if j >= 2: R[j-2, j] = gamma_m_2

        # --- rotate the right-hand side on rows (m, m+1) ---
        g[j], g[j+1] = G_m @ np.array([g[j], 0.0])
        relres = abs(g[j+1]) / b_norm        # = rho0*|tau_m| / ||b||
        relres_hist.append(relres)

        if beta_m == 0:                      # happy breakdown: exact solution
            break

        # --- shift for the next iteration (at the END) ---
        G_m_2, G_m_1 = G_m_1, G_m
        v_m_minus, v_m = v_m, w / beta_m
        beta_m_minus = beta_m
        m += 1

    m_iter = len(V)
    y = solve_triangular(R[:m_iter, :m_iter], g[:m_iter])   # back substitution
    x = x0 + np.column_stack(V) @ y
    x = x.reshape(300, 300)
    return x, m_iter, relres_hist


def build_A(d, beta, c):
    t = np.linspace(0, 1, d)
    s = t
    h = t[1] - t[0]
    A = h * c * np.exp(-beta * (s[:, None] - t[None, :])**2)   # (d, d)
    return A


def filter_epsilon(M, epsilon):

    M = M.copy()
    M[np.abs(M) < epsilon] = 0
    return M




def plot_images_all(original_image, blurred_image, reconstructed_image, m_it):
    images = [original_image, blurred_image, reconstructed_image]
    titles = ["Original Image", "Blurred Image", f"Reconstructed image with MINRES with m_iter={m_it}"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    "<------------------ Variables and constants -------------------->"
    "<--------------------------------------------------------------->"
    beta = 10**5
    c = 1
    lamb = 1.6681005372000592e-08
    # lamb = 0.5
    sigma = 1e-5
    d = 300
    epsilon = 1e-6
    tol = 1e-3
    max_it = 100
    "<--------------------------------------------------------------->"
    "<--------------------------------------------------------------->"

    "<--------------- Now to constructsparse A and b ---------------->"
    "<--------------------------------------------------------------->"
    A = build_A(d, beta, c)
    A_trunc = filter_epsilon(A, epsilon)
    A_sp = sp.csr_matrix(A_trunc)

    e = sigma*np.random.normal(0, 1, size=(np.shape(im)))
    b = A_sp@im@(A_sp.T) + e

    blurred_im = b

    # plt.imshow(b, cmap="gray")
    # plt.title(f"Observed blurry and noisy image b")
    # plt.show()


    # b_filtered = filter_epsilon(b, epsilon)
    # plt.imshow(b_filtered, cmap="gray")
    # plt.show()
    "<--------------------------------------------------------------->"
    "<--------------------------------------------------------------->"


    "<--------------------- Now to construct L --------------------->"
    "<--------------------------------------------------------------->"
    L1 = (np.eye(d) - np.eye(d, k=1))[:-1, :]
    L1_trunc = filter_epsilon(L1, epsilon)
    L1_sp = sp.csr_matrix(L1_trunc)
    L2_sp = L1_sp

    L_G_sp = sp.vstack([
        sp.kron(sp.eye(d), L1_sp),
        sp.kron(L2_sp, sp.eye(d)),
    ]).tocsr()

    rows, cols = L_G_sp.shape
    total = rows * cols

    # plt.spy(L_G_sp)
    # plt.show()
    "<--------------------------------------------------------------->"
    "<--------------------------------------------------------------->"

    print(f"A_sp has shape: {np.shape(A_sp)}")
    print(f"A_sp.T@A_sp has shape: {np.shape(A_sp.T@A_sp)}")
    print(f"L_G_sp has shape: {np.shape(L_G_sp)}")
    print(f"L_G_sp.T@L_G_sp has shape: {np.shape(L_G_sp.T@L_G_sp)}")

    AtA = (A_sp.T@A_sp).tocsr()
    # M = AtA + lamb**2 * L_G_sp.T@L_G_sp
    # M = sp.kron(AtA, AtA).tocsr() + lamb**2 * (L_G_sp.T @ (L_G_sp @ p_m))
    M = sp.kron(AtA, AtA).tocsr() + lamb**2 * (L_G_sp.T @ L_G_sp)
    c_vectorised = (A_sp.T @ b @ A_sp).ravel()

    print(f"M has shape: {np.shape(M)}")
    print(f"c_vectorised has shape: {np.shape(c_vectorised)}")

    x_guess, m_iter_try, relres_hist_try = MINRES(M, c_vectorised, np.zeros(300*300), tol, max_it)

    # print(f"relres_hist_try={relres_hist_try}")

    print(f"x_guess has shape:{np.shape(x_guess)}")


    print(f"x_guess has shape:{np.shape(x_guess)}")

    # plt.imshow(x_guess_im, cmap="gray")
    # plt.title(f"Reconstructed image with MINRES with m_iter={m_iter_try}")
    # plt.show()
    plot_images_all(org_im, blurred_im, x_guess, m_iter_try)