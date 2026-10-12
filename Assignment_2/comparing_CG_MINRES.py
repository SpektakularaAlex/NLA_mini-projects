import numpy as np
from scipy.linalg import qr, svd, null_space
import matplotlib.pyplot as plt
from scipy import sparse as sp
from PIL import Image
from scipy.linalg import solve_triangular

from MINRES import MINRES
from CG import CG_mine
from MINRES import filter_epsilon
from MINRES import build_A





def plot_images_compare(original_image, blurred_image, A_sparse, L_G_sparse, b_signal):

    AtA = (A_sp.T@A_sp).tocsr()
    M = sp.kron(AtA, AtA).tocsr() + lamb**2 * (L_G_sp.T @ L_G_sp)
    c_vectorised = (A_sp.T @ b @ A_sp).ravel()

    recon_MINRES, m_iter_try_MINRES, relres_hist_try = MINRES(M, c_vectorised, np.zeros(300*300), tol, max_it_MINRES)
    
    recon_CG, TF, it_count, norm_res = CG_mine(A_sp, L_G_sp, b_signal, lamb, np.zeros(300*300), tol, max_it_CG)
    
    # images = [original_image, blurred_image, recon_MINRES, recon_CG]
    # titles = ["Original Image", "Blurred Image", f"Reconstructed image with MINRES with m_iter={m_iter_try_MINRES}", f"Reconstructed image with CG with it_count={it_count}"]

    # fig, axes = plt.subplots(2, 2, figsize=(15, 5))
    # for ax, img, title in zip(axes, images, titles):
    #     ax.imshow(img, cmap="gray")
    #     ax.set_title(title)
    #     ax.axis("off")

    # plt.tight_layout()
    # plt.show()
    images = [original_image, blurred_image, recon_MINRES.reshape(d, d), recon_CG]
    titles = ["Original image",
                "Blurred image",
                f"MINRES reconstruction\n(m_iter = {m_iter_try_MINRES})",
                f"CG reconstruction\n(it_count = {it_count})"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    "<---------------------- Original Image ------------------------->"
    "<--------------------------------------------------------------->"
    image = Image.open("img-1.png").convert("L")
    new_size = 300
    image = image.resize((new_size, new_size))

    im = np.asarray(image, dtype=float) / 255.0
    org_im = im
    "<--------------------------------------------------------------->"
    "<--------------------------------------------------------------->"


    "<------------------ Variables and constants -------------------->"
    "<--------------------------------------------------------------->"
    beta = 5*10**4
    c = 1
    lamb = 1.6681005372000592e-4
    # lamb = 0.5
    sigma = 1e-5
    d = 300
    epsilon = 1e-6
    tol = 1e-4
    max_it = 100
    max_it_MINRES = 100
    max_it_CG = 100
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

    "<--------------------------------------------------------------->"
    "<--------------------------------------------------------------->"

    plot_images_compare(org_im, blurred_im, A_sp, L_G_sp, b)