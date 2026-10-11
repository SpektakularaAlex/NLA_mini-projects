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



def MINRES(A_symmetric, vectorised_b, x0, tolerance=1e-3, max_it=100):
    r0 = vectorised_b - A_symmetric @ x0
    v0 = 0
    v1 = r0 / np.norm(r0)

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
        beta_m = np.norm(w)

        if beta_m == 0:
            print(f"Happy ending")
            break

        v_m_minus = v_m
        v_m = w / beta_m
        beta_m_minus = beta_m

        #Ok, this hopwfullt gives an orthonormal base we can use to find the solution to y using MINRES
        beta_m_stored_for_lanczos = beta_m
        
        def G_rot_real(a, b):
            r = np.sqrt(a**2 + b**2)
            c = a/r
            s = b/r

            G = np.array([ [c, s], [-s, c]   ])
            return G
        
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
