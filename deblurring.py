"""
1D deblurring with Tikhonov regularisation via the GSVD of (A, L).

    min_x ||A x - b||^2 + lambda^2 ||L x||^2,     b = A x_true + delta * N(0, 1)

With A = U C W^{-1}, L = V S W^{-1} (see gsvd.py), the solution is

    x_lambda = W y,   y_i = alpha_i / (alpha_i^2 + lambda^2 beta_i^2) * u_i^T b
             = sum_i phi_i(lambda) * (u_i^T b / alpha_i) * w_i

with filter factors phi_i = alpha_i^2 / (alpha_i^2 + lambda^2 beta_i^2).
lambda = 0 gives the unregularised least-squares solution.
"""
import numpy as np
import matplotlib.pyplot as plt

from gsvd import gsvd


# ------------------------------------------------------------------ problem setup

def x_func(t):
    return np.exp(t) * np.sin(np.pi * t)


def box_grid(a, b, n):
    """Midpoints of n equal boxes on [a, b] and the box width h."""
    h = (b - a) / n
    return a + (np.arange(n) + 0.5) * h, h


def build_A(s, t, h, kernel_B, kernel_C=1.0):
    """Box-rule discretisation of the Gaussian-kernel blur: A_ij = h * K(s_i, t_j)."""
    return h * kernel_C * np.exp(-kernel_B * (s[:, None] - t[None, :]) ** 2)


def first_difference(n):
    """(n-1) x n forward-difference approximation of d/dt (unscaled)."""
    return (np.eye(n) - np.eye(n, k=1))[:-1, :]


class DeblurProblem:
    """Holds A, L, the true signal and the GSVD, and solves for any (lambda, b)."""

    def __init__(self, n=40, a=-1.0, b=1.0, kernel_B=100.0, kernel_C=1.0):
        self.n = n
        self.t, h = box_grid(a, b, n)
        self.A = build_A(self.t, self.t, h, kernel_B, kernel_C)
        self.L = first_difference(n)
        self.x_true = x_func(self.t)
        self.b_exact = self.A @ self.x_true

        U, V, C, S, W, Winv = gsvd(self.A, self.L)
        self.U, self.W = U, W
        # Generalised singular value pairs (alpha_i, beta_i), padded to length n.
        # beta_n = 0 because L has a null space (constants) -> never filtered.
        self.alpha = np.zeros(n)
        self.alpha[:min(C.shape)] = np.diag(C)
        self.beta = np.zeros(n)
        self.beta[:min(S.shape)] = np.diag(S)

    def noisy_data(self, delta, z):
        """b = A x_true + delta * z, with z ~ N(0, 1) drawn by the caller."""
        return self.b_exact + delta * z

    def filter_factors(self, lamb):
        return self.alpha**2 / (self.alpha**2 + lamb**2 * self.beta**2)

    def solve(self, b, lamb):
        """Tikhonov solution for regularisation parameter lamb (lamb=0: unregularised)."""
        Utb = self.U[:, :self.n].T @ b
        y = self.filter_factors(lamb) * Utb / self.alpha
        return self.W @ y

    def rel_error(self, x):
        return np.linalg.norm(x - self.x_true) / np.linalg.norm(self.x_true)


# ------------------------------------------------------------------ plots

def plot_signals(prob, b, lamb):
    """5.1: true signal, blurred data and both reconstructions in one figure."""
    x_reg = prob.solve(b, lamb)
    x_unreg = prob.solve(b, 0.0)
    t = prob.t

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, prob.x_true, color="black", lw=2, label="x_true")
    ax.plot(t, prob.b_exact, color="tab:gray", ls="--", label="A x_true (blurred, no noise)")
    ax.plot(t, b, ".", color="tab:gray", label="b (blurred + noise)")
    ax.plot(t, x_unreg, color="tab:orange",
            label=f"unregularised (rel. err {prob.rel_error(x_unreg):.2g})")
    ax.plot(t, x_reg, color="tab:blue",
            label=f"regularised, λ={lamb:g} (rel. err {prob.rel_error(x_reg):.2g})")
    ax.set(xlabel="t", ylabel="signal", title="True, observed and reconstructed signals")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_filter_factors(prob, lambdas):
    """5.2: phi_i(lambda) against the generalised singular values gamma_i = alpha_i / beta_i."""
    finite = prob.beta > 0                       # gamma = inf for the null space of L
    gamma = prob.alpha[finite] / prob.beta[finite]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lambdas)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for lamb, col in zip(lambdas, colors):
        ax.semilogx(gamma, prob.filter_factors(lamb)[finite], "o-", ms=3, color=col,
                    label=f"λ={lamb:.2g}")
    ax.set(xlabel=r"generalised singular value $\gamma_i = \alpha_i/\beta_i$",
           ylabel=r"filter factor $\varphi_i(\lambda)$", title="Generalised filter factors")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_reg_over_lambda(prob, b, lambdas):
    """5.3: regularised reconstructions for several lambda, same noisy data b."""
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lambdas)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for lamb, col in zip(lambdas, colors):
        x_reg = prob.solve(b, lamb)
        ax.plot(prob.t, x_reg, color=col,
                label=f"λ={lamb:.2g} (err {prob.rel_error(x_reg):.2g})")
    ax.plot(prob.t, prob.x_true, "k--", lw=2, label="x_true")
    ax.set(xlabel="t", ylabel="signal", title="Regularised reconstructions for varying λ")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_reg_vs_unreg_over_delta(prob, deltas, lamb, z):
    """
    5.4: unregularised (left, oranges) and regularised (right, blues) reconstructions
    side by side for several noise levels delta at a fixed lambda.
    Darker shade = larger delta. The same noise direction z is scaled by each delta,
    so only the noise *level* changes between curves.
    """
    deltas = np.sort(np.asarray(deltas, dtype=float))
    shades = np.linspace(0.35, 0.95, len(deltas))
    unreg_colors = plt.cm.Oranges(shades)
    reg_colors = plt.cm.Blues(shades)

    fig, (ax_u, ax_r) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    for delta, cu, cr in zip(deltas, unreg_colors, reg_colors):
        b = prob.noisy_data(delta, z)
        x_unreg = prob.solve(b, 0.0)
        x_reg = prob.solve(b, lamb)
        ax_u.plot(prob.t, x_unreg, color=cu,
                  label=f"δ={delta:g} (err {prob.rel_error(x_unreg):.2g})")
        ax_r.plot(prob.t, x_reg, color=cr,
                  label=f"δ={delta:g} (err {prob.rel_error(x_reg):.2g})")

    for ax, title in [(ax_u, "Unregularised (λ = 0)"),
                      (ax_r, f"Regularised (λ = {lamb:g})")]:
        ax.plot(prob.t, prob.x_true, "k--", lw=1.5, label="x_true")
        ax.set(xlabel="t", ylabel="signal", title=title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Reconstructions for increasing noise level δ (darker = more noise)")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ experiments

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    prob = DeblurProblem(n=40, kernel_B=100.0)
    z = rng.standard_normal(prob.n)          # one fixed noise realisation

    delta = 1e-2
    lamb = 0.14
    b = prob.noisy_data(delta, z)

    # 5.1
    plot_signals(prob, b, lamb)

    # 5.2
    plot_filter_factors(prob, np.logspace(-3, 1, 9))

    # 5.3
    plot_reg_over_lambda(prob, b, np.logspace(-3, 1, 9))

    # 5.4
    plot_reg_vs_unreg_over_delta(prob, [1e-4, 1e-3, 1e-2, 5e-2, 1e-1], lamb, z)

    plt.show()
