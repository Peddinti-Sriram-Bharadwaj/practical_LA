"""
03 - Symmetric matrices: spectral theorem, Rayleigh quotient, PCA, curvature
Calafiore & El Ghaoui, "Optimization Models", Ch. 4

ML / optimization takeaways
  * PCA = eigendecomposition of the covariance matrix (variational: Rayleigh quotient).
  * The Hessian's eigenvalues set the curvature. The condition number
    kappa = lambda_max / lambda_min sets how fast gradient descent converges.
  * PSD <-> convex quadratic; this is the bridge to Ch. 8.
"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)

# %% [1] Spectral theorem: A = Q Lambda Q^T with orthogonal Q
B = rng.standard_normal((4, 4))
A = (B + B.T) / 2
lam, Q = np.linalg.eigh(A)  # eigh: for symmetric matrices, real ascending eigenvalues
print("[1] eigenvalues:", lam)
print("    Q orthogonal:", np.allclose(Q.T @ Q, np.eye(4)),
      " A == Q diag(lam) Q^T:", np.allclose(A, Q @ np.diag(lam) @ Q.T))
print("    A == sum lam_i q_i q_i^T:",
      np.allclose(A, sum(l * np.outer(q, q) for l, q in zip(lam, Q.T))))

# %% [2] Rayleigh quotient: lambda_min <= x^T A x / x^T x <= lambda_max
X = rng.standard_normal((4, 100_000))
R = np.einsum("ij,ij->j", X, A @ X) / np.einsum("ij,ij->j", X, X)
print(f"\n[2] Rayleigh quotient over 1e5 random x: [{R.min():.4f}, {R.max():.4f}]"
      f"  vs  [lambda_min, lambda_max] = [{lam[0]:.4f}, {lam[-1]:.4f}]")
print("    attained at eigenvectors:", Q[:, -1] @ A @ Q[:, -1])

# %% [3] PCA on handwritten digits (64-dim -> 2-dim)
digits = load_digits()
D, y = digits.data, digits.target
Dc = D - D.mean(axis=0)
C = Dc.T @ Dc / (len(Dc) - 1)
evals, evecs = np.linalg.eigh(C)
evals, evecs = evals[::-1], evecs[:, ::-1]          # descending
_, s, Vt = np.linalg.svd(Dc, full_matrices=False)   # same thing via SVD
print("\n[3] eig(cov) == sigma^2/(n-1):", np.allclose(evals[:10], s[:10] ** 2 / (len(Dc) - 1)))
print("    |cos(eigvec, right sing vec)| for top 3:", np.abs(np.sum(evecs[:, :3] * Vt[:3].T, axis=0)))
explained = np.cumsum(evals) / evals.sum()
print("    components for 90% variance:", np.searchsorted(explained, 0.9) + 1, "of 64")

Z = Dc @ evecs[:, :2]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sc = axes[0].scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=6)
axes[0].set_title("Digits projected on top-2 principal directions")
fig.colorbar(sc, ax=axes[0])
axes[1].plot(np.arange(1, 65), explained, marker=".")
axes[1].axhline(0.9, ls="--", c="gray"); axes[1].grid(alpha=0.3)
axes[1].set_xlabel("k"); axes[1].set_title("Cumulative explained variance")
fig.tight_layout(); fig.savefig(FIG / "03_pca_digits.png", dpi=120)

# %% [4] Hessian spectrum controls gradient descent
# f(x) = 1/2 x^T H x, H = diag(L, mu). Optimal fixed step 2/(L+mu) gives rate (kappa-1)/(kappa+1).
def gd(H, x0, step, iters):
    xs = [x0]
    for _ in range(iters):
        xs.append(xs[-1] - step * H @ xs[-1])
    return np.array(xs)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
x0 = np.array([1.0, 1.0])
print("\n[4] gradient descent on quadratics")
for kappa in [2, 10, 100]:
    H = np.diag([float(kappa), 1.0])
    xs = gd(H, x0, 2 / (kappa + 1), 60)
    rate = (kappa - 1) / (kappa + 1)
    axes[0].semilogy(np.linalg.norm(xs, axis=1), label=rf"$\kappa$={kappa}, rate={rate:.3f}")
    print(f"    kappa={kappa:4d}: theoretical rate {rate:.3f}, "
          f"observed {np.linalg.norm(xs[40]) / np.linalg.norm(xs[39]):.3f}")
axes[0].set_title(r"GD error $\|x_k\|$ (step $2/(L+\mu)$)"); axes[0].legend(); axes[0].grid(alpha=0.3)

kappa = 10
H = np.diag([float(kappa), 1.0])
g = np.linspace(-1.2, 1.2, 200)
G1, G2 = np.meshgrid(g, g)
axes[1].contour(G1, G2, 0.5 * (kappa * G1**2 + G2**2), levels=15, cmap="Greys")
traj = gd(H, np.array([1.0, 1.0]), 1.8 / kappa, 25)
axes[1].plot(traj[:, 0], traj[:, 1], "o-", ms=3, c="C3", label="GD zig-zag (step 1.8/L)")
axes[1].set_aspect("equal"); axes[1].legend(); axes[1].set_title(r"Ill-conditioned level sets, $\kappa$=10")
fig.tight_layout(); fig.savefig(FIG / "03_gd_conditioning.png", dpi=120)

# %% [5] PSD tests: eigenvalues, Cholesky, Gram matrices
def is_psd(M, tol=1e-10):
    return np.linalg.eigvalsh(M).min() >= -tol

F = rng.standard_normal((10, 3))
K = F @ F.T  # Gram matrix: always PSD, rank <= 3
print("\n[5] Gram matrix PSD:", is_psd(K), " rank:", np.linalg.matrix_rank(K))
try:
    np.linalg.cholesky(K)
except np.linalg.LinAlgError:
    print("    Cholesky fails on PSD-but-singular K (needs strictly PD)")
np.linalg.cholesky(K + 1e-6 * np.eye(10))
print("    Cholesky works on K + eps I (the 'jitter' trick used in Gaussian processes)")

# %% Interview check
"""
1. Prove eigenvectors of a symmetric matrix for distinct eigenvalues are orthogonal.
2. Show max_{||x||=1} x^T A x = lambda_max. What is max over x ⟂ q_1? (lambda_2: Courant-Fischer.)
3. Why is the first principal component the top eigenvector of the covariance?
   Why compute PCA via SVD of the data instead of eig of X^T X? (Conditioning is squared.)
4. For f(x) = 1/2 x^T H x - b^T x, which step sizes make GD converge? (0 < eta < 2/lambda_max.)
5. Is A + B PSD if A, B are PSD? Is AB? (Yes; not necessarily, AB need not even be symmetric.)
"""
