"""
04 - Singular value decomposition and low-rank approximation
Calafiore & El Ghaoui, "Optimization Models", Ch. 5

ML / optimization takeaways
  * Eckart-Young: truncated SVD is the best rank-k approximation (2-norm and Frobenius).
  * Compression, denoising, latent factors (recommenders, LSA) are all truncated SVD.
  * Pseudo-inverse gives the minimum-norm least-squares solution.
  * Operator norm = sigma_max; condition number = sigma_max / sigma_min.
"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)

# %% [1] SVD basics and norms
A = rng.standard_normal((6, 4))
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("[1] sigma:", s)
print("    ||A||_2 == sigma_1:", np.isclose(np.linalg.norm(A, 2), s[0]))
print("    ||A||_F == sqrt(sum sigma^2):", np.isclose(np.linalg.norm(A, "fro"), np.sqrt(np.sum(s**2))))
print("    nuclear norm == sum sigma:", np.isclose(np.linalg.norm(A, "nuc"), s.sum()))
print("    eig(A^T A) == sigma^2:", np.allclose(np.sort(np.linalg.eigvalsh(A.T @ A))[::-1], s**2))

# %% [2] Eckart-Young on a real image
with cbook.get_sample_data("grace_hopper.jpg") as f:
    img = plt.imread(f).astype(float).mean(axis=2) / 255  # grayscale
U, s, Vt = np.linalg.svd(img, full_matrices=False)
m, n = img.shape

def rank_k(k):
    return (U[:, :k] * s[:k]) @ Vt[:k]

print(f"\n[2] image {m}x{n}")
ks = [5, 20, 50, 100]
fig, axes = plt.subplots(1, len(ks) + 1, figsize=(16, 4))
axes[0].imshow(img, cmap="gray"); axes[0].set_title(f"original (rank {len(s)})")
for ax, k in zip(axes[1:], ks):
    Ak = rank_k(k)
    err2, errF = np.linalg.norm(img - Ak, 2), np.linalg.norm(img - Ak, "fro")
    print(f"    k={k:3d}: ||A-A_k||_2={err2:.4f} (sigma_k+1={s[k]:.4f}) "
          f"||A-A_k||_F={errF:.3f} (tail={np.sqrt(np.sum(s[k:]**2)):.3f}) "
          f"storage {k*(m+n+1)/(m*n):.1%}")
    ax.imshow(Ak, cmap="gray"); ax.set_title(f"rank {k}")
for ax in axes:
    ax.axis("off")
fig.tight_layout(); fig.savefig(FIG / "04_svd_image.png", dpi=100)

# Any other rank-k matrix does worse, e.g. keep rank-k from a random projection:
k = 20
Qr, _ = np.linalg.qr(rng.standard_normal((n, k)))
print(f"    random rank-{k} approx error {np.linalg.norm(img - img @ Qr @ Qr.T, 'fro'):.3f}"
      f" vs SVD {np.linalg.norm(img - rank_k(k), 'fro'):.3f}")

# %% [3] Denoising a low-rank matrix (a ratings-like matrix: users x items)
users, items, true_rank = 200, 100, 5
L_true = rng.standard_normal((users, true_rank)) @ rng.standard_normal((true_rank, items))
noisy = L_true + 2.0 * rng.standard_normal((users, items))
Un, sn, Vnt = np.linalg.svd(noisy, full_matrices=False)
errs = [np.linalg.norm((Un[:, :k] * sn[:k]) @ Vnt[:k] - L_true, "fro") for k in range(1, 30)]
best = int(np.argmin(errs)) + 1
print(f"\n[3] noisy error {np.linalg.norm(noisy - L_true, 'fro'):.1f}; "
      f"best truncation k={best} -> error {min(errs):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].semilogy(sn, ".-"); axes[0].axvline(true_rank - 0.5, ls="--", c="gray")
axes[0].set_title("singular values of noisy matrix: signal | noise floor"); axes[0].grid(alpha=0.3)
axes[1].plot(range(1, 30), errs, ".-"); axes[1].axvline(true_rank, ls="--", c="gray")
axes[1].set_xlabel("truncation rank k"); axes[1].set_title(r"$\|\hat L_k - L\|_F$"); axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "04_svd_denoise.png", dpi=120)

# %% [4] Pseudo-inverse: minimum-norm solution of an underdetermined system
A = rng.standard_normal((3, 6))  # 3 equations, 6 unknowns
b = rng.standard_normal(3)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
A_pinv = Vt.T @ np.diag(1 / s) @ U.T
x_mn = A_pinv @ b
print("\n[4] pinv via SVD == np.linalg.pinv:", np.allclose(A_pinv, np.linalg.pinv(A)))
print("    A x_mn == b:", np.allclose(A @ x_mn, b))
null = np.linalg.svd(A)[2][3:].T
x_other = x_mn + null @ rng.standard_normal(3)
print(f"    ||x_mn|| = {np.linalg.norm(x_mn):.4f} < ||x_other|| = {np.linalg.norm(x_other):.4f}"
      f"  (x_mn ⟂ N(A): {np.allclose(null.T @ x_mn, 0)})")
# Gradient descent from 0 on ||Ax-b||^2 converges to x_mn: the "implicit bias" of GD.
x = np.zeros(6)
for _ in range(5000):
    x -= 0.05 * A.T @ (A @ x - b)
print("    GD from x0=0 converges to the min-norm solution:", np.allclose(x, x_mn, atol=1e-6))

# %% Interview check
"""
1. Relate the SVD of A to the eigendecompositions of A^T A and A A^T.
2. Prove ||A||_2 = sigma_1. Why is ||A - A_k||_2 = sigma_{k+1}?
3. How many numbers to store a rank-k approx of an m x n matrix? (k(m+n+1).)
4. Why does GD initialized at 0 find the min-norm solution? (Iterates stay in R(A^T) = N(A)^⟂.)
   Why does this matter for over-parameterized neural nets?
5. What does a tiny sigma_min mean for solving Ax = b? (Noise amplified by 1/sigma_min.)
"""
