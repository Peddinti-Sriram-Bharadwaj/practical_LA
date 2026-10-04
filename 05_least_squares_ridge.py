"""
05 - Linear equations, least squares, conditioning, regularization
Calafiore & El Ghaoui, "Optimization Models", Ch. 6 (and Ch. 7 for the algorithms)

ML / optimization takeaways
  * Linear regression = least squares. Normal equations, QR and SVD give the same
    answer in exact arithmetic but not in floating point: kappa(A^T A) = kappa(A)^2.
  * Ridge regression shrinks along singular directions by the filter
    factor sigma^2 / (sigma^2 + lambda): a bias-variance knob.
"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)

# %% [1] Three ways to solve least squares, and how they behave as conditioning worsens
def solve_normal(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)

def solve_qr(A, b):
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b)  # R is upper triangular (back-substitution)

def solve_svd(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]

t = np.linspace(0, 1, 100)
print("[1] polynomial fit with a Vandermonde matrix; true coefficients all = 1")
print("    deg  kappa(A)    err normal   err QR       err SVD")
for deg in [3, 6, 9, 12, 15]:
    A = np.vander(t, deg + 1, increasing=True)
    w_true = np.ones(deg + 1)
    b = A @ w_true
    errs = [np.linalg.norm(f(A, b) - w_true) for f in (solve_normal, solve_qr, solve_svd)]
    print(f"    {deg:3d}  {np.linalg.cond(A):.1e}   " + "   ".join(f"{e:.1e}" for e in errs))
# Normal equations break down once kappa(A)^2 approaches 1/eps ~ 1e16.

# %% [2] Geometry: the residual is orthogonal to the columns
A = rng.standard_normal((50, 3))
b = rng.standard_normal(50)
w = solve_qr(A, b)
r = b - A @ w
print("\n[2] A^T r ~ 0:", np.allclose(A.T @ r, 0),
      f"| ||b||^2 = ||Aw||^2 + ||r||^2: {b@b:.4f} = {(A@w)@(A@w) + r@r:.4f}")

# %% [3] Ridge regression through the SVD lens
# Overfitting setup: few noisy samples, high-degree polynomial features
n_train, deg, noise = 20, 14, 0.15
f_true = lambda x: np.sin(2 * np.pi * x)
x_tr = np.sort(rng.uniform(0, 1, n_train))
x_te = np.linspace(0, 1, 300)
y_tr = f_true(x_tr) + noise * rng.standard_normal(n_train)
y_te = f_true(x_te)

def features(x):
    return np.vander(2 * x - 1, deg + 1, increasing=True)  # map to [-1, 1] for conditioning

A_tr, A_te = features(x_tr), features(x_te)
U, s, Vt = np.linalg.svd(A_tr, full_matrices=False)

def ridge(lam):
    # w = V diag(sigma / (sigma^2 + lam)) U^T y  == argmin ||Aw - y||^2 + lam ||w||^2
    return Vt.T @ ((s / (s**2 + lam)) * (U.T @ y_tr))

lam = 1e-3
w_closed = np.linalg.solve(A_tr.T @ A_tr + lam * np.eye(deg + 1), A_tr.T @ y_tr)
print("\n[3] ridge via SVD == closed form:", np.allclose(ridge(lam), w_closed))

lams = np.logspace(-10, 2, 60)
tr_err = [np.mean((A_tr @ ridge(l) - y_tr) ** 2) for l in lams]
te_err = [np.mean((A_te @ ridge(l) - y_te) ** 2) for l in lams]
best = lams[np.argmin(te_err)]
print(f"    best lambda ~ {best:.1e}, test MSE {min(te_err):.4f} "
      f"(lambda->0 test MSE {te_err[0]:.2f})")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
axes[0].loglog(lams, tr_err, label="train"); axes[0].loglog(lams, te_err, label="test")
axes[0].axvline(best, ls="--", c="gray"); axes[0].set_xlabel(r"$\lambda$")
axes[0].set_title("bias-variance trade-off"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(x_tr, y_tr, "ko", ms=4, label="data")
axes[1].plot(x_te, y_te, "k--", lw=1, label="truth")
for l in [1e-10, best, 10]:
    axes[1].plot(x_te, A_te @ ridge(l), label=rf"$\lambda$={l:.0e}")
axes[1].set_ylim(-2, 2); axes[1].legend(fontsize=8); axes[1].set_title("fits")

for l in [1e-6, best, 1e-1, 10]:
    axes[2].semilogx(s, s**2 / (s**2 + l), "o-", ms=3, label=rf"$\lambda$={l:.0e}")
axes[2].set_xlabel(r"singular value $\sigma_i$"); axes[2].set_title(r"filter $\sigma^2/(\sigma^2+\lambda)$")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "05_ridge.png", dpi=120)

# %% [4] Sensitivity: relative error in x <= kappa(A) * relative error in b
A = np.vander(np.linspace(0, 1, 8), 8, increasing=True)  # square, ill-conditioned
x = np.ones(8)
b = A @ x
db = 1e-10 * rng.standard_normal(8)
x_pert = np.linalg.solve(A, b + db)
amp = (np.linalg.norm(x_pert - x) / np.linalg.norm(x)) / (np.linalg.norm(db) / np.linalg.norm(b))
print(f"\n[4] kappa(A) = {np.linalg.cond(A):.1e}, observed error amplification = {amp:.1e}")

# %% Interview check
"""
1. Derive the normal equations by setting the gradient of ||Aw - b||^2 to zero.
   When is the solution unique? (A has full column rank.)
2. Why prefer QR/SVD over the normal equations? Show kappa(A^T A) = kappa(A)^2.
3. Write ridge in SVD form. What happens to directions with sigma_i^2 << lambda?
4. Ridge = MAP estimate with which prior? (Gaussian prior w ~ N(0, sigma^2/lambda I).)
5. Cost: QR of an m x n matrix is O(mn^2). For m = 1e6, n = 1e3, what would you do? (SGD /
   conjugate gradient / sketching.)
"""
