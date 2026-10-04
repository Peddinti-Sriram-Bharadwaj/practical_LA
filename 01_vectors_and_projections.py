"""
01 - Vectors, norms, inner products, projections
Calafiore & El Ghaoui, "Optimization Models", Ch. 2

Run as a script (`python 01_vectors_and_projections.py`) or cell-by-cell
in VS Code / Jupyter (the `# %%` markers are cells).

ML / optimization takeaways
  * Cosine similarity = normalized inner product -> document / embedding search.
  * Projection onto a subspace IS least squares (residual is orthogonal).
  * The shape of a norm ball explains why L1 regularization gives sparse solutions.
  * Classical Gram-Schmidt loses orthogonality in floating point; QR (Householder) does not.
"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)

# %% [1] Cosine similarity of documents (bag-of-words), book Example 2.x
docs = [
    "gradient descent minimizes a convex loss",
    "stochastic gradient descent for convex loss minimization",
    "the cat sat on the mat",
    "a cat and a dog sat together",
]
vocab = sorted({w for d in docs for w in d.split()})
X = np.array([[d.split().count(w) for w in vocab] for d in docs], dtype=float)

Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
S = Xn @ Xn.T  # Gram matrix of unit vectors = cosine similarities
print("[1] cosine similarity between documents:\n", S)
# docs 0/1 (optimization) and 2/3 (cats) cluster; docs with no shared words are orthogonal.
# Note 0 and 3 still score 0.27 only because they share "a": why stop-word removal / TF-IDF exists.

# %% [2] Cauchy-Schwarz and Hölder: |x^T y| <= ||x||_p ||y||_q, 1/p + 1/q = 1
x, y = rng.standard_normal(5), rng.standard_normal(5)
print("\n[2] |x.y| =", abs(x @ y))
print("    ||x||_2 ||y||_2   =", np.linalg.norm(x) * np.linalg.norm(y))
print("    ||x||_1 ||y||_inf =", np.linalg.norm(x, 1) * np.linalg.norm(y, np.inf))
# Dual norm of L1 is Linf: this is why the subgradient of ||x||_1 lives in [-1, 1]^n

# %% [3] Projection onto a line and onto a subspace
a = np.array([2.0, 1.0])
b = np.array([1.0, 3.0])
p = (a @ b) / (a @ a) * a  # projection of b onto span(a)
print("\n[3] projection onto line:", p, " residual . a =", (b - p) @ a)

A = rng.standard_normal((6, 2))  # subspace = range(A) in R^6
b6 = rng.standard_normal(6)
P = A @ np.linalg.solve(A.T @ A, A.T)  # P = A (A^T A)^-1 A^T
Q, _ = np.linalg.qr(A)
print("    P idempotent:", np.allclose(P @ P, P), " symmetric:", np.allclose(P, P.T))
print("    P == Q Q^T  :", np.allclose(P, Q @ Q.T))
# Least squares gives the same point: A x* = P b
x_ls, *_ = np.linalg.lstsq(A, b6, rcond=None)
print("    A x_ls == P b:", np.allclose(A @ x_ls, P @ b6),
      " residual ⟂ range(A):", np.allclose(A.T @ (b6 - A @ x_ls), 0))

# %% [4] Unit balls of L1, L2, Linf and why L1 promotes sparsity
t = np.linspace(0, 2 * np.pi, 400)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
ax = axes[0]
ax.plot(np.cos(t), np.sin(t), label=r"$\|x\|_2 = 1$")
ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], label=r"$\|x\|_1 = 1$")
ax.plot([1, -1, -1, 1, 1], [1, 1, -1, -1, 1], label=r"$\|x\|_\infty = 1$")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(); ax.set_title("Unit norm balls")

# Minimize ||x|| subject to a single linear constraint c^T x = 1:
# the L1 solution sits on a vertex (sparse), the L2 solution does not.
c = np.array([1.0, 0.4])
x_l2 = c / (c @ c)                       # min-norm solution
x_l1 = np.zeros(2); i = np.argmax(abs(c)); x_l1[i] = 1 / c[i]  # vertex of L1 ball
ax = axes[1]
xs = np.linspace(-1.5, 2.5, 10)
ax.plot(xs, (1 - c[0] * xs) / c[1], "k--", label=r"$c^T x = 1$")
r2, r1 = np.linalg.norm(x_l2), np.linalg.norm(x_l1, 1)
ax.plot(r2 * np.cos(t), r2 * np.sin(t), label=f"L2 ball, x*={np.round(x_l2, 2)}")
ax.plot(r1 * np.array([1, 0, -1, 0, 1]), r1 * np.array([0, 1, 0, -1, 0]),
        label=f"L1 ball, x*={np.round(x_l1, 2)} (sparse)")
ax.plot(*x_l2, "o", c="C0"); ax.plot(*x_l1, "o", c="C1")
ax.set_xlim(-1.5, 2.5); ax.set_ylim(-1.5, 3); ax.set_aspect("equal")
ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title("Smallest ball touching the constraint")
fig.tight_layout(); fig.savefig(FIG / "01_norm_balls.png", dpi=120)
print("\n[4] L2 solution", x_l2, " L1 solution", x_l1, "-> saved figures/01_norm_balls.png")

# %% [5] Gram-Schmidt: classical vs modified vs Householder QR
def gram_schmidt(A, modified):
    A = A.astype(float).copy()
    m, n = A.shape
    Q = np.zeros((m, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            # classical projects the ORIGINAL column, modified projects the updated v
            coeff = Q[:, i] @ (v if modified else A[:, j])
            v -= coeff * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return Q

n = 10
H = 1.0 / (np.arange(n)[:, None] + np.arange(n)[None, :] + 1)  # Hilbert, kappa ~ 1e13
print(f"\n[5] Hilbert({n}) condition number: {np.linalg.cond(H):.2e}")
for name, Qh in [("classical GS", gram_schmidt(H, False)),
                 ("modified GS ", gram_schmidt(H, True)),
                 ("Householder ", np.linalg.qr(H)[0])]:
    print(f"    {name}: ||Q^T Q - I|| = {np.linalg.norm(Qh.T @ Qh - np.eye(n)):.2e}")

# %% Interview check
"""
1. Why is P = A(A^T A)^{-1}A^T symmetric and idempotent? What are its eigenvalues?
   (0 and 1; multiplicity of 1 = rank A.)
2. Show that the projection minimizes ||b - Ax||_2 using the Pythagorean theorem.
3. What is the dual norm of ||.||_1? Of ||.||_2? Prove ||x||_inf <= ||x||_2 <= sqrt(n)||x||_inf.
4. Why does Lasso (L1) give sparse weights but Ridge (L2) doesn't? (Picture in [4].)
5. Why does classical Gram-Schmidt lose orthogonality? (Cancellation errors in the
   projection coefficients get amplified by the condition number.)
"""
