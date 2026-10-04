"""
02 - Matrices as linear maps, the four fundamental subspaces, PageRank
Calafiore & El Ghaoui, "Optimization Models", Ch. 3

ML / optimization takeaways
  * rank / null space tell you whether a linear model is identifiable.
  * A matrix maps the unit circle to an ellipse (the SVD is waiting in Ch. 5).
  * PageRank = dominant eigenvector of a stochastic matrix, found by power iteration.
    The convergence rate is |lambda_2 / lambda_1|, which damping controls.
"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)

# %% [1] Four fundamental subspaces of a rank-deficient matrix
# A 5x4 matrix of rank 2: two "real" features plus two redundant ones
B = rng.standard_normal((5, 2))
A = np.column_stack([B[:, 0], B[:, 1], B[:, 0] + B[:, 1], 2 * B[:, 0]])
U, s, Vt = np.linalg.svd(A)
r = int(np.sum(s > 1e-10 * s[0]))
print("[1] singular values:", s, " rank =", r)

range_A = U[:, :r]   # R(A)   in R^5, dim r
null_AT = U[:, r:]   # N(A^T) in R^5, dim m - r
range_AT = Vt[:r].T  # R(A^T) in R^4, dim r
null_A = Vt[r:].T    # N(A)   in R^4, dim n - r
print("    dims: R(A)=%d N(A^T)=%d R(A^T)=%d N(A)=%d"
      % (range_A.shape[1], null_AT.shape[1], range_AT.shape[1], null_A.shape[1]))
print("    A @ N(A) ~ 0      :", np.allclose(A @ null_A, 0))
print("    R(A^T) ⟂ N(A)     :", np.allclose(range_AT.T @ null_A, 0))
# Consequence for regression: y = A w has infinitely many solutions w + N(A).
w = rng.standard_normal(4)
w_alt = w + null_A @ rng.standard_normal(null_A.shape[1])
print("    different weights, same predictions:", np.allclose(A @ w, A @ w_alt))

# %% [2] A matrix maps the unit circle to an ellipse
M = np.array([[2.0, 1.0], [0.5, 1.0]])
t = np.linspace(0, 2 * np.pi, 300)
circle = np.vstack([np.cos(t), np.sin(t)])
ellipse = M @ circle
Um, sm, Vmt = np.linalg.svd(M)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(*circle, label="unit circle")
ax.plot(*ellipse, label="M @ circle")
for k in range(2):
    ax.arrow(0, 0, *Vmt[k], color="C0", width=0.02, length_includes_head=True)
    ax.arrow(0, 0, *(sm[k] * Um[:, k]), color="C1", width=0.02, length_includes_head=True)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
ax.set_title(r"$M v_i = \sigma_i u_i$: axes of the ellipse are $\sigma_i u_i$")
fig.tight_layout(); fig.savefig(FIG / "02_linear_map.png", dpi=120)
print("\n[2] ellipse semi-axes =", sm, "| area scale |det M| =", abs(np.linalg.det(M)),
      "= prod(sigma) =", np.prod(sm))

# %% [3] PageRank via power iteration
# Small web: edge i -> j means page i links to page j. Page 5 is a dangling node.
edges = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2), (4, 3), (4, 2), (1, 4)]
n = 6
L = np.zeros((n, n))
for i, j in edges:
    L[j, i] = 1.0
out_deg = L.sum(axis=0)
P = np.where(out_deg > 0, L / np.where(out_deg == 0, 1, out_deg), 1.0 / n)  # dangling -> uniform
assert np.allclose(P.sum(axis=0), 1)  # column-stochastic

def google_matrix(alpha):
    return alpha * P + (1 - alpha) / n * np.ones((n, n))

def power_iteration(G, iters=100):
    x = np.ones(n) / n
    hist = []
    for _ in range(iters):
        x = G @ x
        x /= x.sum()
        hist.append(x)
    return x, np.array(hist)

alpha = 0.85
G = google_matrix(alpha)
x_pi, hist = power_iteration(G)
evals, evecs = np.linalg.eig(G)
k = np.argmax(evals.real)
x_eig = np.abs(evecs[:, k].real); x_eig /= x_eig.sum()
print("\n[3] PageRank (power iteration):", x_pi)
print("    PageRank (eigenvector)     :", x_eig, " match:", np.allclose(x_pi, x_eig))
print("    ranking:", np.argsort(-x_pi))

fig, ax = plt.subplots(figsize=(6, 4))
for a_ in [0.5, 0.85, 0.99]:
    Ga = google_matrix(a_)
    ev = np.sort(np.abs(np.linalg.eigvals(Ga)))[::-1]
    xa_true = power_iteration(Ga, 3000)[0]
    _, h = power_iteration(Ga, 80)
    ax.semilogy(np.linalg.norm(h - xa_true, 1, axis=1),
                label=rf"$\alpha$={a_}, $|\lambda_2|$={ev[1]:.2f}")
ax.set_xlabel("iteration"); ax.set_ylabel(r"$\|x_k - x^*\|_1$"); ax.grid(alpha=0.3)
ax.legend(); ax.set_title(r"Power iteration converges like $|\lambda_2|^k \leq \alpha^k$")
fig.tight_layout(); fig.savefig(FIG / "02_pagerank_convergence.png", dpi=120)

# %% Interview check
"""
1. State rank-nullity. If A is 100x300, what is the minimum dim of N(A)? (200)
2. Prove R(A) ⟂ N(A^T). Use it to say when Ax = b is solvable.
3. Why does a column-stochastic matrix have eigenvalue 1? (1^T P = 1^T.)
   Why is |lambda| <= 1 for all others? (||P||_1 = 1.)
4. Why does damping make PageRank unique, and why does it speed convergence?
   (Perron-Frobenius on a positive matrix; |lambda_2| <= alpha.)
5. Cost of one power-iteration step for a sparse web graph? (O(#links), not O(n^2).)
"""
