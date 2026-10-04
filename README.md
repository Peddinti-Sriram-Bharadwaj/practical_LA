# Practical Linear Algebra for ML & Optimization

Runnable demonstrations of linear algebra ideas, each applied to a machine-learning or
optimization problem. The structure follows **Calafiore & El Ghaoui, _Optimization
Models_ (Cambridge, 2014)**.

Each module is a plain Python script with `# %%` cells. You can run it end-to-end, or
step through it cell by cell in VS Code / Jupyter. Every claim is **checked numerically**
(`allclose`, observed vs. theoretical rates), and every module ends with
interview-style questions.

```bash
pip install -r requirements.txt
python 03_symmetric_matrices_pca_gd.py     # prints checks, writes plots to figures/
```

## Part I: Linear algebra models (done)

| # | Module | Book | Concept → Application |
|---|--------|------|-----------------------|
| 01 | `01_vectors_and_projections.py` | Ch. 2 | inner products → cosine similarity of documents; projection = least squares; norm balls → why L1 is sparse; Gram-Schmidt stability |
| 02 | `02_matrices_and_pagerank.py` | Ch. 3 | four fundamental subspaces → identifiability; linear maps as ellipses; **PageRank** by power iteration, convergence vs. damping |
| 03 | `03_symmetric_matrices_pca_gd.py` | Ch. 4 | spectral theorem, Rayleigh quotient → **PCA** on digits; Hessian condition number → **gradient descent rate**; PSD & Cholesky jitter |
| 04 | `04_svd_low_rank.py` | Ch. 5 | Eckart-Young → **image compression**, low-rank denoising (recommenders); pseudo-inverse → min-norm solution and GD's implicit bias |
| 05 | `05_least_squares_ridge.py` | Ch. 6-7 | normal equations vs QR vs SVD and κ² blow-up; **ridge** as SVD filter, bias-variance; sensitivity to perturbations |

## Roadmap

**Part I (continued)**
- 06 Matrix algorithms (Ch. 7): power / QR iteration, conjugate gradient vs GD, randomized SVD

**Part II: Convex optimization models**
- 07 Convexity (Ch. 8): convex sets and functions, Hessian PSD test, first-order condition, Lagrange duality
- 08 LP & QP (Ch. 9): Lasso as QP, SVM (hinge loss) as QP, portfolio QP, LP relaxations
- 09 SOCP & robust models (Ch. 10): robust least squares, robust LP, norm constraints
- 10 SDP (Ch. 11): max-cut relaxation, matrix completion via nuclear norm, eigenvalue optimization
- 11 Algorithms (Ch. 12): gradient / Newton / proximal gradient (ISTA/FISTA) / ADMM from scratch

**Part III: Applications**
- 12 Learning from data (Ch. 13): logistic regression, sparse PCA, kernels
- 13 Finance (Ch. 14): Markowitz frontier, risk factor models
- 14 Control (Ch. 15): LQR via Riccati, least-squares control

**Phase 2: Probability, measure theory, stochastic optimization** (after the LA track)
- Fundamentals + puzzles (Bertsekas–Tsitsiklis), verified by Monte Carlo
- Measure-theoretic probability (Durrett PTE Ch. 1–5): convergence modes, LLN, Borel–Cantelli, CLT, martingales, Markov chains
- Stochastic optimization arising naturally: concentration (Hoeffding, sub-Gaussian), Robbins–Monro & the ODE method (Borkar), linear SA / TD(0), SGD, bandits (UCB, best-arm identification)

## Interview cheat sheet (Part I)

- `P = A(AᵀA)⁻¹Aᵀ` is symmetric and idempotent, with eigenvalues in {0, 1}; its rank equals the rank of A.
- `κ(AᵀA) = κ(A)²` is the reason to solve least squares with QR or SVD, not normal equations.
- `‖A‖₂ = σ₁`, `‖A‖_F = √Σσᵢ²`, `‖A − A_k‖₂ = σ_{k+1}` (Eckart-Young).
- GD on a quadratic: converges iff `0 < η < 2/λ_max`; the best rate is `(κ−1)/(κ+1)`.
- PageRank converges at rate `|λ₂| ≤ α` (damping factor).
- Ridge: `w = Σ σᵢ/(σᵢ²+λ) (uᵢᵀy) vᵢ`, so directions with small σ are damped.
