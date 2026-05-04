# =============================================================================
#  IELDTM solver for the one-dimensional Burgers-Fisher equation
# -----------------------------------------------------------------------------
#  Equation:    u_t + alpha*u*u_x = epsilon*u_xx + beta*u*(1 - u),
#               x in [0, 1], t in [0, t_f]
#
#  This script implements the baseline Implicit-Explicit Local Differential
#  Transform Method (IELDTM) coupled with global-in-time Chebyshev spectral
#  collocation (ChSCM) for the 1D Burgers-Fisher equation. 
# =============================================================================

import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
import time

# -----------------------------------------------------------------------------
# 0) Parameters
# -----------------------------------------------------------------------------
# Discretization parameters
N      = 25          # Order of the Chebyshev polynomial expansion in time
M      = 20          # Number of spatial mesh intervals (M+1 nodes total)
TO     = 4           # Order of the local Taylor series expansion in space
alfa   = 0.5         # Internal collocation parameter theta in [0, 1]
                     # used for the C^0 / C^1 continuity matching points

# Physical parameters of the Burgers-Fisher equation
v      = 2**(-12)    # Diffusion coefficient (epsilon in the manuscript)
V      = 0.001       # Advection coefficient alpha
beta   = 0.001       # Reaction coefficient beta

# Constants of the analytical traveling-wave solution
# (see Eqs. (48)-(51) of the manuscript)
delta1 = -V / (4 * v)
delta2 = V / 2 + (2 * v * beta) / V

# Temporal domain [a, b] = [0, t_f]
tf     = 1.0
a, b   = 0.0, tf
factor = 2.0 / (b - a)   # Affine mapping factor from [a, b] -> [-1, 1]

# Spatial domain [xb, xf]
xb, xf = 0.0, 1.0
dx     = (xf - xb) / M   # Uniform spatial step size

# -----------------------------------------------------------------------------
# 1) Chebyshev collocation nodes in time
# -----------------------------------------------------------------------------
# Shifted Chebyshev-Gauss-Lobatto nodes on [a, b], cf. Eq. (7)
i_vals = np.arange(N + 1)
c = 0.5 * ((a + b) - (b - a) * np.cos(np.pi * i_vals / N))

# -----------------------------------------------------------------------------
# 2) Build the Chebyshev time-differentiation matrix A
# -----------------------------------------------------------------------------
# A is the (N+1) x (N+1) Chebyshev differentiation matrix (Eq. (11)).
# It maps nodal solution values u(x, t_n) to their time derivatives at the
# collocation nodes.
A = np.zeros((N + 1, N + 1))
for i_idx in range(N + 1):
    for n_idx in range(N + 1):
        # Quadrature weight P (1/N at endpoints, 2/N otherwise)
        P = 1.0 / N if (n_idx == 0 or n_idx == N) else 2.0 / N
        top = 0.0
        theta = np.pi - i_idx * np.pi / N
        for j in range(N + 1):
            if 0 < i_idx < N:
                # Interior collocation nodes
                top += (factor * j / np.sqrt(1 - np.cos(theta) ** 2)) \
                       * np.sin(j * theta) \
                       * np.cos(j * (np.pi - n_idx * np.pi / N))
            elif i_idx == 0:
                # Upper boundary node (i = 0)
                top -= ((-1) ** (j + 2)) * (factor * j ** 2) \
                       * np.cos(j * (np.pi - n_idx * np.pi / N))
            else:
                # Lower boundary node (i = N)
                phi = np.pi - (N + n_idx) * np.pi / N
                top += ((-1) ** (j + 2)) * (factor * j ** 2) * np.cos(j * phi)
        A[i_idx, n_idx] = P * top

# Reduced sub-block A1 obtained by removing the row/column associated with
# the known initial-condition coefficient c_0(x) (cf. Eq. (14))
A1 = A[1:, 1:]

# -----------------------------------------------------------------------------
# 3) Symbolic source term F(x): contribution of the known initial profile
# -----------------------------------------------------------------------------
# z(x) is the initial condition u(x, 0) of the Burgers-Fisher problem
# (Eq. (48)). Its successive derivatives, divided by the corresponding
# factorial, give the local Taylor coefficients of the initial data, which
# enter the right-hand side z^{(k)}(x) of the reduced BVP (Eq. (14)).
x = sp.symbols('x')
z      = sp.Rational(1, 2) + sp.Rational(1, 2) * sp.tanh(delta1 * x)
f_syms = [z] + [sp.diff(z, x, i) / sp.factorial(i) for i in range(1, TO + 1)]

# Build an N x (TO+1) matrix that scales the Taylor-coefficient vector by the
# first column of the time-differentiation matrix A (i.e. the contribution of
# the imposed initial condition to the right-hand side).
F_expr = sp.Matrix([[A[i + 1, 0] * f_syms[j]
                     for j in range(TO + 1)]
                    for i in range(N)])
# Lambdify for fast numerical evaluation at any spatial point
F_func = sp.lambdify(x, F_expr, 'numpy')

# -----------------------------------------------------------------------------
# 4) Initial guess for the nonlinear algebraic system
# -----------------------------------------------------------------------------
# The unknown vector contains the local Taylor coefficients C_i(0) and C_i(1)
# (initial values and slopes) at all interior spatial nodes (cf. Eq. (25)).
# Its size is 2 * M * N.
S0 = np.zeros(2 * M * N)

# -----------------------------------------------------------------------------
# 5) Residual function for the coupled IELDTM-ChSCM system
# -----------------------------------------------------------------------------
def obj(u):
    """
    Residual of the IELDTM-ChSCM nonlinear algebraic system.

    Given the unknown vector u (containing the local Taylor coefficients at
    every spatial node and every Chebyshev time node), this function:
      (i)   reconstructs the full Taylor expansion R[i, n, :] up to order TO+2
            by applying the differential-transform recurrence (Eq. (24));
      (ii)  enforces C^0 and C^1 continuity between adjacent spatial intervals
            at the matching points x_i + (1 - theta) * dx (Eqs. (20)-(23));
      (iii) returns the resulting residual vector to be driven to zero by the
            nonlinear solver.
    """

    # Boundary conditions u(0, t) and u(1, t) of the Burgers-Fisher problem
    # (Eqs. (49)-(50))
    f_ic = lambda t: 0.5 + 0.5 * np.tanh(-delta1 * delta2 * t)
    g_ic = lambda t: 0.5 + 0.5 * np.tanh(delta1 - delta1 * delta2 * t)

    # R[i, n, :] holds the n-th local Taylor coefficient at spatial node i,
    # evaluated at all Chebyshev time nodes c[1:] (size N).
    R = np.zeros((M + 1, TO + 3, N))

    # --- Boundary node i = 0 ---
    R[0, 0, :] = f_ic(c[1:])     # C_0(0) prescribed by the left BC
    R[0, 1, :] = u[0:N]          # C_0(1) is unknown

    # --- Interior nodes i = 1, ..., M-1 ---
    for i_idx in range(1, M):
        start = (2 * i_idx - 1) * N
        R[i_idx, 0, :] = u[start         : start + N]      # C_i(0)
        R[i_idx, 1, :] = u[start + N     : start + 2 * N]  # C_i(1)

    # --- Boundary node i = M ---
    start = (2 * M - 1) * N
    R[M, 0, :] = g_ic(c[1:])     # C_M(0) prescribed by the right BC
    R[M, 1, :] = u[start:start + N]  # C_M(1) is unknown

    # --- Differential-transform recurrence (Eq. (24)) ---
    # Compute higher-order Taylor coefficients R[i, n+2, :] for n = 0, ..., TO-1
    # using the local nonlinear ADR recurrence at every spatial node.
    for i_idx in range(M + 1):
        # Source contribution from the imposed initial data at x = i*dx
        GV = np.array(F_func(i_idx * dx), dtype=float)     # shape (N, TO+1)
        for n in range(TO):
            C  = R[i_idx, n, :]
            # Convolution sum for the reaction term u^2 (Burgers-Fisher)
            DD = sum(R[i_idx, k, :] * R[i_idx, n - k, :]
                     for k in range(n + 1))
            # Convolution sum for the advection term u*u_x
            D  = sum(R[i_idx, k, :] * (n - k + 1) * R[i_idx, n - k + 1, :]
                     for k in range(n + 1))
            FV = GV[:, n]
            # Recurrence relation derived from the governing PDE
            R[i_idx, n + 2, :] = (
                A1.dot(C) + V * D + FV - beta * C + beta * DD
            ) / (v * (n + 1) * (n + 2))

    # --- Continuity residuals at the matching points ---
    # The unknowns are determined by enforcing C^0 and C^1 continuity of the
    # local Taylor expansions of two adjacent spatial intervals at the
    # interior matching points x_i + (1 - theta) * dx (Eqs. (20)-(23)).
    fobj = np.zeros(2 * M * N)

    # First block: C^0 continuity (function value)
    for i_idx in range(M):
        TOP1 = sum(R[i_idx + 1, j, :] * ((-alfa * dx) ** j)
                   for j in range(TO + 1))
        TOP2 = sum(R[i_idx,     j, :] * (((1 - alfa) * dx) ** j)
                   for j in range(TO + 1))
        fobj[i_idx * N:(i_idx + 1) * N] = TOP1 - TOP2

    # Second block: C^1 continuity (first derivative)
    for i_idx in range(M):
        TOP1 = sum((j + 1) * R[i_idx + 1, j + 1, :] * ((-alfa * dx) ** j)
                   for j in range(TO))
        TOP2 = sum((j + 1) * R[i_idx,     j + 1, :] * (((1 - alfa) * dx) ** j)
                   for j in range(TO))
        start = M * N + i_idx * N
        fobj[start:start + N] = TOP1 - TOP2

    return fobj

# -----------------------------------------------------------------------------
# 6) Solve the nonlinear algebraic system with fsolve
# -----------------------------------------------------------------------------
start = time.process_time()
start_time = time.perf_counter()
RS, infodict, ier, mesg = fsolve(
    obj, S0,
    xtol=1e-10,
    maxfev=500_000,
    full_output=True
)
end = time.process_time()
print(f"CPU time: {end - start:.6f} seconds")

# -----------------------------------------------------------------------------
# 7) Extract nodal solution values at the final time t = t_f
# -----------------------------------------------------------------------------
# DF stores the values C_i(0) at all interior spatial nodes for each Chebyshev
# time node. The N-th row corresponds to t = t_f.
DF = np.zeros((N, M - 1))
s = 0
for j in range(1, 2 * M - 1, 2):
    DF[:, s] = RS[j * N : j * N + N]
    s += 1

# -----------------------------------------------------------------------------
# 8) Analytical solution and pointwise error at t = t_f
# -----------------------------------------------------------------------------
# Exact traveling-wave solution of the Burgers-Fisher equation (Eq. (51))
SOL = lambda x_val, t: 0.5 + 0.5 * np.tanh(delta1 * x_val - delta1 * delta2 * t)

EX  = np.zeros(M - 1)
ERR = np.zeros(M - 1)
for i_idx in range(M - 1):
    EX[i_idx]  = SOL((i_idx + 1) * dx, tf)
    ERR[i_idx] = abs(EX[i_idx] - DF[N - 1, i_idx])

print("Max abs error at t = t_f:", max(ERR))

# -----------------------------------------------------------------------------
# 9) Visualization at the final time t = t_f
# -----------------------------------------------------------------------------
p = np.linspace(dx, xf - dx, M - 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Numerical vs. analytical solution
ax1.plot(p, DF[N - 1, :], 'o', label='Numerical')
ax1.plot(p, EX,           '-', label='Analytical')
ax1.set_xlabel('x')
ax1.set_ylabel('u(x, t_f)')
ax1.set_title('Numerical vs. analytical solution at t = t_f')
ax1.legend()

# Pointwise absolute error
ax2.plot(p, ERR)
ax2.set_xlabel('x')
ax2.set_ylabel('|u_exact - u_num|')
ax2.set_title('Absolute error at t = t_f')

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 10) Spatiotemporal error matrix over all Chebyshev time nodes
# -----------------------------------------------------------------------------
# DF has shape (N, M-1):
#   - rows: Chebyshev time nodes c[1:] (excluding t_0, including t_f)
#   - cols: interior spatial grid points x = dx, 2*dx, ..., xf - dx
t_nodes = c[1:]
x_nodes = np.linspace(dx, xf - dx, M - 1)

# Build the analytical solution on the (t_nodes, x_nodes) grid via broadcasting
T = t_nodes[:, None]
X = x_nodes[None, :]
EX_mat  = 0.5 + 0.5 * np.tanh(delta1 * X - delta1 * delta2 * T)
ERR_mat = np.abs(EX_mat - DF)

print("ERR_mat shape:", ERR_mat.shape)
print("Max error over all time and space:", np.max(ERR_mat))

# -----------------------------------------------------------------------------
# 11) 3D surface plots over the (x, t) domain
# -----------------------------------------------------------------------------
# Build a 2D (x, t) mesh for surface plotting. We sort the Chebyshev time
# nodes in ascending order so that the surface is rendered with a monotonic
# t-axis (Chebyshev nodes are returned in descending order from c[1:]).
sort_idx = np.argsort(t_nodes)
t_sorted = t_nodes[sort_idx]
DF_sorted     = DF[sort_idx, :]
EX_mat_sorted = EX_mat[sort_idx, :]
ERR_mat_sorted = ERR_mat[sort_idx, :]

X_grid, T_grid = np.meshgrid(x_nodes, t_sorted)

fig = plt.figure(figsize=(18, 5.5))

# (a) Numerical solution surface
ax_num = fig.add_subplot(1, 3, 1, projection='3d')
surf_num = ax_num.plot_surface(
    X_grid, T_grid, DF_sorted,
    cmap='viridis', edgecolor='none', antialiased=True
)
ax_num.set_xlabel('x')
ax_num.set_ylabel('t')
ax_num.set_zlabel('u(x, t)')
ax_num.set_title('Numerical solution (IELDTM)')
fig.colorbar(surf_num, ax=ax_num, shrink=0.6, aspect=12)

# (b) Analytical solution surface
ax_ex = fig.add_subplot(1, 3, 2, projection='3d')
surf_ex = ax_ex.plot_surface(
    X_grid, T_grid, EX_mat_sorted,
    cmap='viridis', edgecolor='none', antialiased=True
)
ax_ex.set_xlabel('x')
ax_ex.set_ylabel('t')
ax_ex.set_zlabel('u(x, t)')
ax_ex.set_title('Analytical solution')
fig.colorbar(surf_ex, ax=ax_ex, shrink=0.6, aspect=12)

# (c) Absolute error surface
ax_err = fig.add_subplot(1, 3, 3, projection='3d')
surf_err = ax_err.plot_surface(
    X_grid, T_grid, ERR_mat_sorted,
    cmap='magma', edgecolor='none', antialiased=True
)
ax_err.set_xlabel('x')
ax_err.set_ylabel('t')
ax_err.set_zlabel('|u_exact - u_num|')
ax_err.set_title('Absolute error')
fig.colorbar(surf_err, ax=ax_err, shrink=0.6, aspect=12)

plt.tight_layout()
plt.show()
