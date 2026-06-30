# =============================================================================
#  Multi-interval IELDTM solver for the one-dimensional Burgers-Fisher equation
# -----------------------------------------------------------------------------
#  Equation:    u_t + alpha*u*u_x = epsilon*u_xx + beta*u*(1 - u),
#               x in [0, 1], t in [0, t_f]
#
#  This script implements the Implicit-Explicit Local Differential Transform
#  Method (IELDTM) coupled with a multi-interval Chebyshev spectral collocation
#  strategy in time. In contrast to the baseline uniform-mesh implementation,
#  the present code allows a different spatial mesh to be used in each time
#  subinterval. Therefore, the solver can run either with a fixed uniform mesh
#  or with adaptive meshes imported from an external Excel file.
#
#  Main workflow:
#    1) Define physical, spatial, and temporal parameters.
#    2) Construct Chebyshev time-collocation matrices on each time subinterval.
#    3) Solve the local nonlinear IELDTM system on the current spatial mesh.
#    4) Transfer the terminal Taylor data to the next mesh by quintic Hermite
#       projection when the mesh changes between consecutive time intervals.
#    5) Compare the numerical solution with the analytical traveling-wave
#       solution and visualize the error and mesh density.
# =============================================================================

import numpy as np
import sympy as sp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time
import math

# -----------------------------------------------------------------------------
# 0) Parameters
# -----------------------------------------------------------------------------
# Discretization parameters
N      = 5     # Order of the Chebyshev polynomial expansion in time
M      = 32    # Baseline number of spatial mesh points used in the uniform mesh
TO     = 4     # Order of the local Taylor series expansion in space
sub_dt = 0.2   # Length of each time subinterval
alfa   = 0.5   # Internal collocation parameter theta in [0, 1]
               # used for the C^0 / C^1 continuity matching points

# Physical parameters of the Burgers-Fisher equation
v      = 2**(-8)   # Diffusion coefficient epsilon
V      = 0.1       # Advection coefficient alpha
beta   = 0.1       # Reaction coefficient beta

# Constants of the analytical traveling-wave solution
delta1 = -V / (4 * v)
delta2 = V / 2 + (2 * v * beta) / V

# Temporal and spatial domains
tf     = 1.0
xb, xf = 0.0, 1.0

# Time partition for the multi-interval formulation:
# [0.0, 0.2], [0.2, 0.4], ..., [0.8, 1.0]
t_points = np.arange(0.0, tf + 1e-9, sub_dt)
i_vals   = np.arange(N + 1)

# -----------------------------------------------------------------------------
# 1) Boundary conditions and analytical solution
# -----------------------------------------------------------------------------
# The functions below are obtained from the exact traveling-wave solution of the
# Burgers-Fisher equation. They are used both to impose Dirichlet boundary
# conditions and to compute the final error.
f_ic = lambda t: 0.5 + 0.5 * np.tanh(-delta1 * delta2 * t)             # x = 0
g_ic = lambda t: 0.5 + 0.5 * np.tanh(delta1 - delta1 * delta2 * t)     # x = 1
SOL  = lambda x_val, t: 0.5 + 0.5 * np.tanh(delta1 * x_val - delta1 * delta2 * t)

# =============================================================================
# 2) Mesh definition
# -----------------------------------------------------------------------------
#  x_mesh_list contains the spatial mesh used in each time subinterval. Its
#  length must be equal to len(t_points)-1. Each entry is a strictly increasing
#  NumPy array in the interval [0, 1]. This structure makes it possible to use
#  the same uniform mesh in all subintervals or a different adaptive mesh in
#  each subinterval.
# =============================================================================

def load_adaptive_meshes_from_excel(file_path, expected_K=None):
    """
    Load adaptive spatial meshes from an Excel file.

    The Excel file is assumed to store one adaptive mesh per column. Each column
    corresponds to one time subinterval [t_k, t_{k+1}], while the non-empty
    entries in that column give the spatial mesh points. For the present setting
    with t_points = [0.0, 0.2, ..., 1.0], the file should contain five columns.

    Parameters
    ----------
    file_path : str
        Path of the Excel file containing the adaptive mesh points.
    expected_K : int, optional
        Expected number of time subintervals. If given, the function checks that
        the number of imported meshes is consistent with len(t_points)-1.

    Returns
    -------
    list of numpy.ndarray
        List of adaptive spatial meshes to be used by solve_with_given_meshes.
    """
    import pandas as pd

    df = pd.read_excel(file_path, header=None)
    adaptive_mesh_list = []

    for col in df.columns:
        # Remove empty cells and convert the column into a numerical mesh array.
        mesh = df[col].dropna().to_numpy(dtype=float)
        if mesh.size == 0:
            continue

        # Sort the mesh points and remove accidental duplicates.
        mesh = np.unique(mesh)
        mesh.sort()

        # The IELDTM solver expects every mesh to cover the full spatial domain.
        if not np.isclose(mesh[0], xb) or not np.isclose(mesh[-1], xf):
            raise ValueError(
                f"Column {col} must contain a mesh starting at {xb} and ending at {xf}."
            )
        if not np.all(np.diff(mesh) > 0):
            raise ValueError(f"Column {col} must contain strictly increasing mesh points.")

        adaptive_mesh_list.append(mesh)

    if expected_K is not None and len(adaptive_mesh_list) != expected_K:
        raise ValueError(
            f"The Excel file must contain {expected_K} adaptive meshes, "
            f"but {len(adaptive_mesh_list)} were found."
        )

    return adaptive_mesh_list


# --- Option 1: Uniform mesh in all time subintervals --------------------------
# This is the baseline setting. The same equally spaced spatial mesh is used
# from t = 0 to t = t_f. This option is useful for validating the method before
# switching to an adaptive mesh.
x_mesh_list = [
    np.linspace(xb, xf, M),
    np.linspace(xb, xf, M),
    np.linspace(xb, xf, M),
    np.linspace(xb, xf, M),
    np.linspace(xb, xf, M),
]

# --- Option 2: Adaptive mesh loaded from Excel --------------------------------
# Uncomment the following line when the adaptive meshes are stored in an Excel
# file. The Excel file should contain one column for each time subinterval. Each
# column must start with xb=0 and end with xf=1.
# x_mesh_list = load_adaptive_meshes_from_excel(
#     "adaptive_mesh.xlsx", expected_K=len(t_points)-1
# )

# -----------------------------------------------------------------------------
# 3) Chebyshev time-differentiation matrix on a time subinterval
# -----------------------------------------------------------------------------
def build_A_and_blocks(factor):
    """
    Construct the Chebyshev time-differentiation matrix and its reduced blocks.

    The full matrix A maps nodal solution values at the Chebyshev-Gauss-Lobatto
    points to their time derivatives on the current time subinterval. Since the
    initial value at the left endpoint of the subinterval is known, the reduced
    block A1 and the first-column contribution Acol are used in the local
    differential-transform recurrence.
    """
    A = np.zeros((N + 1, N + 1))

    for i_idx in range(N + 1):
        for n_idx in range(N + 1):
            # Quadrature weight P: endpoint nodes have weight 1/N,
            # interior nodes have weight 2/N.
            P = 1.0 / N if (n_idx == 0 or n_idx == N) else 2.0 / N
            top = 0.0
            theta = np.pi - i_idx * np.pi / N

            for j in range(N + 1):
                if 0 < i_idx < N:
                    # Interior Chebyshev collocation nodes
                    top += (factor * j / np.sqrt(1 - np.cos(theta) ** 2)) \
                         * np.sin(j * theta) \
                         * np.cos(j * (np.pi - n_idx * np.pi / N))
                elif i_idx == 0:
                    # Upper boundary node of the Chebyshev interval
                    top -= ((-1) ** (j + 2)) * (factor * j ** 2) \
                         * np.cos(j * (np.pi - n_idx * np.pi / N))
                else:
                    # Lower boundary node of the Chebyshev interval
                    phi = np.pi - (N + n_idx) * np.pi / N
                    top += ((-1) ** (j + 2)) * (factor * j ** 2) * np.cos(j * phi)

            A[i_idx, n_idx] = P * top

    # A1 acts on the unknown nodal coefficients, while Acol accounts for the
    # known initial data at the beginning of the current time subinterval.
    A1   = A[1:, 1:]   # shape: (N, N)
    Acol = A[1:, 0]    # shape: (N,)
    return A1, Acol

# -----------------------------------------------------------------------------
# 4) Initial Taylor coefficients from the exact solution
# -----------------------------------------------------------------------------
x_sy = sp.symbols('x')

def exact_f_divfac_at(x_mesh, t0):
    """
    Evaluate z^(j)(x)/j! for j = 0, ..., TO at the beginning of a subinterval.

    For the first time subinterval, the initial Taylor coefficients are obtained
    directly from the analytical solution. These values provide the local
    spatial Taylor data needed by the IELDTM recurrence.
    """
    z = sp.Rational(1, 2) + sp.Rational(1, 2) * sp.tanh(delta1 * x_sy - delta1 * delta2 * t0)
    derivs = [z] + [sp.diff(z, x_sy, j) / sp.factorial(j) for j in range(1, TO + 1)]
    funcs  = [sp.lambdify(x_sy, d, 'numpy') for d in derivs]
    return np.column_stack([f(x_mesh) for f in funcs])  # shape: (M+1, TO+1)

# -----------------------------------------------------------------------------
# 5) Quintic Hermite projection between consecutive spatial meshes
# -----------------------------------------------------------------------------
def _quintic_coeffs(f0, f1, fp0, fp1, fpp0, fpp1, h):
    """
    Compute the coefficients of a quintic Hermite polynomial on one cell.

    The polynomial is determined from function values, first derivatives, and
    second derivatives at the two endpoints of the old mesh cell. This C^2
    reconstruction is used to transfer terminal data from the old mesh to the
    next mesh when adaptive meshes vary in time.
    """
    a0 = f0
    a1 = fp0
    a2 = 0.5 * fpp0

    H  = np.array([
        [h**3,     h**4,       h**5],
        [3*h**2,   4*h**3,     5*h**4],
        [6*h,      12*h**2,    20*h**3]
    ], dtype=float)

    b  = np.array([
        f1 - (a0 + a1 * h + a2 * h * h),
        fp1 - (a1 + 2 * a2 * h),
        fpp1 - (2 * a2)
    ], dtype=float)

    a3, a4, a5 = np.linalg.solve(H, b)
    return np.array([a0, a1, a2, a3, a4, a5], dtype=float)


def _poly_derivs_at(c, y, TO):
    """
    Evaluate a polynomial and its derivatives at a local coordinate y.

    The output contains the actual derivatives p^(j)(y), j = 0, ..., TO. These
    are later divided by j! to recover the IELDTM Taylor coefficients.
    """
    vals = np.zeros(TO + 1, dtype=float)

    # Function value p(y)
    yy = 1.0
    p = 0.0
    for m in range(6):
        p += c[m] * yy
        yy *= y
    vals[0] = p

    # Derivatives p^(r)(y)
    for r in range(1, TO + 1):
        s = 0.0
        for m in range(r, 6):
            fact = 1.0
            for t in range(r):
                fact *= (m - t)
            s += c[m] * fact * (y ** (m - r))
        vals[r] = s

    return vals


def project_R_end_to_new_mesh(x_old, R_end, x_new, TO):
    """
    Project terminal Taylor data from the old spatial mesh to the new mesh.

    Parameters
    ----------
    x_old : numpy.ndarray
        Spatial mesh used in the current time subinterval.
    R_end : numpy.ndarray
        Terminal Taylor coefficients z^(j)/j! at t = b_k on x_old.
    x_new : numpy.ndarray
        Spatial mesh to be used in the next time subinterval.
    TO : int
        Taylor expansion order.

    Returns
    -------
    numpy.ndarray
        Projected Taylor coefficients z^(j)/j! at t = b_k on x_new.
    """
    fact = np.array([math.factorial(j) for j in range(TO + 1)], dtype=float)

    # Convert Taylor coefficients z^(j)/j! into actual derivatives z^(j).
    D_old = R_end * fact[None, :]

    f_new = np.zeros((len(x_new), TO + 1), dtype=float)
    M_old = len(x_old) - 1
    edges = x_old

    for idx, xq in enumerate(x_new):
        # Locate the old mesh cell containing the new mesh point xq.
        if xq <= x_old[0]:
            i = 0
            h = x_old[1] - x_old[0]
            y = 0.0
        elif xq >= x_old[-1]:
            i = M_old - 1
            h = x_old[i + 1] - x_old[i]
            y = h
        else:
            i = np.searchsorted(edges, xq) - 1
            i = max(0, min(i, M_old - 1))
            h = x_old[i + 1] - x_old[i]
            y = xq - x_old[i]

        # Endpoint data used by the local quintic Hermite polynomial.
        f0,  f1  = D_old[i, 0],   D_old[i + 1, 0]
        fp0, fp1 = D_old[i, 1],   D_old[i + 1, 1]
        fpp0 = D_old[i, 2]     if TO >= 2 else 0.0
        fpp1 = D_old[i + 1, 2] if TO >= 2 else 0.0

        c = _quintic_coeffs(f0, f1, fp0, fp1, fpp0, fpp1, h)
        derivs = _poly_derivs_at(c, y, TO)

        # Convert actual derivatives back to Taylor coefficients.
        f_new[idx, :] = derivs / fact

    return f_new

# -----------------------------------------------------------------------------
# 6) Reconstruction of the full local Taylor tensor R
# -----------------------------------------------------------------------------
def fill_R_from_u(RS, c, A1, Acol, f_divfac_current, M):
    """
    Reconstruct the local Taylor coefficients after solving the nonlinear system.

    The nonlinear solver returns only the unknown coefficients stored in RS.
    This function rebuilds the complete tensor R, including boundary values and
    higher-order Taylor coefficients generated by the IELDTM recurrence.

    Parameters
    ----------
    RS : numpy.ndarray
        Unknown vector of size 2*M*N returned by fsolve.
    c : numpy.ndarray
        Chebyshev time nodes of the current subinterval.
    A1, Acol : numpy.ndarray
        Reduced Chebyshev differentiation blocks.
    f_divfac_current : numpy.ndarray
        Initial Taylor coefficients z^(j)/j! on the current mesh.
    M : int
        Number of spatial intervals in the current mesh.

    Returns
    -------
    numpy.ndarray
        Full Taylor tensor R with shape (M+1, TO+3, N).
    """
    R = np.zeros((M + 1, TO + 3, N))

    # Boundary values prescribed at all Chebyshev time nodes of the subinterval.
    R[0, 0, :] = f_ic(c[1:])
    R[M, 0, :] = g_ic(c[1:])

    # Insert the unknown coefficients from the fsolve vector RS.
    R[0, 1, :] = RS[0:N]
    for i_idx in range(1, M):
        start = (2 * i_idx - 1) * N
        R[i_idx, 0, :] = RS[start         : start + N]      # C_i(0)
        R[i_idx, 1, :] = RS[start + N     : start + 2 * N]  # C_i(1)

    start = (2 * M - 1) * N
    R[M, 1, :] = RS[start:start + N]

    # Differential-transform recurrence for higher-order spatial coefficients.
    for i_idx in range(M + 1):
        GV = np.outer(Acol, f_divfac_current[i_idx, :])  # shape: N x (TO+1)

        for n in range(TO):
            C  = R[i_idx, n, :]

            # Convolution associated with the nonlinear reaction term u^2.
            DD = sum(R[i_idx, k, :] * R[i_idx, n - k, :] for k in range(n + 1))

            # Convolution associated with the nonlinear advection term u*u_x.
            D  = sum(
                R[i_idx, k, :] * (n - k + 1) * R[i_idx, n - k + 1, :]
                for k in range(n + 1)
            )

            FV = GV[:, n]

            # Local recurrence derived from the Burgers-Fisher equation.
            R[i_idx, n + 2, :] = (
                A1.dot(C) + V * D + FV - beta * C + beta * DD
            ) / (v * (n + 1) * (n + 2))

    return R

# -----------------------------------------------------------------------------
# 7) Initial guess for the nonlinear algebraic system
# -----------------------------------------------------------------------------
def initial_guess_from_profile(f_divfac_current, M):
    """
    Build a warm-start vector from the current initial profile.

    This optional initialization uses the function value and first derivative of
    the current profile to populate the unknown vector. In the main solver below,
    a zero initial guess is currently used; this function is kept as an
    alternative warm-start strategy.
    """
    u_prof  = f_divfac_current[:, 0]
    ux_prof = f_divfac_current[:, 1]

    u0 = np.zeros(2 * M * N)

    # Left boundary derivative block
    u0[0:N] = ux_prof[0]

    # Interior value and derivative blocks
    for i_idx in range(1, M):
        start = (2 * i_idx - 1) * N
        u0[start         : start + N]     = u_prof[i_idx]
        u0[start + N     : start + 2 * N] = ux_prof[i_idx]

    # Right boundary derivative block
    start = (2 * M - 1) * N
    u0[start:start + N] = ux_prof[-1]

    return u0

# -----------------------------------------------------------------------------
# 8) Main multi-interval solver
# -----------------------------------------------------------------------------
def solve_with_given_meshes(x_mesh_list):
    """
    Solve the Burgers-Fisher equation using the prescribed sequence of meshes.

    The solver advances from one time subinterval to the next. On each
    subinterval, it constructs the local Chebyshev differentiation matrix,
    solves the nonlinear IELDTM system, stores the terminal solution, and then
    projects the terminal Taylor coefficients onto the mesh of the next
    subinterval if the mesh changes.
    """
    K = len(t_points) - 1

    if len(x_mesh_list) != K:
        raise ValueError(f"x_mesh_list must have length {K}, but {len(x_mesh_list)} was given.")

    solutions    = []
    U_end_list   = []   # u(x, b_k) at the end of each subinterval
    Ux_end_list  = []   # u_x(x, b_k) at the end of each subinterval

    # Initial Taylor coefficients on the first mesh are taken from the exact
    # initial condition at t = 0.
    x_mesh = x_mesh_list[0]
    if not np.all(np.diff(x_mesh) > 0):
        raise ValueError("x_mesh_list[0] must be strictly increasing.")

    f_divfac_next = exact_f_divfac_at(x_mesh, 0.0)

    for k in range(K):
        a = t_points[k]
        b = t_points[k + 1]

        # Spatial mesh used on the current time subinterval [a, b].
        x_mesh = x_mesh_list[k]
        if not np.all(np.diff(x_mesh) > 0):
            raise ValueError(f"x_mesh_list[{k}] must be strictly increasing.")

        Mcur   = len(x_mesh) - 1
        dx_arr = np.diff(x_mesh)

        # Chebyshev-Gauss-Lobatto nodes mapped to the current time subinterval.
        factor = 2.0 / (b - a)
        c = 0.5 * ((a + b) - (b - a) * np.cos(np.pi * i_vals / N))
        A1, Acol = build_A_and_blocks(factor)

        # Initial Taylor coefficients z^(j)/j! on the current mesh.
        f_divfac_current = f_divfac_next
        if f_divfac_current.shape[0] != (Mcur + 1):
            raise ValueError(
                f"k={k}: f_divfac_current shape is incompatible with the mesh: "
                f"{f_divfac_current.shape[0]} vs {Mcur + 1}"
            )

        # Initial guess for fsolve. The warm-start alternative is shown below.
        # u_prev = initial_guess_from_profile(f_divfac_current, Mcur)
        u_prev = np.zeros(2 * Mcur * N, dtype=float)

        # ---------------------------------------------------------------------
        # Residual function for the current local IELDTM system
        # ---------------------------------------------------------------------
        def obj(u):
            """
            Residual vector of the nonlinear algebraic system on one subinterval.

            The unknown vector u contains the local Taylor coefficients C_i(0)
            and C_i(1). The function reconstructs higher-order coefficients
            using the IELDTM recurrence and enforces C^0 / C^1 continuity at
            the matching points between neighboring spatial cells.
            """
            R = np.zeros((Mcur + 1, TO + 3, N))

            # Dirichlet boundary values at the Chebyshev time nodes.
            R[0,     0, :] = f_ic(c[1:])
            R[Mcur,  0, :] = g_ic(c[1:])

            # Unknown coefficient placement.
            R[0,     1, :] = u[0:N]
            for i_idx in range(1, Mcur):
                start = (2 * i_idx - 1) * N
                R[i_idx, 0, :] = u[start         : start + N]
                R[i_idx, 1, :] = u[start + N     : start + 2 * N]

            start = (2 * Mcur - 1) * N
            R[Mcur, 1, :] = u[start:start + N]

            # Differential-transform recurrence for higher-order coefficients.
            for i_idx in range(Mcur + 1):
                GV = np.outer(Acol, f_divfac_current[i_idx, :])

                for n in range(TO):
                    C  = R[i_idx, n, :]

                    # Nonlinear reaction contribution u^2.
                    DD = sum(
                        R[i_idx, kk, :] * R[i_idx, n - kk, :]
                        for kk in range(n + 1)
                    )

                    # Nonlinear advection contribution u*u_x.
                    D  = sum(
                        R[i_idx, kk, :] * (n - kk + 1) * R[i_idx, n - kk + 1, :]
                        for kk in range(n + 1)
                    )

                    FV = GV[:, n]

                    R[i_idx, n + 2, :] = (
                        A1.dot(C) + V * D + FV - beta * C + beta * DD
                    ) / (v * (n + 1) * (n + 2))

            # Continuity residuals. The first block enforces C^0 continuity,
            # and the second block enforces C^1 continuity.
            fobj = np.zeros(2 * Mcur * N)

            # C^0 continuity: matching of function values.
            for i_idx in range(Mcur):
                dx_i = dx_arr[i_idx]
                TOP1 = sum(
                    R[i_idx + 1, j, :] * ((-alfa * dx_i) ** j)
                    for j in range(TO + 1)
                )
                TOP2 = sum(
                    R[i_idx, j, :] * (((1 - alfa) * dx_i) ** j)
                    for j in range(TO + 1)
                )
                fobj[i_idx * N:(i_idx + 1) * N] = TOP1 - TOP2

            # C^1 continuity: matching of first derivatives.
            for i_idx in range(Mcur):
                dx_i = dx_arr[i_idx]
                TOP1 = sum(
                    (j + 1) * R[i_idx + 1, j + 1, :] * ((-alfa * dx_i) ** j)
                    for j in range(TO)
                )
                TOP2 = sum(
                    (j + 1) * R[i_idx, j + 1, :] * (((1 - alfa) * dx_i) ** j)
                    for j in range(TO)
                )
                start = Mcur * N + i_idx * N
                fobj[start:start + N] = TOP1 - TOP2

            return fobj

        # Solve the nonlinear algebraic system on the current subinterval.
        t0 = time.perf_counter()
        RS, infodict, ier, mesg = fsolve(
            obj, u_prev,
            xtol=1e-10,
            maxfev=500_000,
            full_output=True
        )
        dt = time.perf_counter() - t0

        print(f"[k={k}] [{a:.1f}–{b:.1f}] M={Mcur}  elapsed time={dt:.2f}s  ier={ier}")

        if ier != 1:
            raise RuntimeError(f"fsolve failed: {mesg}")

        solutions.append(RS)

        # Reconstruct R and extract the terminal solution at t = b_k.
        R_full = fill_R_from_u(RS, c, A1, Acol, f_divfac_current, Mcur)
        U_end  = R_full[:, 0, -1].copy()   # u(x, b_k)
        Ux_end = R_full[:, 1, -1].copy()   # u_x(x, b_k)

        U_end_list.append(U_end)
        Ux_end_list.append(Ux_end)

        # If another subinterval remains, project terminal Taylor data from the
        # current mesh to the mesh of the next time subinterval.
        if k < K - 1:
            x_mesh_next = x_mesh_list[k + 1]
            if not np.all(np.diff(x_mesh_next) > 0):
                raise ValueError(f"x_mesh_list[{k + 1}] must be strictly increasing.")

            # R_end stores z^(j)/j! at t = b_k on the current mesh.
            R_end = np.column_stack([R_full[:, j, -1] for j in range(TO + 1)])
            f_divfac_next = project_R_end_to_new_mesh(x_mesh, R_end, x_mesh_next, TO)

            # Reimpose boundary values to avoid small projection-induced
            # inconsistencies at x = 0 and x = 1.
            a_next = t_points[k + 1]
            f_divfac_next[0, 0]  = f_ic(a_next)
            f_divfac_next[-1, 0] = g_ic(a_next)

    return {
        "solutions": solutions,
        "U_end_list": U_end_list,
        "Ux_end_list": Ux_end_list,
    }

# -----------------------------------------------------------------------------
# 9) Run the solver and compute the final-time error
# -----------------------------------------------------------------------------
start = time.process_time()
out = solve_with_given_meshes(x_mesh_list)
end = time.process_time()

print(f"CPU time: {end - start:.6f} seconds")

# Compare the numerical and analytical solutions on the last spatial mesh.
x_last = x_mesh_list[-1]
u_num  = out["U_end_list"][-1]
u_ex   = SOL(x_last, tf)
err    = np.abs(u_ex - u_num)

print("Maximum error (last subinterval):", err.max())
print("Minimum error (last subinterval):", err.min())

plt.figure(figsize=(10, 4))
plt.plot(x_last, u_num, 'o', label='Numerical (t=1.0)')
plt.plot(x_last, u_ex,  '-', label='Analytical (t=1.0)')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 10) Solution visualization at all time-subinterval endpoints
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 5))

# Plot the numerical and analytical profiles at the end of each subinterval.
for k, (u_num, x_mesh) in enumerate(zip(out["U_end_list"], x_mesh_list)):
    t_val = t_points[k + 1]
    u_ex  = SOL(x_mesh, t_val)

    plt.plot(x_mesh, u_num, 'o', label=f'Numerical t={t_val:.1f}')
    plt.plot(x_mesh, u_ex,  '-', label=f'Analytical t={t_val:.1f}')

plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title(f"Test solution at v={v:.5f}, t in [0,1]")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 11) Mesh-density heatmap
# -----------------------------------------------------------------------------
# The density map visualizes how strongly the spatial nodes are concentrated
# across the domain. For each time subinterval, the local density is defined as
# 1/Delta x and then interpolated onto a common plotting grid.
nx = 200
nt = len(x_mesh_list)
x_grid = np.linspace(0, 1, nx)
t_grid = t_points[1:]

density_map = np.zeros((nt, nx))

for k, x_mesh in enumerate(x_mesh_list):
    dx_local = np.diff(x_mesh)
    local_density = 1.0 / dx_local

    # Density is associated with cell centers and interpolated to x_grid.
    x_mid = 0.5 * (x_mesh[:-1] + x_mesh[1:])
    density_map[k, :] = np.interp(x_grid, x_mid, local_density)

plt.figure(figsize=(8, 5))
plt.imshow(
    density_map,
    extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]],
    aspect='auto',
    cmap='hot'
)
plt.colorbar(label='Mesh density (1/Δx)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Adaptive Mesh Density (Heatmap)')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 12) Two-dimensional error map
# -----------------------------------------------------------------------------
# The error is computed on each adaptive mesh and interpolated onto a common
# uniform x-grid so that the temporal evolution of the error can be visualized
# as a two-dimensional image.
nx = 200
x_grid = np.linspace(0, 1, nx)
t_grid = t_points[1:]
err_map = np.zeros((len(t_grid), nx))

for k, (x_mesh, u_num) in enumerate(zip(x_mesh_list, out["U_end_list"])):
    u_ex = SOL(x_mesh, t_points[k + 1])
    err = np.abs(u_ex - u_num)
    err_map[k, :] = np.interp(x_grid, x_mesh, err)

plt.figure(figsize=(8, 5))
im = plt.imshow(
    err_map,
    extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]],
    aspect='auto',
    cmap='viridis'
)

cbar = plt.colorbar(im)
cbar.set_label('')
cbar.ax.tick_params(labelsize=15, width=1.2)

plt.xlabel('x', fontsize=17)
plt.ylabel('t', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Error Dynamics', fontsize=17)
plt.tight_layout()
plt.show()
