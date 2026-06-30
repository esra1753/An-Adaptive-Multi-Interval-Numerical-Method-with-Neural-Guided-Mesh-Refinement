# =============================================================================
#  Data-driven adaptive MLP framework for the one-dimensional Burgers equation
# -----------------------------------------------------------------------------
#  Mapping:     (x, t, epsilon) -> u(x,t;epsilon)
#  Residual:    R = u_t - epsilon*u_xx + V*u*u_x + beta*(u - u^2)
#
#  This script generates numerical solution data with an IELDTM-MI solver,
#  trains a parametric neural network, adaptively enriches the training set
#  according to PINN-type residual indicators, and finally visualizes only
#  the exact solution, the approximate solution, and the error heatmap.
# =============================================================================
import time
import numpy as np
import sympy as sp
from scipy.optimize import fsolve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import math

# -----------------------------------------------------------------------------
# 0) Global discretization and physical parameters
# -----------------------------------------------------------------------------
N      = 5       # Order of the Chebyshev polynomial expansion in each time subinterval
TO     = 4       # Order of the local Taylor series expansion in space
tf     = 1.0    # Final time
sub_dt = 0.2    # Length of each time subinterval
alfa   = 0.5    # Internal collocation parameter for C^0/C^1 matching
V      = 1      # Coefficient of the nonlinear advection term u*u_x
beta   = 0      # Reaction coefficient
xb, xf = 0.0, 1.0  # Spatial domain [xb, xf]

# Adaptive training settings
INITIAL_EPOCHS     = 800   # Initial data-driven training epochs
ADAPT_CYCLES       = 10    # Number of adaptive enrichment cycles
EPOCHS_PER_CYCLE   = 100   # Number of epochs in each adaptive cycle
V_POOL_SIZE        = 200   # Number of candidate diffusion values in the residual pool
TOP_K              = 3     # Number of most difficult diffusion values added per cycle
RES_NX, RES_NT     = 96, 64  # Residual grid resolution used for diffusion-parameter selection

# Adaptive x-mesh settings used for selected diffusion values
ADAPT_TIME_PROBES        = 10   # Number of time probes in each time subinterval
ADAPT_INSERT_POINTS_FACT = 1.0  # Relative amount of inserted interior points per time segment

t_points = np.arange(0.0, tf + 1e-12, sub_dt)   # Time-subinterval endpoints
i_vals   = np.arange(N+1)

# Quantities updated whenever the diffusion coefficient changes:
delta1 = None
delta2 = None
f_ic = None
g_ic = None
SOL  = None
v    = None

def set_diffusion(nu: float):
    """Set the diffusion coefficient and update the boundary/exact-solution functions."""
    global v, delta1, delta2, f_ic, g_ic, SOL
    v = float(nu)
    # Parameters of the analytical traveling-wave solution
    delta1 = -V/(4.0*v)
    delta2 = V/2.0 - (2.0*v*beta)/V  # Sign convention used in the exact solution
    f_ic = lambda t: 0.5 + 0.5*np.tanh(-delta1*delta2*t)                        # u(0,t)
    g_ic = lambda t: 0.5 + 0.5*np.tanh(delta1 - delta1*delta2*t)                # u(1,t)
    SOL  = lambda x_val, t: 0.5 + 0.5*np.tanh(delta1*x_val - delta1*delta2*t)   # exact solution

# -----------------------------------------------------------------------------
# 1) Chebyshev time-differentiation matrix blocks
# -----------------------------------------------------------------------------
def build_A_and_blocks(factor):
    A = np.zeros((N+1, N+1))
    # Assemble the Chebyshev differentiation matrix on the current time interval.
    for i_idx in range(N+1):
        for n_idx in range(N+1):
            P = 1.0/N if (n_idx==0 or n_idx==N) else 2.0/N
            top = 0.0
            theta = np.pi - i_idx*np.pi/N
            for j in range(N+1):
                if 0 < i_idx < N:
                    top += (factor*j/np.sqrt(1 - np.cos(theta)**2)) \
                         * np.sin(j*theta) * np.cos(j*(np.pi - n_idx*np.pi/N))
                elif i_idx == 0:
                    top -= ((-1)**(j+2)) * (factor * j**2) * np.cos(j*(np.pi - n_idx*np.pi/N))
                else:
                    phi = np.pi - (N + n_idx)*np.pi/N
                    top += ((-1)**(j+2)) * (factor * j**2) * np.cos(j*phi)
            A[i_idx, n_idx] = P * top
    A1   = A[1:, 1:]   # Reduced block associated with the unknown time coefficients
    Acol = A[1:, 0]    # First-column contribution from the known initial value
    return A1, Acol

# -----------------------------------------------------------------------------
# 2) Exact Taylor coefficients of the initial profile
# -----------------------------------------------------------------------------
x_sy = sp.symbols('x')
def exact_f_divfac_at(x_mesh, t0):
    z = sp.Rational(1,2) + sp.Rational(1,2)*sp.tanh(delta1*x_sy - delta1*delta2*t0)
    # Store z^(j)/j!, j=0,...,TO, as local Taylor coefficients.
    derivs = [z] + [sp.diff(z, x_sy, j)/sp.factorial(j) for j in range(1, TO+1)]
    funcs  = [sp.lambdify(x_sy, d, 'numpy') for d in derivs]
    arr = np.column_stack([f(x_mesh) for f in funcs])
    return np.asarray(arr, dtype=float)  # (M+1, TO+1)

# -----------------------------------------------------------------------------
# 3) Quintic Hermite projection between consecutive spatial meshes
# -----------------------------------------------------------------------------
def _quintic_coeffs(f0, f1, fp0, fp1, fpp0, fpp1, h):
    a0 = f0; a1 = fp0; a2 = 0.5*fpp0
    H  = np.array([[h**3, h**4, h**5],[3*h**2,4*h**3,5*h**4],[6*h,12*h**2,20*h**3]], dtype=float)
    b  = np.array([f1-(a0+a1*h+a2*h*h), fp1-(a1+2*a2*h), fpp1-(2*a2)], dtype=float)
    a3,a4,a5 = np.linalg.solve(H, b)
    return np.array([a0,a1,a2,a3,a4,a5], dtype=float)

def _poly_derivs_at(c, y, TO_):
    vals = np.zeros(TO_+1, dtype=float)
    yy=1.0; p=0.0
    for m in range(6): p += c[m]*yy; yy *= y
    vals[0]=p
    for r in range(1, TO_+1):
        s=0.0
        for m in range(r,6):
            fact=1.0
            for t in range(r): fact *= (m-t)
            s += c[m]*fact*(y**(m-r))
        vals[r]=s
    return vals

def project_R_end_to_new_mesh(x_old, R_end, x_new, TO_):
    fact = np.array([math.factorial(j) for j in range(TO_+1)], dtype=float)
    # Convert stored Taylor coefficients z^(j)/j! into physical derivatives.
    D_old = R_end * fact[None, :]
    f_new = np.zeros((len(x_new), TO_+1), dtype=float)
    M_old = len(x_old) - 1
    edges = x_old
    for idx, xq in enumerate(x_new):
        if xq <= x_old[0]:
            i = 0; h = x_old[1]-x_old[0]; y = 0.0
        elif xq >= x_old[-1]:
            i = M_old-1; h = x_old[i+1]-x_old[i]; y = h
        else:
            i = np.searchsorted(edges, xq) - 1
            i = max(0, min(i, M_old-1))
            h = x_old[i+1] - x_old[i]
            y = xq - x_old[i]
        f0,f1  = D_old[i,0],   D_old[i+1,0]
        fp0,fp1= D_old[i,1],   D_old[i+1,1]
        fpp0 = D_old[i,2]   if TO_>=2 else 0.0
        fpp1 = D_old[i+1,2] if TO_>=2 else 0.0
        c = _quintic_coeffs(f0,f1,fp0,fp1,fpp0,fpp1,h)
        derivs = _poly_derivs_at(c, y, TO_)
        f_new[idx,:] = derivs / fact
    return f_new

# -----------------------------------------------------------------------------
# 4) Reconstruction of the local Taylor-coefficient tensor R from fsolve output
# -----------------------------------------------------------------------------
def fill_R_from_u(RS, c, A1, Acol, f_divfac_current, M):
    """
    Residual form: u_t - epsilon*u_xx + V*u*u_x + beta*(u - u**2) = 0.
    The higher-order Taylor coefficients are computed through the IELDTM recurrence.
    """
    R = np.zeros((M+1, TO+3, N))
    # Prescribe boundary values and insert the unknown value/slope blocks.
    R[0, 0, :] = f_ic(c[1:])
    R[M, 0, :] = g_ic(c[1:])
    R[0, 1, :] = RS[0:N]
    for i_idx in range(1, M):
        start = (2*i_idx - 1)*N
        R[i_idx, 0, :] = RS[start         : start + N]
        R[i_idx, 1, :] = RS[start + N     : start + 2*N]
    start = (2*M - 1)*N
    R[M, 1, :] = RS[start:start+N]

    # Apply the differential-transform recurrence at every spatial node.
    for i_idx in range(M+1):
        GV = np.outer(Acol, f_divfac_current[i_idx, :])  # N x (TO+1)
        for n in range(TO):
            C  = R[i_idx, n, :]
            DD = sum(R[i_idx, k,    :]*R[i_idx, n-k,   :] for k in range(n+1))              # u^2
            D_ = sum(R[i_idx, k,    :]*(n-k+1)*R[i_idx, n-k+1, :] for k in range(n+1))      # u*u_x
            FV = GV[:, n]
            R[i_idx, n+2, :] = (A1.dot(C) + V*D_ + FV - beta*C + beta*DD) / (v*(n+1)*(n+2))
    return R

# -----------------------------------------------------------------------------
# 5) Initial guess construction for the nonlinear algebraic system
# -----------------------------------------------------------------------------
def initial_guess_from_profile(f_divfac_current, M):
    u_prof  = f_divfac_current[:, 0]
    ux_prof = f_divfac_current[:, 1]
    u0 = np.zeros(2*M*N, dtype=float)
    # The vector follows the same ordering as the fsolve unknown vector.
    u0[0:N] = ux_prof[0]
    for i_idx in range(1, M):
        start = (2*i_idx - 1)*N
        u0[start         : start + N]   = u_prof[i_idx]
        u0[start + N     : start + 2*N] = ux_prof[i_idx]
    start = (2*M - 1)*N
    u0[start:start+N] = ux_prof[-1]
    return u0

# -----------------------------------------------------------------------------
# 6) Residual function for one time subinterval
# -----------------------------------------------------------------------------
def make_obj_for_step(x_mesh, a, b, A1, Acol, f_divfac_current):
    Mcur   = len(x_mesh) - 1
    dx_arr = np.diff(x_mesh)
    c = 0.5 * ((a + b) - (b - a) * np.cos(np.pi * i_vals / N))  # N+1
    def obj(u):
        # Reconstruct local Taylor coefficients from the current nonlinear iterate.
        R = np.zeros((Mcur+1, TO+3, N))
        R[0,     0, :] = f_ic(c[1:])
        R[Mcur,  0, :] = g_ic(c[1:])
        R[0,     1, :] = u[0:N]
        for i_idx in range(1, Mcur):
            start = (2*i_idx - 1)*N
            R[i_idx, 0, :] = u[start         : start + N]
            R[i_idx, 1, :] = u[start + N     : start + 2*N]
        start = (2*Mcur - 1)*N
        R[Mcur, 1, :] = u[start:start+N]

        for i_idx in range(Mcur+1):
            GV = np.outer(Acol, f_divfac_current[i_idx, :])
            for n in range(TO):
                C  = R[i_idx, n, :]
                DD = sum(R[i_idx, kk,   :]*R[i_idx, n-kk,   :] for kk in range(n+1))
                D_ = sum(R[i_idx, kk,   :]*(n-kk+1)*R[i_idx, n-kk+1, :] for kk in range(n+1))
                FV = GV[:, n]
                R[i_idx, n+2, :] = (A1.dot(C) + V*D_ + FV + beta*C - beta*DD) / (v*(n+1)*(n+2))

        # Continuity residuals for the function value and first derivative
        fobj = np.zeros(2 * Mcur * N)
        # C^0 continuity
        for i_idx in range(Mcur):
            dx_i = dx_arr[i_idx]
            TOP1 = sum(R[i_idx+1, j, :]*((-alfa*dx_i)**j)    for j in range(TO+1))
            TOP2 = sum(R[i_idx,   j, :]*(((1-alfa)*dx_i)**j) for j in range(TO+1))
            fobj[i_idx*N:(i_idx+1)*N] = TOP1 - TOP2
        # C^1 continuity
        for i_idx in range(Mcur):
            dx_i = dx_arr[i_idx]
            TOP1 = sum((j+1)*R[i_idx+1, j+1, :]*((-alfa*dx_i)**j) for j in range(TO))
            TOP2 = sum((j+1)*R[i_idx,   j+1, :]*(((1-alfa)*dx_i)**j) for j in range(TO))
            start = Mcur*N + i_idx*N
            fobj[start:start+N] = TOP1 - TOP2
        return fobj
    return obj

# -----------------------------------------------------------------------------
# 7) Dataset generation on the baseline uniform meshes
# -----------------------------------------------------------------------------
def build_dataset_multi_v(v_values, x_mesh_list, include_bc=True, verbose=True):
    """
    Build a supervised dataset over all generated x- and t-nodes.
    The input matrix is X=(x,t,epsilon), and the target vector is y=u(x,t).
    """
    K = len(t_points) - 1
    if len(x_mesh_list) != K:
        raise ValueError(f"x_mesh_list must have length {K}, but {len(x_mesh_list)} was given.")
    if not all(np.all(np.diff(xm) > 0) for xm in x_mesh_list):
        raise ValueError("Each spatial mesh must be strictly increasing.")

    X_parts, y_parts = [], []

    for nu in np.asarray(v_values, dtype=float).ravel():
        set_diffusion(nu)
        x_mesh0 = x_mesh_list[0]
        f_divfac_next = exact_f_divfac_at(x_mesh0, 0.0)
        for k in range(K):
            a = t_points[k]; b = t_points[k+1]
            x_mesh = x_mesh_list[k]
            Mcur   = len(x_mesh) - 1
            factor = 2.0 / (b - a)
            c = 0.5 * ((a + b) - (b - a) * np.cos(np.pi * i_vals / N))  # N+1
            A1, Acol = build_A_and_blocks(factor)
            f_divfac_current = f_divfac_next
            if f_divfac_current.shape[0] != (Mcur+1):
                raise ValueError(f"k={k}: f_divfac_current has incompatible shape {f_divfac_current.shape[0]} != {Mcur+1}")
            u_prev = initial_guess_from_profile(f_divfac_current, Mcur)
            obj = make_obj_for_step(x_mesh, a, b, A1, Acol, f_divfac_current)
            RS, infodict, ier, mesg = fsolve(obj, u_prev, xtol=1e-10, maxfev=500_000, full_output=True)
            if ier != 1:
                raise RuntimeError(f"[v={nu:.3g}, k={k}] fsolve failed: {mesg}")
            R_full = fill_R_from_u(RS, c, A1, Acol, f_divfac_current, Mcur)

            x_eval = x_mesh if include_bc else x_mesh[1:-1]
            # Include the left endpoint t=a_k
            u_a = f_divfac_current[:, 0] if include_bc else f_divfac_current[1:-1, 0]
            X_parts.append(np.column_stack([x_eval,
                                            np.full_like(x_eval, a, float),
                                            np.full_like(x_eval, nu, float)]))
            y_parts.append(u_a.reshape(-1,1))
            # Include the interior Chebyshev nodes and the right endpoint t=b_k
            for m in range(N):
                t_m  = float(c[m+1])
                u_m  = R_full[:, 0, m]
                u_m  = (u_m if include_bc else u_m[1:-1])
                X_parts.append(np.column_stack([x_eval,
                                                np.full_like(x_eval, t_m, float),
                                                np.full_like(x_eval, nu, float)]))
                y_parts.append(u_m.reshape(-1,1))

            if k < K-1:
                x_next = x_mesh_list[k+1]
                R_end  = np.column_stack([R_full[:, j, -1] for j in range(TO+1)])
                f_divfac_next = project_R_end_to_new_mesh(x_mesh, R_end, x_next, TO)
                a_next = t_points[k+1]
                f_divfac_next[0,  0] = f_ic(a_next)
                f_divfac_next[-1, 0] = g_ic(a_next)
        if verbose:
            print(f"[dataset] x-t samples were added for epsilon={nu:.4g}.")

    X = np.vstack(X_parts)
    y = np.vstack(y_parts)
    return X, y

# -----------------------------------------------------------------------------
# 8) Standardization, neural-network architecture, and training utilities
# -----------------------------------------------------------------------------
class Standardizer:
    def fit(self, X, y):
        self.x_mu  = X.mean(axis=0, keepdims=True)
        self.x_std = X.std(axis=0, keepdims=True) + 1e-12
        self.y_mu  = y.mean(axis=0, keepdims=True)
        self.y_std = y.std(axis=0, keepdims=True) + 1e-12
        return self
    def transform_X(self, X): return (X - self.x_mu) / self.x_std
    def transform_y(self, y): return (y - self.y_mu) / self.y_std
    def inverse_y(self, yscaled): return yscaled * self.y_std + self.y_mu
    def state_dict(self):
        return {"x_mu": self.x_mu, "x_std": self.x_std, "y_mu": self.y_mu, "y_std": self.y_std}

class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=1, width=32, depth=5, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

def choose_batch_size(n_samples, target_steps_per_epoch=50, max_bs=4096, min_bs=256):
    bs = int(np.ceil(n_samples / target_steps_per_epoch))
    bs = min(max_bs, max(min_bs, bs))
    return bs

def make_loader_from_dataset(X, y, scaler, batch_size=None, shuffle=True):
    Xs = scaler.transform_X(X).astype(np.float32)
    ys = scaler.transform_y(y).astype(np.float32)
    if batch_size is None:
        batch_size = choose_batch_size(Xs.shape[0])
    ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return dl


def train_epochs(model, loader, epochs=100, lr=3e-3):
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    grad_clip = 1.0
    model.train()
    n = len(loader.dataset)

    start_time = time.time()  # Start timing

    for ep in range(1, epochs+1):
        run_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            run_loss += loss.item() * xb.size(0)
        sched.step()
        if ep % 100 == 0 or ep == 1 or ep == epochs:
            print(f"    [{ep:4d}/{epochs}] train MSE={run_loss/n:.4e}")

    elapsed = time.time() - start_time  # Elapsed training time in seconds
    print(f"Training time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")
    return elapsed

# -----------------------------------------------------------------------------
# 9) PINN-type residual evaluation by automatic differentiation
# -----------------------------------------------------------------------------
def _torch_scaler_tensors(scaler, device):
    x_mu  = torch.from_numpy(scaler.x_mu.astype(np.float32)).to(device)
    x_std = torch.from_numpy(scaler.x_std.astype(np.float32)).to(device)
    y_mu  = torch.from_numpy(scaler.y_mu.astype(np.float32)).to(device)
    y_std = torch.from_numpy(scaler.y_std.astype(np.float32)).to(device)
    return x_mu, x_std, y_mu, y_std

def pinn_residual_rms_for_v(model, scaler, v_val, nx=RES_NX, nt=RES_NT, device=None):
    """
    R = u_t - v u_xx + V u u_x + beta*(u - u^2)
    Return RMS(R). The derivatives u_x, u_xx, and u_t are computed by autograd.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    xs = torch.linspace(xb, xf, nx, device=device, dtype=torch.float32)
    ts = torch.linspace(0.0, tf, nt, device=device, dtype=torch.float32)
    Xg, Tg = torch.meshgrid(xs, ts, indexing='xy')  # (nx,nt)

    x = Xg.reshape(-1,1).clone().requires_grad_(True)
    t = Tg.reshape(-1,1).clone().requires_grad_(True)
    vcol = torch.full_like(x, float(v_val))

    x_mu, x_std, y_mu, y_std = _torch_scaler_tensors(scaler, device)
    Xraw = torch.cat([x, t, vcol], dim=1)            # (N,3)
    Xs   = (Xraw - x_mu) / x_std

    u_scaled = model(Xs)                             # (N,1)
    u = u_scaled * y_std + y_mu

    ones = torch.ones_like(u)
    u_x  = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_t  = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=False)[0]

    v_t = torch.tensor(float(v_val), device=device, dtype=torch.float32)
    R = u_t - v_t * u_xx + V * u * u_x + beta * (u - u*u)

    rms = torch.sqrt(torch.mean(R**2)).detach().item()
    return rms

# -----------------------------------------------------------------------------
# 10) Spatial residual indicator for adaptive mesh generation
# -----------------------------------------------------------------------------
def pinn_residual_per_x_nodes(model, scaler, v_val, a, b, x_nodes, n_time_probes=ADAPT_TIME_PROBES, device=None):
    """
    For each time segment [a,b]:
      - sample t_j = linspace(a,b,n_time_probes),
      - evaluate R(x,t_j) on the prescribed spatial nodes,
      - take the RMS in time to obtain one residual indicator per x-node.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    # Torch tensors used for autograd
    x = torch.from_numpy(np.asarray(x_nodes, dtype=np.float32)[:,None]).to(device).requires_grad_(True)
    ts = torch.linspace(float(a), float(b), n_time_probes, device=device, dtype=torch.float32)
    # Tensor-product grid: (nx, nt)
    Xg, Tg = torch.meshgrid(x.squeeze(1), ts, indexing='ij')
    x_flat = Xg.reshape(-1,1).clone().requires_grad_(True)
    t_flat = Tg.reshape(-1,1).clone().requires_grad_(True)
    vcol   = torch.full_like(x_flat, float(v_val))

    x_mu, x_std, y_mu, y_std = _torch_scaler_tensors(scaler, device)
    Xraw = torch.cat([x_flat, t_flat, vcol], dim=1)
    Xs   = (Xraw - x_mu) / x_std

    u_scaled = model(Xs)
    u = u_scaled * y_std + y_mu

    ones = torch.ones_like(u)
    u_x  = torch.autograd.grad(u, x_flat, grad_outputs=ones, create_graph=True)[0]
    u_t  = torch.autograd.grad(u, t_flat, grad_outputs=ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_flat, grad_outputs=torch.ones_like(u_x), create_graph=False)[0]

    v_t = torch.tensor(float(v_val), device=device, dtype=torch.float32)
    R = u_t - v_t * u_xx + V * u * u_x + beta * (u - u*u)  # (nx*nt,1)
    R = R.reshape(len(x_nodes), n_time_probes)
    rms_x = torch.sqrt(torch.mean(R**2, dim=1)).detach().cpu().numpy()
    return rms_x

# -----------------------------------------------------------------------------
# 11) Residual-driven adaptive nonuniform spatial mesh construction
# -----------------------------------------------------------------------------
def custom_adaptive_mesh(x_old, residuals, M, eps=1e-12):
    n_intervals = len(x_old) - 1
    interval_avg_res = []
    for i in range(n_intervals):
        ort = (abs(residuals[i]) + abs(residuals[i + 1])) / 2
        interval_avg_res.append(ort)
    interval_avg_res = np.array(interval_avg_res)

    log_res = np.log(interval_avg_res + eps)
    log_res = log_res - log_res.min()
    log_res = np.clip(log_res, 0, 10)
    total_indicator = log_res.sum()

    if total_indicator == 0:
        ratios = np.ones_like(log_res) / n_intervals
    else:
        ratios = log_res / total_indicator

    points_per_interval = np.round(ratios * M).astype(int)
    difference = M - points_per_interval.sum()

    while difference != 0:
        if difference > 0:
            idx = np.argmax(log_res)
            points_per_interval[idx] += 1
            difference -= 1
        else:
            idx = np.argmin(log_res)
            if points_per_interval[idx] > 0:
                points_per_interval[idx] -= 1
                difference += 1
            else:
                non_zero = np.where(points_per_interval > 0)[0]
                if len(non_zero) == 0:
                    break
                points_per_interval[non_zero[0]] -= 1
                difference += 1

    x_new = [x_old[0]]
    for i in range(n_intervals):
        if points_per_interval[i] > 0:
            points = np.linspace(x_old[i], x_old[i + 1], points_per_interval[i] + 2)[1:-1]
            x_new.extend(points)
        x_new.append(x_old[i + 1])

    x_new = np.unique(np.array(x_new))
    return x_new

# -----------------------------------------------------------------------------
# 12) Adaptive x-mesh list generation for a selected diffusion value
# -----------------------------------------------------------------------------
def make_adaptive_x_mesh_list_for_v(model, scaler, v_val, M_base, n_time_probes=ADAPT_TIME_PROBES, insert_points_factor=ADAPT_INSERT_POINTS_FACT):
    """
    For each segment [a_k,b_k]:
      1) start from a coarse uniform mesh,
      2) compute the residual indicator over the segment,
      3) insert additional points according to the residual distribution.
    The output is a list of segment-dependent spatial meshes.
    """
    K = len(t_points) - 1
    x_mesh_list_v = []
    M_insert = 22
    for k in range(K):
        a = t_points[k]; b = t_points[k+1]
        x_old = np.linspace(xb, xf, M_base+1)
        rms_x = pinn_residual_per_x_nodes(model, scaler, v_val, a, b, x_old, n_time_probes=n_time_probes)
        x_new = custom_adaptive_mesh(x_old, rms_x, M_insert, eps=1e-12)
        if not np.all(np.diff(x_new) > 0):
            raise RuntimeError("custom_adaptive_mesh did not produce a strictly increasing mesh.")
        x_mesh_list_v.append(x_new.astype(float))
    return x_mesh_list_v

# -----------------------------------------------------------------------------
# 13) Dataset generation for one selected diffusion value on adaptive meshes
# -----------------------------------------------------------------------------
def build_dataset_for_v_with_mesh_list(nu, x_mesh_list_v, include_bc=True, verbose=True):
    K = len(t_points) - 1
    if len(x_mesh_list_v) != K:
        raise ValueError(f"x_mesh_list_v must have length {K}.")
    set_diffusion(nu)

    X_parts, y_parts = [], []

    # Exact initial Taylor coefficients on the first segment mesh
    x_mesh0 = x_mesh_list_v[0]
    f_divfac_next = exact_f_divfac_at(x_mesh0, 0.0)

    for k in range(K):
        a = t_points[k]; b = t_points[k+1]
        x_mesh = x_mesh_list_v[k]
        if not np.all(np.diff(x_mesh) > 0):
            raise ValueError(f"x_mesh_list_v[{k}] must be strictly increasing.")
        Mcur   = len(x_mesh) - 1
        factor = 2.0 / (b - a)
        c = 0.5 * ((a + b) - (b - a) * np.cos(np.pi * i_vals / N))  # N+1
        A1, Acol = build_A_and_blocks(factor)
        f_divfac_current = f_divfac_next
        if f_divfac_current.shape[0] != (Mcur+1):
            raise ValueError(f"k={k}: f_divfac_current has incompatible shape {f_divfac_current.shape[0]} != {Mcur+1}")
        u_prev = initial_guess_from_profile(f_divfac_current, Mcur)
        obj = make_obj_for_step(x_mesh, a, b, A1, Acol, f_divfac_current)
        RS, infodict, ier, mesg = fsolve(obj, u_prev, xtol=1e-10, maxfev=500_000, full_output=True)
        if ier != 1:
            raise RuntimeError(f"[v={nu:.3g}, k={k}] fsolve failed: {mesg}")
        R_full = fill_R_from_u(RS, c, A1, Acol, f_divfac_current, Mcur)

        x_eval = x_mesh if include_bc else x_mesh[1:-1]
        # Include t=a_k
        u_a = f_divfac_current[:, 0] if include_bc else f_divfac_current[1:-1, 0]
        X_parts.append(np.column_stack([x_eval,
                                        np.full_like(x_eval, a, float),
                                        np.full_like(x_eval, nu, float)]))
        y_parts.append(u_a.reshape(-1,1))
        # Include t=c[1], ..., c[N]
        for m in range(N):
            t_m  = float(c[m+1])
            u_m  = R_full[:, 0, m]
            u_m  = (u_m if include_bc else u_m[1:-1])
            X_parts.append(np.column_stack([x_eval,
                                            np.full_like(x_eval, t_m, float),
                                            np.full_like(x_eval, nu, float)]))
            y_parts.append(u_m.reshape(-1,1))

        if k < K-1:
            x_next = x_mesh_list_v[k+1]
            R_end  = np.column_stack([R_full[:, j, -1] for j in range(TO+1)])
            f_divfac_next = project_R_end_to_new_mesh(x_mesh, R_end, x_next, TO)
            a_next = t_points[k+1]
            f_divfac_next[0,  0] = f_ic(a_next)
            f_divfac_next[-1, 0] = g_ic(a_next)

    if verbose:
        print(f"[dataset-adapt] epsilon={nu:.4g}: samples were added on segment-wise nonuniform meshes.")
    return np.vstack(X_parts), np.vstack(y_parts)

# -----------------------------------------------------------------------------
# 14) Error metrics and Excel report utilities
# -----------------------------------------------------------------------------
def exact_u_from_X(X):
    x = X[:,0]; t = X[:,1]; vv = X[:,2]
    d1 = -V/(4.0*vv)
    d2 = V/2.0 - (2.0*vv*beta)/V
    u  = 0.5 + 0.5*np.tanh(d1*x - d1*d2*t)
    return u.reshape(-1,1)

def print_metrics(title, y_true, y_pred):
    diff = y_pred - y_true
    mse  = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(diff)))
    mxe  = float(np.max(np.abs(diff)))
    print(f"{title}: RMSE={rmse:.3e} | MAE={mae:.3e} | MaxAbs={mxe:.3e}")

def print_metrics_by_v(title, X, y_true, y_pred, v_values):
    print(f"{title} by diffusion value:")
    for nu in v_values:
        mask = (np.abs(X[:,2]-nu) <= 1e-15) | (np.isclose(X[:,2], nu, rtol=1e-12, atol=1e-12))
        yt = y_true[mask]; yp = y_pred[mask]
        if yt.size == 0: 
            continue
        diff = yp - yt
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae  = float(np.mean(np.abs(diff)))
        mxe  = float(np.max(np.abs(diff)))
        print(f"  v={nu:.6g}: RMSE={rmse:.3e} | MAE={mae:.3e} | MaxAbs={mxe:.3e}")

def _metrics_dict(y_true, y_pred):
    diff = y_pred - y_true
    mse  = float(np.mean(diff**2))
    return {
        "count": int(y_true.size),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(diff))),
        "MaxAbs": float(np.max(np.abs(diff))),
    }

def export_errors_to_excel(X, y_exact, y_solver, y_pred, v_values,
                           path="errors_report_adaptive.xlsx", save_pointwise=True):
    overall = [
        dict(kind="Solver_vs_Exact", **_metrics_dict(y_exact, y_solver)),
        dict(kind="NN_vs_Exact",     **_metrics_dict(y_exact, y_pred)),
        dict(kind="NN_vs_Solver",    **_metrics_dict(y_solver, y_pred)),
    ]
    df_overall = pd.DataFrame(overall)
    rows = []
    for nu in v_values:
        mask = (np.abs(X[:,2]-nu) <= 1e-15) | (np.isclose(X[:,2], nu, rtol=1e-12, atol=1e-12))
        if not np.any(mask): 
            continue
        rows.append(dict(v=nu, kind="Solver_vs_Exact", **_metrics_dict(y_exact[mask], y_solver[mask])))
        rows.append(dict(v=nu, kind="NN_vs_Exact",     **_metrics_dict(y_exact[mask], y_pred[mask])))
        rows.append(dict(v=nu, kind="NN_vs_Solver",    **_metrics_dict(y_solver[mask], y_pred[mask])))
    df_by_v = pd.DataFrame(rows)
    if save_pointwise:
        df_point = pd.DataFrame({
            "x": X[:,0], "t": X[:,1], "v": X[:,2],
            "u_exact":  y_exact.ravel(),
            "u_solver": y_solver.ravel(),
            "u_nn":     y_pred.ravel(),
            "err_solver": (y_solver - y_exact).ravel(),
            "err_nn":     (y_pred   - y_exact).ravel(),
        })
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_overall.to_excel(writer, index=False, sheet_name="summary_overall")
            df_by_v.to_excel(writer, index=False, sheet_name="summary_by_v")
            if save_pointwise:
                if len(df_point) <= 1_048_000:
                    df_point.to_excel(writer, index=False, sheet_name="pointwise")
                else:
                    chunk = 1_000_000
                    for i in range(0, len(df_point), chunk):
                        df_point.iloc[i:i+chunk].to_excel(
                            writer, index=False, sheet_name=f"pointwise_{i//chunk+1}"
                        )
        print(f"Excel file written: {path}")
    except Exception as e:
        print(f"Excel export failed ({e}). Saving as CSV files...")
        df_overall.to_csv("summary_overall.csv", index=False)
        df_by_v.to_csv("summary_by_v.csv", index=False)
        if save_pointwise:
            df_point.to_csv("errors_pointwise.csv", index=False)
        print("CSV files: summary_overall.csv, summary_by_v.csv, errors_pointwise.csv")

# -----------------------------------------------------------------------------
# 15) Visualization utilities: exact solution, approximate solution, and error
# -----------------------------------------------------------------------------
@torch.no_grad()
def grid_predict(model, scaler, diffusion_value, nx=200, nt=150):
    """
    Evaluate the trained MLP and the analytical solution on a uniform (x, t) grid.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network approximating u(x,t,v).
    scaler : Standardizer
        Input-output scaler used during training.
    diffusion_value : float
        Diffusion coefficient to be used in the visualization.
    nx, nt : int
        Number of spatial and temporal grid points for the heatmaps.

    Returns
    -------
    Xg, Tg : numpy.ndarray
        Meshgrid arrays used only for axis information.
    U_pred : numpy.ndarray
        Approximate solution predicted by the neural network.
    U_exact : numpy.ndarray
        Analytical solution evaluated on the same grid.
    """
    xs = np.linspace(xb, xf, nx)
    ts = np.linspace(0.0, tf, nt)
    Xg, Tg = np.meshgrid(xs, ts)
    vv = np.full_like(Xg, float(diffusion_value), dtype=float)
    X_in = np.column_stack([Xg.ravel(), Tg.ravel(), vv.ravel()])

    device = next(model.parameters()).device
    Xt = torch.from_numpy(scaler.transform_X(X_in).astype(np.float32)).to(device)

    y_mu = torch.from_numpy(scaler.y_mu.astype(np.float32)).to(device)
    y_std = torch.from_numpy(scaler.y_std.astype(np.float32)).to(device)

    U_pred = (model(Xt) * y_std + y_mu).cpu().numpy().reshape(nt, nx)
    U_exact = exact_u_from_X(X_in).reshape(nt, nx)
    return Xg, Tg, U_pred, U_exact


def plot_solution_heatmaps(model, scaler, diffusion_value, nx=200, nt=150):
    """
    Plot only the three requested heatmaps: exact solution, approximate solution,
    and absolute error.
    """
    Xg, Tg, U_pred, U_exact = grid_predict(
        model, scaler, diffusion_value, nx=nx, nt=nt
    )
    error = np.abs(U_pred - U_exact)

    v_power = int(round(math.log2(float(diffusion_value))))
    v_label = f"$\\epsilon = 2^{{{v_power}}}$"

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    # (a) Exact solution
    im0 = ax[0].imshow(
        U_exact, origin='lower', extent=[xb, xf, 0.0, tf], aspect='auto'
    )
    ax[0].set_title(f"{v_label} Exact Solution", fontsize=15)
    ax[0].set_xlabel('$x$', fontsize=15)
    ax[0].set_ylabel('$t$', fontsize=15)
    fig.colorbar(im0, ax=ax[0])

    # (b) Approximate solution obtained from the trained neural network
    im1 = ax[1].imshow(
        U_pred, origin='lower', extent=[xb, xf, 0.0, tf], aspect='auto'
    )
    ax[1].set_title(f"{v_label} Approximate Solution", fontsize=15)
    ax[1].set_xlabel('$x$', fontsize=15)
    ax[1].set_ylabel('$t$', fontsize=15)
    fig.colorbar(im1, ax=ax[1])

    # (c) Absolute error heatmap
    im2 = ax[2].imshow(
        error, origin='lower', extent=[xb, xf, 0.0, tf], aspect='auto'
    )
    ax[2].set_title(f"{v_label} Error Heatmap", fontsize=15)
    ax[2].set_xlabel('$x$', fontsize=15)
    ax[2].set_ylabel('$t$', fontsize=15)
    fig.colorbar(im2, ax=ax[2])

    print(f"Maximum heatmap error for epsilon={diffusion_value:.6g}: {np.max(error):.4e}")
    return fig, ax


# -----------------------------------------------------------------------------
# 16) Main workflow
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(1234); np.random.seed(1234)

    # Baseline uniform mesh list used for the initial training dataset
    M = 32
    x_mesh_list_uniform = [np.linspace(xb, xf, M+1) for _ in range(len(t_points)-1)]

    # Initial log-uniform diffusion-parameter set
    v_init = 2.0 ** (-np.linspace(6, 1, 40))
    v_min, v_max = float(v_init.min()), float(v_init.max())

    # Generate the initial supervised dataset using all x- and t-nodes on the uniform mesh
    X, y_solver = build_dataset_multi_v(v_init, x_mesh_list_uniform, include_bc=True, verbose=True)
    print("Dataset size:", X.shape, y_solver.shape, "  (X=[x,t,v], y=[u])")
    K = len(t_points) - 1
    total_expected = (M+1) * K * (N+1) * len(v_init)
    print(f"Expected number of samples ≈ {(M+1)} * {K} * {(N+1)} * {len(v_init)} = {total_expected}")

    # Compare numerical solver data against the exact solution
    y_exact = exact_u_from_X(X)
    print_metrics("Solver vs Exact (all data)", y_exact, y_solver)
    print_metrics_by_v("Solver vs Exact", X, y_exact, y_solver, v_init)

    # Build the scaler and the MLP model
    scaler = Standardizer().fit(X, y_solver)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=3, out_dim=1, width=64, depth=4, act=nn.Tanh).to(device)

    # Stage 0: initial purely data-driven training
    print("\n[Stage-0] Initial data-driven training starts")
    loader = make_loader_from_dataset(X, y_solver, scaler)
    train_epochs(model, loader, epochs=INITIAL_EPOCHS, lr=3e-3)

    # Adaptive loop: select difficult diffusion values, enrich data, then retrain
    used_vs = set(np.unique(X[:,2]))
    for cyc in range(1, ADAPT_CYCLES+1):
        print(f"\n[Stage-{cyc}] Adaptive cycle {cyc}/{ADAPT_CYCLES}: PINN-based selection")

        # Candidate diffusion-value pool
        v_pool = 2.0 ** (-np.linspace(6, 1, V_POOL_SIZE))
        v_pool = [float(vv) for vv in v_pool if float(vv) not in used_vs]
        if len(v_pool) == 0:
            print("  [select] No new diffusion values are available; adaptive process is complete.")
            break

        chosen_vs = []
        # Score the current model without additional training
        scored = []
        for nu in v_pool:
            rms = pinn_residual_rms_for_v(model, scaler, nu, device=device)
            scored.append((rms, float(nu)))
        scored.sort(reverse=True)
        # Select the parameters with the largest residual scores.
        chosen_vs = [v for _, v in scored[:min(TOP_K, len(scored))]]
        print("  [select] most difficult epsilon values:", ", ".join(f"{v:.4g}" for v in chosen_vs))

        # Enrich the dataset using segment-wise adaptive nonuniform meshes
        X_new_all = []; y_new_all = []
        for nu in chosen_vs:
            x_mesh_list_v = make_adaptive_x_mesh_list_for_v(
                model, scaler, nu, M_base=10,
                n_time_probes=ADAPT_TIME_PROBES,
                insert_points_factor=ADAPT_INSERT_POINTS_FACT
            )
            X_new_v, y_new_v = build_dataset_for_v_with_mesh_list(nu, x_mesh_list_v, include_bc=True, verbose=False)
            X_new_all.append(X_new_v); y_new_all.append(y_new_v)

        if len(X_new_all) > 0:
            X_new = np.vstack(X_new_all); y_new = np.vstack(y_new_all)
            X = np.vstack([X, X_new]);   y_solver = np.vstack([y_solver, y_new])
            for vv in chosen_vs: used_vs.add(float(vv))
            print(f"  [augment] number of added epsilon values: {len(chosen_vs)}; new dataset size: {X.shape[0]}")
        else:
            print("  [augment] No additional data were generated.")

        # Retrain using the enriched dataset
        print(f"  [train] {EPOCHS_PER_CYCLE} training epochs")
        loader = make_loader_from_dataset(X, y_solver, scaler)
        train_epochs(model, loader, epochs=EPOCHS_PER_CYCLE, lr=3e-3)

        # Intermediate metric evaluation without gradient tracking
        Xt = torch.from_numpy(scaler.transform_X(X).astype(np.float32)).to(device)
        y_std_t = torch.from_numpy(scaler.y_std.astype(np.float32)).to(device)
        y_mu_t  = torch.from_numpy(scaler.y_mu.astype(np.float32)).to(device)
        with torch.no_grad():
            y_pred_all = (model(Xt) * y_std_t + y_mu_t).cpu().numpy()
        print_metrics("  Intermediate: NN vs Solver", y_solver, y_pred_all)

    # Final evaluation
    print("\n[Final] Final evaluation and saving")
    Xt = torch.from_numpy(scaler.transform_X(X).astype(np.float32)).to(device)
    y_std_t = torch.from_numpy(scaler.y_std.astype(np.float32)).to(device)
    y_mu_t  = torch.from_numpy(scaler.y_mu.astype(np.float32)).to(device)
    with torch.no_grad():
        y_pred = (model(Xt) * y_std_t + y_mu_t).cpu().numpy()
    print_metrics("NN vs Exact   (all data)", exact_u_from_X(X), y_pred)
    print_metrics("NN vs Solver  (all data)", y_solver, y_pred)

    # Save the trained model and normalization parameters
    payload = {"model_state": model.state_dict(),
               "arch": {"width": 16, "depth": 2, "act": "tanh"},
               "scaler": scaler.state_dict()}
    torch.save(payload, "u_x_t_v_data_driven_adaptive.pt")
    print("Model saved -> u_x_t_v_data_driven_adaptive.pt")

    # ---- Excel report ----
    export_errors_to_excel(
        X, exact_u_from_X(X), y_solver, y_pred, np.unique(X[:,2]),
        path="errors_report_adaptive.xlsx", save_pointwise=True
    )

    # ---- Final visualization ----
    # Only the requested figure is produced: exact solution, approximate
    # solution, and absolute error heatmap for the selected diffusion value.
    diffusion_for_plot = float(2.0**-3)
    fig, _ = plot_solution_heatmaps(
        model, scaler, diffusion_for_plot, nx=200, nt=150
    )
    fig.savefig("solution_exact_approx_error_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

