
"""
Phase 1 — 1D Heat Equation (Periodic) data generator for "two shifted grids" fusion.

Generates datasets produced by the SAME nominal 2nd-order accurate scheme (CN + 2nd-order FD Laplacian),
but sampled on two different uniform grids:
  - Grid A: x_i = i/N
  - Grid B: x_i = (i+0.5)/N

The PDE truth is generated analytically using a truncated Fourier series initial condition.

Outputs one .npz per resolution N containing:
  xA:     [N]
  xB:     [N]
  t_snap: [nt]
  alpha:  [S]
  c0:     [S]          (constant mode)
  a_sin:  [S,K]        sine coefficients
  b_cos:  [S,K]        cosine coefficients
  uA:     [S,nt,N]     CN solution on grid A at t_snap
  uB:     [S,nt,N]     CN solution on grid B at t_snap

We can later compute exact truth anywhere using (alpha, c0, a_sin, b_cos).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Sequence

import numpy as np


# ============================================================
# 1) Exact periodic heat solution with Fourier IC
# ============================================================

@dataclass
class FourierIC:
    c0: float                 # constant mode
    a_sin: np.ndarray         # [K]
    b_cos: np.ndarray         # [K]

def sample_random_ic(K: int,
                     rng: np.random.Generator,
                     coeff_scale: float = 1.0,
                     include_c0: bool = True) -> FourierIC:
    """u0(x) = c0 + sum_{k=1..K} (a_k sin(2πkx) + b_k cos(2πkx))."""
    c0 = float(coeff_scale * rng.standard_normal()) if include_c0 else 0.0
    a = coeff_scale * rng.standard_normal(K)
    b = coeff_scale * rng.standard_normal(K)
    return FourierIC(c0=c0, a_sin=a, b_cos=b)

def eval_u0_periodic(ic: FourierIC, x: np.ndarray) -> np.ndarray:
    """Evaluate u0(x) on x in [0,1). x: [...]. returns same shape."""
    x = np.asarray(x)
    out = ic.c0 * np.ones_like(x, dtype=np.float64)
    K = ic.a_sin.shape[0]
    # broadcast-friendly computation
    k = np.arange(1, K + 1, dtype=np.float64)[:, None]  # [K,1]
    x2 = x.reshape(1, -1)                                # [1,M]
    out_flat = out.reshape(-1)
    # compute modes on flattened x for speed
    s = np.sin(2.0 * np.pi * k * x2)                     # [K,M]
    c = np.cos(2.0 * np.pi * k * x2)                     # [K,M]
    out_flat += (ic.a_sin[:, None] * s + ic.b_cos[:, None] * c).sum(axis=0)
    return out_flat.reshape(x.shape)

def eval_u_exact_periodic(ic: FourierIC, alpha: float, x: np.ndarray, t: float) -> np.ndarray:
    """Exact solution u(x,t) for periodic heat equation with Fourier IC."""
    x = np.asarray(x)
    out = ic.c0 * np.ones_like(x, dtype=np.float64)
    K = ic.a_sin.shape[0]
    k = np.arange(1, K + 1, dtype=np.float64)
    decay = np.exp(-alpha * (2.0 * np.pi * k) ** 2 * t)   # [K]
    k2 = k[:, None]
    x2 = x.reshape(1, -1)
    s = np.sin(2.0 * np.pi * k2 * x2)
    c = np.cos(2.0 * np.pi * k2 * x2)
    out_flat = out.reshape(-1)
    out_flat += ((decay * ic.a_sin)[:, None] * s + (decay * ic.b_cos)[:, None] * c).sum(axis=0)
    return out_flat.reshape(x.shape)


# ============================================================
# 2) 2nd-order scheme data: Crank–Nicolson + 2nd-order Laplacian
#     (periodic) using FFT diagonalization of the discrete Laplacian
# ============================================================

def cn_periodic_snapshots(u0: np.ndarray,
                          alpha: float,
                          x: np.ndarray,
                          t_snap: Sequence[float],
                          dt_target: float,
                          r_target: float = None) -> Tuple[np.ndarray, float]:
    """
    Compute CN solution snapshots at times t_snap for initial u0 sampled on uniform periodic grid x.

    Implementation detail:
      On uniform periodic grid, the discrete Laplacian is diagonal in Fourier space.
      CN update multiplier for mode k:
        m_k = (1 + 0.5*alpha*dt*λ_k) / (1 - 0.5*alpha*dt*λ_k),
      where λ_k is eigenvalue of discrete Laplacian (including 1/h^2).

    Returns:
      u_snap: [nt, N]
      dt: actual dt used (adjusted so that T_final is an integer multiple of dt)
    """
    u0 = np.asarray(u0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    h = float(1.0 / N)  # assumes uniform grid spanning [0,1)

    t_snap = np.asarray(list(t_snap), dtype=np.float64)
    if np.any(t_snap < 0):
        raise ValueError("t_snap must be nonnegative.")
    T_final = float(np.max(t_snap)) if t_snap.size else 0.0
    if T_final == 0.0:
        return np.zeros((len(t_snap), N), dtype=np.float64), 0.0

    # Adjust dt so that T_final is exactly nsteps*dt
    nsteps = int(np.ceil(T_final / dt_target))
    nsteps = max(nsteps, 1)
    dt = T_final / nsteps

    # map snapshots to integer step indices
    n_idx = np.rint(t_snap / dt).astype(int)
    if np.max(np.abs(t_snap - n_idx * dt)) > 1e-10:
        # should be rare due to rint; still warn by printing once.
        pass

    # Fourier multipliers
    # Discrete Laplacian eigenvalues for periodic 2nd-order FD:
    # λ_k = (-4 sin^2(pi k / N)) / h^2
    k = np.arange(N, dtype=np.float64)
    lam = (-4.0 * np.sin(np.pi * k / N) ** 2) / (h ** 2)

    # CN multiplier
    num = (1.0 + 0.5 * alpha * dt * lam)
    den = (1.0 - 0.5 * alpha * dt * lam)
    m = num / den  # [N]

    # FFT of initial condition
    uhat0 = np.fft.fft(u0)

    # Build snapshots by direct powering
    nt = len(t_snap)
    u_snap = np.empty((nt, N), dtype=np.float64)
    for j, n in enumerate(n_idx):
        uhat = uhat0 * (m ** n)
        u = np.fft.ifft(uhat).real
        u_snap[j, :] = u

    return u_snap, dt


# ============================================================
# 3) Data generation
# ============================================================

def make_grids(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform periodic grids A and shifted B."""
    h = 1.0 / N
    xA = np.arange(N, dtype=np.float64) * h
    xB = (np.arange(N, dtype=np.float64) + 0.5) * h
    # wrap into [0,1)
    xB = np.mod(xB, 1.0)
    return xA, xB

def generate_dataset_for_N(
    N: int,
    num_samples: int,
    K: int,
    t_snap: Sequence[float],
    alpha_range: Tuple[float, float],
    coeff_scale: float,
    seed: int,
    dt_r_target: float,
    dtype_store: str = "float32",
) -> dict:
    """
    Generate dataset dict (in-memory) for a single N.
    dt is chosen as dt_target = r_target * h^2 / alpha_max to keep temporal error consistent.
    """
    alpha_min, alpha_max = alpha_range
    if alpha_min <= 0 or alpha_max <= 0 or alpha_min > alpha_max:
        raise ValueError("alpha_range must be positive and alpha_min <= alpha_max")

    rng = np.random.default_rng(seed)

    xA, xB = make_grids(N)
    h = 1.0 / N
    dt_target = dt_r_target * h * h / alpha_max

    nt = len(t_snap)

    alpha = np.empty((num_samples,), dtype=np.float64)
    c0 = np.empty((num_samples,), dtype=np.float64)
    a_sin = np.empty((num_samples, K), dtype=np.float64)
    b_cos = np.empty((num_samples, K), dtype=np.float64)
    uA = np.empty((num_samples, nt, N), dtype=np.float64)
    uB = np.empty((num_samples, nt, N), dtype=np.float64)

    dt_used = None
    for s in range(num_samples):
        ic = sample_random_ic(K=K, rng=rng, coeff_scale=coeff_scale, include_c0=True)
        a = alpha_min + (alpha_max - alpha_min) * rng.random()
        alpha[s] = a
        c0[s] = ic.c0
        a_sin[s, :] = ic.a_sin
        b_cos[s, :] = ic.b_cos

        u0A = eval_u0_periodic(ic, xA)
        u0B = eval_u0_periodic(ic, xB)

        uA_snap, dtA = cn_periodic_snapshots(u0A, a, xA, t_snap, dt_target=dt_target)
        uB_snap, dtB = cn_periodic_snapshots(u0B, a, xB, t_snap, dt_target=dt_target)
        if abs(dtA - dtB) > 1e-14:
            raise RuntimeError("dt mismatch between grids A and B (should not happen).")

        if dt_used is None:
            dt_used = dtA

        uA[s, :, :] = uA_snap
        uB[s, :, :] = uB_snap

    # cast to storage dtype
    store_dtype = np.float32 if dtype_store == "float32" else np.float64

    data = dict(
        N=N,
        K=K,
        num_samples=num_samples,
        t_snap=np.asarray(t_snap, dtype=np.float64),
        xA=xA.astype(store_dtype),
        xB=xB.astype(store_dtype),
        alpha=alpha.astype(store_dtype),
        c0=c0.astype(store_dtype),
        a_sin=a_sin.astype(store_dtype),
        b_cos=b_cos.astype(store_dtype),
        uA=uA.astype(store_dtype),
        uB=uB.astype(store_dtype),
        dt_used=float(dt_used) if dt_used is not None else 0.0,
        dt_target=float(dt_target),
        alpha_min=float(alpha_min),
        alpha_max=float(alpha_max),
        coeff_scale=float(coeff_scale),
        scheme="Crank–Nicolson (time) + 2nd-order FD Laplacian (space), periodic, FFT diagonalization",
    )
    return data


def sanity_check_one_sample(data: dict, sample_idx: int = 0, N_ref: int = 4096) -> dict:
    """
    Compare CN solution vs exact truth for one sample at the dataset's snapshot times.
    Returns errors for quick inspection.
    """
    N = int(data["N"])
    K = int(data["K"])
    xA = np.asarray(data["xA"], dtype=np.float64)
    t_snap = np.asarray(data["t_snap"], dtype=np.float64)
    alpha = float(np.asarray(data["alpha"], dtype=np.float64)[sample_idx])
    c0 = float(np.asarray(data["c0"], dtype=np.float64)[sample_idx])
    a_sin = np.asarray(data["a_sin"], dtype=np.float64)[sample_idx]
    b_cos = np.asarray(data["b_cos"], dtype=np.float64)[sample_idx]
    ic = FourierIC(c0=c0, a_sin=a_sin, b_cos=b_cos)

    # exact on grid A
    errs = []
    for j, t in enumerate(t_snap):
        u_exact_A = eval_u_exact_periodic(ic, alpha, xA, float(t))
        u_cn_A = np.asarray(data["uA"], dtype=np.float64)[sample_idx, j]
        e = u_cn_A - u_exact_A
        l2 = np.sqrt(np.mean(e**2))
        linf = np.max(np.abs(e))
        errs.append((float(t), float(l2), float(linf)))

    return {"sample_idx": sample_idx, "errs": errs, "note": "Errors reflect discretization (CN+FD) vs exact PDE."}


# ============================================================
# 4) CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="heat_phase1_data")
    p.add_argument("--Ns", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--t_snap", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--alpha_min", type=float, default=0.5)
    p.add_argument("--alpha_max", type=float, default=2.0)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--r_target", type=float, default=0.4,
                   help="dt_target = r_target*h^2/alpha_max; smaller => more accurate time integration.")
    p.add_argument("--dtype_store", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--sanity_check", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for N in args.Ns:
        print(f"\n=== Generating N={N} (samples={args.num_samples}, K={args.K}) ===")
        data = generate_dataset_for_N(
            N=N,
            num_samples=args.num_samples,
            K=args.K,
            t_snap=args.t_snap,
            alpha_range=(args.alpha_min, args.alpha_max),
            coeff_scale=args.coeff_scale,
            seed=args.seed + int(N),  # vary per N deterministically
            dt_r_target=args.r_target,
            dtype_store=args.dtype_store,
        )

        fname = f"heat_periodic_AB_CN2_FD2_N{N}_K{args.K}_S{args.num_samples}_T{max(args.t_snap):g}.npz"
        fpath = os.path.join(args.out_dir, fname)
        np.savez_compressed(fpath, **data)
        print("Saved:", fpath)
        print(f"dt_target={data['dt_target']:.3e}, dt_used={data['dt_used']:.3e}")

        if args.sanity_check:
            sc = sanity_check_one_sample(data, sample_idx=0)
            print("Sanity check (CN vs exact PDE on grid A):")
            for (t, l2, linf) in sc["errs"]:
                print(f"  t={t:7.4f} | L2={l2:.3e} | Linf={linf:.3e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
