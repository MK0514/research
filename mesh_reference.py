"""
mesh_reference.py
=================
Stable reference-style body-fitted structured mesh.

Unlike the team version, this script does not draw or save a Cartesian
background grid.  It creates one continuous body-fitted structured band around
nose cap -> rocket body -> base cap, then mirrors it for visualization.  This is
closer to the paper-style mesh appearance while keeping the stable normal/cap
logic from the teammate version.
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

EPS = 1.0e-30


def robust_norm(a):
    a = np.abs(np.asarray(a, dtype=float))
    hi = np.percentile(a, 95) if a.size else 1.0
    if hi < EPS:
        return np.zeros_like(a)
    return np.clip(a / hi, 0.0, 1.0)


def load_profile(path):
    bx, br = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            bx.append(float(row["x_m"]))
            br.append(abs(float(row["r_m"])))
    bx = np.asarray(bx)
    br = np.asarray(br)
    idx = np.argsort(bx)
    bx, br = bx[idx], br[idx]
    mask = br > br.max() * 1e-3
    xb, rb = bx[mask], br[mask]
    L = xb[-1] - xb[0]
    Rmax = rb.max()
    print(f"[Profile] pts={len(xb)} L={L:.4f}m Rmax={Rmax:.4f}m nose_r={rb[0]:.6f}m tail_r={rb[-1]:.6f}m")
    return bx, br, xb, rb, float(L), float(Rmax)


def compute_n_layers(first_cell, growth, target):
    eta = [0.0]
    dh = first_cell
    n = 0
    while eta[-1] < target:
        eta.append(eta[-1] + dh)
        dh *= growth
        n += 1
        if n > 5000:
            raise ValueError("Cannot reach target thickness")
    return n


def build_eta(first_cell, growth, n_layers):
    eta = [0.0]
    dh = first_cell
    for _ in range(n_layers):
        eta.append(eta[-1] + dh)
        dh *= growth
    return np.asarray(eta)


def adaptive_body_resample(xb, rb, n_body):
    ds = np.sqrt(np.diff(xb)**2 + np.diff(rb)**2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    slope = np.gradient(rb, xb, edge_order=1)
    d2r = np.gradient(slope, xb, edge_order=1)
    curv = np.abs(d2r) / np.maximum((1.0 + slope*slope)**1.5, EPS)
    theta = np.unwrap(np.arctan2(np.gradient(rb), np.gradient(xb)))
    turn = np.abs(np.gradient(theta, s + EPS, edge_order=1))
    xi = s / max(s[-1], EPS)
    nose = np.exp(-(xi / 0.08)**2)
    aft = np.exp(-((xi - 1.0) / 0.08)**2)
    density = 1.0 + 0.9*robust_norm(curv) + 0.6*robust_norm(turn) + 0.2*nose + 0.35*aft
    density = np.minimum(density, 2.0)
    for _ in range(4):
        dn = density.copy()
        dn[1:-1] = 0.25*density[:-2] + 0.50*density[1:-1] + 0.25*density[2:]
        density = dn
    se = np.zeros_like(s)
    se[1:] = np.cumsum(ds * 0.5*(density[:-1] + density[1:]))
    su = np.linspace(0.0, se[-1], n_body)
    return np.interp(su, se, xb), np.interp(su, se, rb)


def smooth_normals(nx, nr, passes):
    nx = nx.copy()
    nr = nr.copy()
    for _ in range(passes):
        nxn = nx.copy()
        nrn = nr.copy()
        nxn[1:-1] = 0.05*nx[:-2] + 0.90*nx[1:-1] + 0.05*nx[2:]
        nrn[1:-1] = 0.05*nr[:-2] + 0.90*nr[1:-1] + 0.05*nr[2:]
        mag = np.sqrt(nxn*nxn + nrn*nrn) + EPS
        nx, nr = nxn/mag, nrn/mag
    return nx, nr


def build_body_fitted_band(xb, rb, first_cell, growth, target_thickness, n_body, n_cap, smooth_iter):
    n_layers = compute_n_layers(first_cell, growth, target_thickness)
    eta = build_eta(first_cell, growth, n_layers)
    thickness = float(eta[-1])
    print(f"[Eta] layers={n_layers} Ni={len(eta)} first={eta[1]:.3e}m thickness={thickness:.5f}m")

    xs, rs = adaptive_body_resample(xb, rb, n_body)

    # Use the stable cap topology from the teammate version: vertical nose/base
    # wall caps, with horizontal outward normals at the cap endpoints.  This is
    # much more robust for Falcon9's tiny nose radius than a synthetic round cap.
    nose_x = np.full(n_cap, xs[0])
    nose_r = np.linspace(0.0, rs[0], n_cap)
    nose_nx = np.full(n_cap, -1.0)
    nose_nr = np.zeros(n_cap)

    tx = np.gradient(xs)
    tr = np.gradient(rs)
    mag = np.sqrt(tx*tx + tr*tr) + EPS
    tx, tr = tx/mag, tr/mag
    body_nx = -tr
    body_nr = tx
    if body_nr[len(body_nr)//2] < 0:
        body_nx *= -1
        body_nr *= -1

    tail_x = np.full(n_cap, xs[-1])
    tail_r = np.linspace(rs[-1], 0.0, n_cap)
    tail_nx = np.full(n_cap, 1.0)
    tail_nr = np.zeros(n_cap)

    inner_x = np.concatenate([nose_x, xs[1:-1], tail_x])
    inner_r = np.concatenate([nose_r, rs[1:-1], tail_r])
    nx = np.concatenate([nose_nx, body_nx[1:-1], tail_nx])
    nr = np.concatenate([nose_nr, body_nr[1:-1], tail_nr])

    nx, nr = smooth_normals(nx, nr, smooth_iter)
    nx[0], nr[0] = -1.0, 0.0
    nx[-1], nr[-1] = 1.0, 0.0

    X = inner_x[None, :] + eta[:, None]*nx[None, :]
    R = inner_r[None, :] + eta[:, None]*nr[None, :]
    R = np.maximum(R, 0.0)

    # Keep far envelope visually smoother without changing near-wall layers.
    X[-1, :] = smooth_curve(X[-1, :], 12)
    R[-1, :] = smooth_curve(R[-1, :], 12)
    R[-1, :] = np.maximum(R[-1, :], inner_r + 0.9*target_thickness)

    j_body0 = n_cap
    j_body1 = n_cap + len(xs[1:-1]) - 1
    return X, R, eta, xs, rs, j_body0, j_body1


def smooth_curve(a, passes):
    a = a.copy()
    left, right = a[0], a[-1]
    for _ in range(passes):
        b = a.copy()
        b[1:-1] = 0.25*a[:-2] + 0.50*a[1:-1] + 0.25*a[2:]
        a = b
        a[0], a[-1] = left, right
    return a


def metrics(X, R):
    dxi = X[1:,:-1] - X[:-1,:-1]
    dri = R[1:,:-1] - R[:-1,:-1]
    dxj = X[:-1,1:] - X[:-1,:-1]
    drj = R[:-1,1:] - R[:-1,:-1]
    signed = dxi*drj - dri*dxj
    orient = np.sign(np.nanmedian(signed)) if signed.size else 1.0
    if orient == 0:
        orient = 1.0
    signed *= orient
    area = np.abs(signed)
    li = np.sqrt(dxi*dxi + dri*dri + EPS)
    lj = np.sqrt(dxj*dxj + drj*drj + EPS)
    asp = np.maximum(li, lj) / np.maximum(np.minimum(li, lj), EPS)
    fa = asp[np.isfinite(asp)]
    return {
        "cells": int((X.shape[0]-1)*(X.shape[1]-1)),
        "neg": int((signed <= 0).sum()),
        "area_min": float(area.min()),
        "area_max": float(area.max()),
        "p95": float(np.percentile(fa, 95)),
        "p99": float(np.percentile(fa, 99)),
    }


def intrusion_count(X, R, xb, rb):
    xx = X[1:, :]
    rr = R[1:, :]
    r_body = np.interp(xx, xb, rb, left=0.0, right=0.0)
    inx = (xx >= xb[0]) & (xx <= xb[-1])
    return int((inx & (rr < r_body - 1e-6)).sum())


def split_blocks(X, R, j_body0, j_body1):
    # 3-block-compatible output: nose cap / body / base cap.
    X1, R1 = X[:, :j_body0+1], R[:, :j_body0+1]
    X2, R2 = X[:, j_body0:j_body1+1], R[:, j_body0:j_body1+1]
    X3, R3 = X[:, j_body1:], R[:, j_body1:]
    return X1, R1, X2, R2, X3, R3


def plot_mesh(X, R, bx_all, br_all, q, out_png):
    fig = plt.figure(figsize=(24, 12))
    ax = fig.add_subplot(1, 1, 1)
    Ni, Nj = X.shape
    si = max(1, Ni//85)
    sj = max(1, Nj//180)
    segs = []
    for i in range(0, Ni, si):
        segs.append(list(zip(X[i, :], R[i, :])))
        segs.append(list(zip(X[i, :], -R[i, :])))
    for j in range(0, Nj, sj):
        segs.append(list(zip(X[:, j], R[:, j])))
        segs.append(list(zip(X[:, j], -R[:, j])))
    ax.add_collection(LineCollection(segs, colors="black", lw=0.42, alpha=0.72))
    ax.fill_between(bx_all, br_all, -br_all, color="white", zorder=5)
    ax.plot(bx_all, br_all, color="black", lw=2.0, zorder=6)
    ax.plot(bx_all, -br_all, color="black", lw=2.0, zorder=6)
    ax.axhline(0.0, color="0.35", lw=0.7, ls="--", alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(X.min(), X.max())
    ymax = max(R.max(), br_all.max()*2.5)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("r [m]")
    ax.set_title(f"Reference-style body-fitted structured band | cells={q['cells']:,} neg={q['neg']} AR99={q['p99']:.1f}")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out_png}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--first_cell", type=float, default=5e-5)
    p.add_argument("--growth", type=float, default=1.08)
    p.add_argument("--target_thickness", type=float, default=None)
    p.add_argument("--nj_body", type=int, default=800)
    p.add_argument("--nj_cap", type=int, default=90)
    p.add_argument("--smooth_iter", type=int, default=3000)
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.splitext(os.path.basename(args.profile))[0].replace("_profile", "")
    out = args.output or (base + "_reference")
    bx_all, br_all, xb, rb, L, Rmax = load_profile(args.profile)
    target = args.target_thickness if args.target_thickness is not None else 1.35*Rmax
    X, R, eta, xs, rs, j0, j1 = build_body_fitted_band(
        xb, rb, args.first_cell, args.growth, target,
        args.nj_body, args.nj_cap, args.smooth_iter,
    )
    q = metrics(X, R)
    inside = intrusion_count(X, R, xs, rs)
    print(f"[Quality] cells={q['cells']:,} neg={q['neg']} inside={inside} area=[{q['area_min']:.3e},{q['area_max']:.3e}] AR95={q['p95']:.1f} AR99={q['p99']:.1f}")
    X1, R1, X2, R2, X3, R3 = split_blocks(X, R, j0, j1)
    np.savez_compressed(
        out + ".npz",
        X=X, R=R, X1=X1, R1=R1, X2=X2, R2=R2, X3=X3, R3=R3,
        body_x=xs, body_r=rs, eta=eta,
        q_negative_cells=np.array(q["neg"]), q_body_intrusion=np.array(inside),
        q_p95_aspect=np.array(q["p95"]), q_p99_aspect=np.array(q["p99"]),
        p_target_thickness=np.array(target),
    )
    print(f"[Save] {out}.npz")
    plot_mesh(X, R, bx_all, br_all, q, out + "_mesh.png")
    print("[Done]")


if __name__ == "__main__":
    main()
