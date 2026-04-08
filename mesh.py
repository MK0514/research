import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1.0e-12


def normalize_pair(xv: np.ndarray, rv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.sqrt(xv * xv + rv * rv) + EPS
    return xv / mag, rv / mag


def robust_norm(values: np.ndarray) -> np.ndarray:
    arr = np.abs(np.asarray(values, dtype=float))
    hi = np.percentile(arr, 95.0) if arr.size else 1.0
    if hi < EPS:
        return np.zeros_like(arr)
    return np.clip(arr / hi, 0.0, 1.0)


def load_profile(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    bx, br = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            bx.append(float(row["x_m"]))
            br.append(abs(float(row["r_m"])))

    bx = np.asarray(bx, dtype=float)
    br = np.asarray(br, dtype=float)
    order = np.argsort(bx)
    bx = bx[order]
    br = br[order]

    body_mask = br > br.max() * 1.0e-3
    bx_body = bx[body_mask]
    br_body = br[body_mask]
    if bx_body.size < 5:
        raise ValueError("Profile body points are insufficient after removing axis cap points.")

    L = float(bx_body[-1] - bx_body[0])
    R_max = float(br_body.max())
    print(
        f"[Profile] body_pts={len(bx_body)}  L={L:.4f}m  R_max={R_max:.4f}m  "
        f"nose_r={br_body[0]:.6f}m  tail_r={br_body[-1]:.6f}m"
    )
    return bx, br, bx_body, br_body, L, R_max


def build_layer_vector(first_cell: float, growth: float, n_layers: int) -> np.ndarray:
    if n_layers <= 0:
        return np.zeros(0, dtype=float)
    if abs(growth - 1.0) < 1.0e-10:
        return np.full(n_layers, first_cell, dtype=float)
    return first_cell * growth ** np.arange(n_layers, dtype=float)


def build_eta(
    first_cell: float,
    growth_inner: float,
    growth_outer: float,
    dense_band_thickness: float,
    r_far: float,
    ni_body: int,
    ni_outer: int,
) -> tuple[np.ndarray, dict]:
    inner_steps = build_layer_vector(first_cell, growth_inner, ni_body)
    eta = [0.0]
    accum = 0.0
    for dh in inner_steps:
        accum += dh
        eta.append(accum)

    while accum + EPS < dense_band_thickness:
        accum += max(inner_steps[-1] if inner_steps.size else first_cell, first_cell)
        eta.append(accum)

    inner_depth = accum
    outer_dh = max(inner_steps[-1] if inner_steps.size else first_cell, first_cell)
    for _ in range(ni_outer):
        outer_dh *= growth_outer
        accum += outer_dh
        eta.append(accum)
        if accum >= r_far:
            break

    eta = np.asarray(eta, dtype=float)
    eta = eta[eta < r_far]
    eta = np.append(eta, r_far)
    eta = np.unique(eta)
    stats = {
        "inner_depth": float(inner_depth),
        "n_total": int(len(eta)),
        "n_inner_like": int(np.sum(eta <= inner_depth + EPS)),
        "last_cell": float(eta[-1] - eta[-2]) if len(eta) > 1 else 0.0,
    }
    print(
        f"[Eta] Ni={len(eta)}  first={eta[1]:.3e}m  dense_depth={inner_depth:.4f}m  "
        f"last_cell={stats['last_cell']:.4f}m  r_far={eta[-1]:.3f}m"
    )
    return eta, stats


def resample_body(
    bx_body: np.ndarray,
    br_body: np.ndarray,
    nj_body: int,
    curvature_weight: float,
    slope_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    dx = np.gradient(bx_body)
    dr = np.gradient(br_body)
    ds = np.sqrt(dx * dx + dr * dr) + EPS
    s = np.zeros_like(bx_body)
    s[1:] = np.cumsum(np.sqrt(np.diff(bx_body) ** 2 + np.diff(br_body) ** 2))

    slope = np.gradient(br_body, bx_body, edge_order=1)
    d2r = np.gradient(slope, bx_body, edge_order=1)
    curvature = np.abs(d2r) / np.power(1.0 + slope * slope, 1.5)
    theta = np.unwrap(np.arctan2(dr, dx))
    slope_change = np.abs(np.gradient(theta, s, edge_order=1))

    curv_n = robust_norm(curvature)
    slope_n = robust_norm(slope_change)

    xi = (s - s[0]) / max(s[-1] - s[0], EPS)
    aft_boost = np.exp(-((xi - 1.0) / 0.07) ** 2)
    nose_boost = np.exp(-(xi / 0.07) ** 2)

    density = (
        1.0
        + curvature_weight * curv_n
        + slope_weight * slope_n
        + 1.5 * aft_boost
        + 0.8 * nose_boost
    )
    density = np.maximum(density, 1.0)

    seg_density = 0.5 * (density[:-1] + density[1:])
    s_eff = np.zeros_like(s)
    s_eff[1:] = np.cumsum(np.sqrt(np.diff(bx_body) ** 2 + np.diff(br_body) ** 2) * seg_density)

    target = np.linspace(0.0, s_eff[-1], nj_body)
    bx_b = np.interp(target, s_eff, bx_body)
    br_b = np.interp(target, s_eff, br_body)

    stats = {
        "curv_max": float(curvature.max()),
        "slope_change_max": float(slope_change.max()),
        "density_max": float(density.max()),
    }
    print(
        f"[Resample] nj_body={nj_body}  density_max={stats['density_max']:.2f}  "
        f"curv_max={stats['curv_max']:.3e}  slope_change_max={stats['slope_change_max']:.3e}"
    )
    return bx_b, br_b, stats


def geometric_axis_positions(x0: float, x1: float, n_cols: int, bias: float = 1.12) -> np.ndarray:
    if n_cols <= 1:
        return np.asarray([x0], dtype=float)
    if abs(bias - 1.0) < 1.0e-10:
        frac = np.linspace(0.0, 1.0, n_cols)
    else:
        weights = bias ** np.arange(n_cols - 1, dtype=float)
        frac = np.concatenate([[0.0], np.cumsum(weights) / np.sum(weights)])
    return x0 + (x1 - x0) * frac


def compute_body_normals(bx_b: np.ndarray, br_b: np.ndarray, interface_blend_cols: int) -> tuple[np.ndarray, np.ndarray]:
    tx = np.gradient(bx_b)
    tr = np.gradient(br_b)
    tx, tr = normalize_pair(tx, tr)
    nx, nr = -tr, tx

    if np.mean(nr) < 0.0:
        nx *= -1.0
        nr *= -1.0

    cap = max(2, min(interface_blend_cols, max(2, len(bx_b) // 20)))
    nx[0] = 0.0
    nr[0] = 1.0
    for j in range(1, cap):
        t = j / cap
        nx[j] = (1.0 - t) * 0.0 + t * nx[j]
        nr[j] = (1.0 - t) * 1.0 + t * nr[j]
        nxj, nrj = normalize_pair(np.asarray([nx[j]]), np.asarray([nr[j]]))
        nx[j] = nxj[0]
        nr[j] = nrj[0]

    nx[-1] = 0.0
    nr[-1] = 1.0
    for k in range(1, cap):
        j = len(bx_b) - 1 - k
        t = k / cap
        nx[j] = (1.0 - t) * 0.0 + t * nx[j]
        nr[j] = (1.0 - t) * 1.0 + t * nr[j]
        nxj, nrj = normalize_pair(np.asarray([nx[j]]), np.asarray([nr[j]]))
        nx[j] = nxj[0]
        nr[j] = nrj[0]

    # Limit axial leaning of normals. This preserves body-fitted behavior while
    # avoiding column crossing near sharp slope changes and under-resolved steps.
    nx = np.clip(nx, -0.45, 0.45)
    nx, nr = normalize_pair(nx, nr)

    return nx, nr


def assemble_block2_from_normals(
    bx_body: np.ndarray,
    br_body: np.ndarray,
    bx_b: np.ndarray,
    br_b: np.ndarray,
    eta: np.ndarray,
    nx: np.ndarray,
    nr: np.ndarray,
    smooth_iter: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    X = bx_b[None, :] + eta[:, None] * nx[None, :]
    R = br_b[None, :] + eta[:, None] * nr[None, :]
    R = np.maximum(R, 0.0)
    X[:, 0] = bx_b[0]
    X[:, -1] = bx_b[-1]

    if smooth_iter > 0:
        omega = 0.35
        for _ in range(smooth_iter):
            Xn = X.copy()
            Rn = R.copy()
            Xn[1:-1, 1:-1] = (
                omega
                * 0.25
                * (X[:-2, 1:-1] + X[2:, 1:-1] + X[1:-1, :-2] + X[1:-1, 2:])
                + (1.0 - omega) * X[1:-1, 1:-1]
            )
            Rn[1:-1, 1:-1] = (
                omega
                * 0.25
                * (R[:-2, 1:-1] + R[2:, 1:-1] + R[1:-1, :-2] + R[1:-1, 2:])
                + (1.0 - omega) * R[1:-1, 1:-1]
            )
            Xn[0, :] = bx_b
            Rn[0, :] = br_b
            Xn[-1, :] = X[-1, :]
            Rn[-1, :] = R[-1, :]
            Xn[:, 0] = bx_b[0]
            Xn[:, -1] = bx_b[-1]
            Rn[:, 0] = R[:, 0]
            Rn[:, -1] = R[:, -1]
            X, R = Xn, np.maximum(Rn, 0.0)

    for i in range(1, len(eta)):
        body_r_at_x = np.interp(X[i, :], bx_body, br_body, left=0.0, right=0.0)
        in_body_x = (X[i, :] >= bx_body[0]) & (X[i, :] <= bx_body[-1])
        min_clearance = 0.20 * eta[i]
        R[i, in_body_x] = np.maximum(R[i, in_body_x], body_r_at_x[in_body_x] + min_clearance)

    metrics = cell_metrics(X, R)
    intrusion = body_intrusion_count([(X, R)], bx_body, br_body)
    return X, R, metrics["negative_cells"], intrusion


def build_block2(
    bx_body: np.ndarray,
    br_body: np.ndarray,
    eta: np.ndarray,
    nj_body: int,
    curvature_weight: float,
    slope_weight: float,
    smooth_iter: int,
    interface_blend_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    bx_b, br_b, resample_stats = resample_body(
        bx_body, br_body, nj_body, curvature_weight, slope_weight
    )
    nx_base, nr_base = compute_body_normals(bx_b, br_b, interface_blend_cols)

    if smooth_iter > 0:
        print(f"[Block2] smoothing interior ({smooth_iter} iter)")

    best = None
    best_score = None
    for relax in np.linspace(1.0, 0.0, 9):
        nx_try = relax * nx_base
        nr_try = (1.0 - relax) * 1.0 + relax * nr_base
        nx_try, nr_try = normalize_pair(nx_try, nr_try)
        X_try, R_try, neg_try, intrusion_try = assemble_block2_from_normals(
            bx_body, br_body, bx_b, br_b, eta, nx_try, nr_try, smooth_iter
        )
        score = (neg_try > 0, intrusion_try > 0, neg_try, intrusion_try, -relax)
        if best is None or score < best_score:
            best = (X_try, R_try, relax, neg_try, intrusion_try)
            best_score = score
        if neg_try == 0 and intrusion_try == 0:
            break

    X, R, chosen_relax, chosen_neg, chosen_intrusion = best
    print(
        f"[Block2] normal_relax={chosen_relax:.2f}  neg={chosen_neg}  "
        f"intrusion={chosen_intrusion}"
    )

    stats = {
        "body_tail_r": float(br_b[-1]),
        "nose_x": float(bx_b[0]),
        "tail_x": float(bx_b[-1]),
        **resample_stats,
    }
    print(
        f"[Block2] nj={X.shape[1]}  ni={X.shape[0]}  nose_x={bx_b[0]:.4f}m  "
        f"tail_x={bx_b[-1]:.4f}m  tail_r={br_b[-1]:.4f}m"
    )
    return X, R, bx_b, br_b, stats


def build_block1(X2: np.ndarray, R2: np.ndarray, x_up: float, dx0: float) -> tuple[np.ndarray, np.ndarray]:
    x_nose = float(X2[0, 0])
    r_nose = R2[:, 0].copy()
    x_vals = [x_nose]
    dxc = max(dx0, EPS)
    while x_vals[-1] > x_up:
        x_vals.append(max(x_up, x_vals[-1] - dxc))
        dxc = min(dxc * 1.10, max((x_nose - x_up) * 0.25, dxc))
        if abs(x_vals[-1] - x_up) < EPS:
            break
    x1_1d = np.sort(np.unique(np.asarray(x_vals, dtype=float)))
    X1 = np.repeat(x1_1d[None, :], len(r_nose), axis=0)
    R1 = np.repeat(r_nose[:, None], len(x1_1d), axis=1)
    print(
        f"[Block1] nj={X1.shape[1]}  dx_min={np.diff(x1_1d).min():.5f}m  "
        f"dx_max={np.diff(x1_1d).max():.3f}m"
    )
    return X1, R1


def build_base_nodes(tail_r: float, first_cell: float, growth_inner: float) -> np.ndarray:
    nodes = [0.0]
    dh = max(first_cell, tail_r / 200.0, 1.0e-6)
    while nodes[-1] + dh < tail_r - EPS:
        nodes.append(nodes[-1] + dh)
        dh *= max(growth_inner, 1.02)
    if tail_r - nodes[-1] > EPS:
        nodes.append(tail_r)
    return np.asarray(nodes, dtype=float)


def build_block4_wake(
    X2: np.ndarray,
    R2: np.ndarray,
    wake_length: float,
    wake_nx: int,
    first_cell: float,
    growth_inner: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_tail = float(X2[0, -1])
    tail_col = R2[:, -1].copy()
    tail_r = float(tail_col[0])
    base_nodes = build_base_nodes(tail_r, first_cell, growth_inner)

    left_r = np.concatenate([base_nodes[:-1], tail_col])
    x_right = x_tail + wake_length
    right_r = left_r.copy()

    x_line = geometric_axis_positions(x_tail, x_right, wake_nx, bias=1.14)
    X4 = np.repeat(x_line[None, :], len(left_r), axis=0)
    R4 = np.repeat(left_r[:, None], len(x_line), axis=1)
    print(
        f"[Block4] wake/base connector  ni={X4.shape[0]}  nj={X4.shape[1]}  "
        f"wake_length={wake_length:.4f}m  base_rows={len(base_nodes)}"
    )
    return X4, R4, right_r


def build_block3(
    r_start: np.ndarray,
    x_start: float,
    x_dn: float,
    dx0: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_vals = [x_start]
    dxc = max(dx0, EPS)
    while x_vals[-1] < x_dn:
        x_vals.append(min(x_dn, x_vals[-1] + dxc))
        dxc = min(dxc * 1.08, max((x_dn - x_start) * 0.25, dxc))
        if abs(x_vals[-1] - x_dn) < EPS:
            break
    x3_1d = np.asarray(x_vals, dtype=float)
    X3 = np.repeat(x3_1d[None, :], len(r_start), axis=0)
    R3 = np.repeat(r_start[:, None], len(x3_1d), axis=1)
    print(
        f"[Block3] nj={X3.shape[1]}  dx_min={np.diff(x3_1d).min():.5f}m  "
        f"dx_max={np.diff(x3_1d).max():.3f}m"
    )
    return X3, R3


def cell_metrics(X: np.ndarray, R: np.ndarray) -> dict:
    dxi = X[1:, :-1] - X[:-1, :-1]
    dri = R[1:, :-1] - R[:-1, :-1]
    dxj = X[:-1, 1:] - X[:-1, :-1]
    drj = R[:-1, 1:] - R[:-1, :-1]
    signed_area = dxi * drj - dri * dxj
    if signed_area.size:
        orientation = np.sign(np.nanmedian(signed_area))
        if orientation == 0.0:
            orientation = 1.0
        signed_area = signed_area * orientation
    area = np.abs(signed_area)
    li = np.sqrt(dxi * dxi + dri * dri + EPS)
    lj = np.sqrt(dxj * dxj + drj * drj + EPS)
    aspect = np.maximum(li, lj) / np.maximum(np.minimum(li, lj), EPS)
    finite_aspect = aspect[np.isfinite(aspect)]
    return {
        "signed_area": signed_area,
        "area": area,
        "aspect": aspect,
        "negative_cells": int(np.sum(signed_area <= 0.0)),
        "cells": int((X.shape[0] - 1) * (X.shape[1] - 1)),
        "area_min": float(area.min()) if area.size else 0.0,
        "area_max": float(area.max()) if area.size else 0.0,
        "p95_aspect": float(np.percentile(finite_aspect, 95.0)) if finite_aspect.size else 0.0,
        "p99_aspect": float(np.percentile(finite_aspect, 99.0)) if finite_aspect.size else 0.0,
    }


def body_intrusion_count(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    bx_body: np.ndarray,
    br_body: np.ndarray,
) -> int:
    count = 0
    for X, R in blocks:
        sample_x = X[1:, :]
        sample_r = R[1:, :]
        on_body_x = (sample_x >= bx_body[0]) & (sample_x <= bx_body[-1])
        body_r = np.interp(sample_x, bx_body, br_body, left=0.0, right=0.0)
        count += int(np.sum(on_body_x & (sample_r < body_r - 1.0e-6)))
    return count


def trailing_edge_connectivity_check(
    X2: np.ndarray,
    R2: np.ndarray,
    X4: np.ndarray,
    R4: np.ndarray,
    X3: np.ndarray,
    R3: np.ndarray,
) -> dict:
    offset = X4.shape[0] - X2.shape[0]
    err24 = max(
        float(np.max(np.abs(X2[:, -1] - X4[offset:, 0]))),
        float(np.max(np.abs(R2[:, -1] - R4[offset:, 0]))),
    )
    err43 = max(
        float(np.max(np.abs(X4[:, -1] - X3[:, 0]))),
        float(np.max(np.abs(R4[:, -1] - R3[:, 0]))),
    )
    return {
        "block2_to_block4": err24,
        "block4_to_block3": err43,
        "ok": bool(err24 < 1.0e-10 and err43 < 1.0e-10),
    }


def summarize_quality(named_blocks: list[tuple[str, np.ndarray, np.ndarray]]) -> tuple[dict, list[dict]]:
    block_stats = []
    all_area = []
    all_aspect = []
    neg_total = 0
    cell_total = 0
    for name, X, R in named_blocks:
        stats = cell_metrics(X, R)
        stats["name"] = name
        block_stats.append(stats)
        all_area.append(stats["area"].ravel())
        all_aspect.append(stats["aspect"].ravel())
        neg_total += stats["negative_cells"]
        cell_total += stats["cells"]
        print(
            f"  {name}: cells={stats['cells']:,}  neg={stats['negative_cells']}  "
            f"area=[{stats['area_min']:.3e}, {stats['area_max']:.3e}]  "
            f"AR95={stats['p95_aspect']:.1f}  AR99={stats['p99_aspect']:.1f}"
        )

    all_area = np.concatenate(all_area) if all_area else np.zeros(1)
    all_aspect = np.concatenate(all_aspect) if all_aspect else np.zeros(1)
    finite_aspect = all_aspect[np.isfinite(all_aspect)]
    total = {
        "cells": cell_total,
        "negative_cells": neg_total,
        "area_min": float(all_area.min()) if all_area.size else 0.0,
        "area_max": float(all_area.max()) if all_area.size else 0.0,
        "p95_aspect": float(np.percentile(finite_aspect, 95.0)) if finite_aspect.size else 0.0,
        "p99_aspect": float(np.percentile(finite_aspect, 99.0)) if finite_aspect.size else 0.0,
    }
    return total, block_stats


def plot_block(ax, X: np.ndarray, R: np.ndarray, si: int, sj: int, mirror: bool = True) -> None:
    for i in range(0, X.shape[0], max(1, si)):
        ax.plot(X[i, :], R[i, :], color="royalblue", lw=0.35, alpha=0.55)
        if mirror:
            ax.plot(X[i, :], -R[i, :], color="royalblue", lw=0.35, alpha=0.35)
    for j in range(0, X.shape[1], max(1, sj)):
        ax.plot(X[:, j], R[:, j], color="royalblue", lw=0.35, alpha=0.55)
        if mirror:
            ax.plot(X[:, j], -R[:, j], color="royalblue", lw=0.35, alpha=0.35)


def plot_mesh(
    X1: np.ndarray,
    R1: np.ndarray,
    X2: np.ndarray,
    R2: np.ndarray,
    X3: np.ndarray,
    R3: np.ndarray,
    X4: np.ndarray,
    R4: np.ndarray,
    bx_all: np.ndarray,
    br_all: np.ndarray,
    bx_body: np.ndarray,
    br_body: np.ndarray,
    eta_stats: dict,
    quality_total: dict,
    save_path: str,
) -> None:
    fig = plt.figure(figsize=(22, 17))
    gs = fig.add_gridspec(3, 2, hspace=0.28, wspace=0.22)

    blocks = [(X1, R1), (X2, R2), (X4, R4), (X3, R3)]
    x_min = min(b[0].min() for b in blocks)
    x_max = max(b[0].max() for b in blocks)
    r_max = max(b[1].max() for b in blocks)
    L = bx_body[-1] - bx_body[0]
    tail_x = bx_body[-1]
    tail_r = br_body[-1]

    def decorate(ax, title: str, xlim: tuple[float, float], ylim: tuple[float, float], mirror: bool) -> None:
        for X, R in blocks:
            plot_block(ax, X, R, si=max(1, X.shape[0] // 25), sj=max(1, X.shape[1] // 30), mirror=mirror)
        if mirror:
            ax.fill_between(bx_all, br_all, -br_all, color="0.45", alpha=0.85)
            ax.plot(bx_all, br_all, color="red", lw=1.2)
            ax.plot(bx_all, -br_all, color="red", lw=1.2)
        else:
            ax.fill_between(bx_all, 0.0, br_all, color="0.45", alpha=0.85)
            ax.plot(bx_all, br_all, color="red", lw=1.5)
        ax.axhline(0.0, color="k", lw=0.7, ls="--", alpha=0.35)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("r [m]")
        ax.grid(True, alpha=0.15)

    ax1 = fig.add_subplot(gs[0, :])
    decorate(
        ax1,
        f"Full domain  cells={quality_total['cells']:,}  neg={quality_total['negative_cells']}  AR99={quality_total['p99_aspect']:.1f}",
        (x_min, x_max),
        (-0.55 * r_max, 0.55 * r_max),
        True,
    )
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(gs[1, 0])
    decorate(
        ax2,
        "Body zoom",
        (bx_body[0] - 0.25 * L, bx_body[-1] + 0.30 * L),
        (-0.02 * br_body.max(), 2.8 * br_body.max()),
        False,
    )

    nose_dx = max(0.08 * L, 0.5)
    ax3 = fig.add_subplot(gs[1, 1])
    decorate(
        ax3,
        "Nose zoom",
        (bx_body[0] - 0.15 * nose_dx, bx_body[0] + nose_dx),
        (0.0, min(r_max, br_body.max() * 2.2)),
        False,
    )

    aft_dx = max(0.12 * L, 0.8)
    ax4 = fig.add_subplot(gs[2, 0])
    decorate(
        ax4,
        "Aft / trailing-edge zoom",
        (tail_x - 0.35 * aft_dx, tail_x + 1.05 * max(X4[0, -1] - tail_x, aft_dx)),
        (0.0, min(r_max, max(tail_r * 2.3, br_body.max() * 1.8))),
        False,
    )

    ax5 = fig.add_subplot(gs[2, 1])
    x_mid = bx_body[0] + 0.55 * L
    dense_top = float(np.interp(x_mid, bx_body, br_body)) + eta_stats["inner_depth"] * 1.10
    decorate(
        ax5,
        "Inflation / dense band detail",
        (x_mid - 0.05 * L, x_mid + 0.05 * L),
        (float(np.interp(x_mid, bx_body, br_body)) - 0.01 * br_body.max(), dense_top),
        False,
    )

    fig.suptitle(
        "Structured rocket mesh: weighted resampling + deep dense band + wake connector",
        fontsize=14,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {save_path}")


def save_mesh(
    path: str,
    X1: np.ndarray,
    R1: np.ndarray,
    X2: np.ndarray,
    R2: np.ndarray,
    X3: np.ndarray,
    R3: np.ndarray,
    X4: np.ndarray,
    R4: np.ndarray,
    bx_body: np.ndarray,
    br_body: np.ndarray,
    eta: np.ndarray,
    params: dict,
    quality_total: dict,
    connectivity: dict,
) -> None:
    numeric_params = {f"p_{k}": np.array(v) for k, v in params.items() if isinstance(v, (int, float, bool))}
    np.savez_compressed(
        path,
        X1=X1,
        R1=R1,
        X2=X2,
        R2=R2,
        X3=X3,
        R3=R3,
        X4=X4,
        R4=R4,
        body_x=bx_body,
        body_r=br_body,
        eta=eta,
        q_negative_cells=np.array(quality_total["negative_cells"]),
        q_area_min=np.array(quality_total["area_min"]),
        q_area_max=np.array(quality_total["area_max"]),
        q_p95_aspect=np.array(quality_total["p95_aspect"]),
        q_p99_aspect=np.array(quality_total["p99_aspect"]),
        q_trailing_edge_ok=np.array(int(connectivity["ok"])),
        **numeric_params,
    )
    print(f"[Save] {path}.npz")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structured axisymmetric rocket mesh generator")
    p.add_argument("--profile", required=True, help="Rocket profile CSV with columns x_m, r_m")
    p.add_argument("--nj_body", type=int, default=1600, help="Axial body points after weighted resampling")
    p.add_argument("--ni_body", type=int, default=140, help="Inner dense-band radial layers")
    p.add_argument("--ni_outer", type=int, default=90, help="Outer radial layers after dense band")
    p.add_argument("--first_cell", type=float, default=2.0e-5, help="First normal cell height [m]")
    p.add_argument("--growth_inner", type=float, default=1.05, help="Inner dense-band growth ratio")
    p.add_argument("--growth_outer", type=float, default=1.10, help="Outer growth ratio")
    p.add_argument("--dense_band_thickness", type=float, default=0.20, help="Minimum near-body dense envelope thickness [m]")
    p.add_argument("--wake_length", type=float, default=1.50, help="Structured wake/base connector length [m]")
    p.add_argument("--wake_nx", type=int, default=110, help="Axial columns in wake/base connector")
    p.add_argument("--curvature_weight", type=float, default=5.0, help="Curvature weight for body resampling")
    p.add_argument("--slope_weight", type=float, default=3.0, help="Slope-change weight for body resampling")
    p.add_argument("--r_far", type=float, default=None, help="Far-field radius [m], default 20*R_max")
    p.add_argument("--x_up_mult", type=float, default=5.0, help="Upstream extent multiplier")
    p.add_argument("--x_dn_mult", type=float, default=8.0, help="Downstream extent multiplier from wake end")
    p.add_argument("--supersonic", action="store_true", help="Set upstream extent multiplier to 10")
    p.add_argument("--smooth_iter", type=int, default=180, help="Block2 interior smoothing iterations")
    p.add_argument("--interface_blend_cols", type=int, default=10, help="Columns blended to vertical at the nose interface")
    p.add_argument("--output", default=None, help="Output prefix")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    base = os.path.splitext(os.path.basename(args.profile))[0].replace("_profile", "")
    outpfx = args.output or f"{base}_mesh"

    print("=" * 72)
    print(" Structured axisymmetric rocket mesh generator")
    print("=" * 72)
    print(f" profile={args.profile}")
    print(
        f" nj_body={args.nj_body}  ni_body={args.ni_body}  ni_outer={args.ni_outer}  wake_nx={args.wake_nx}"
    )
    print(
        f" first_cell={args.first_cell:.3e}m  growth_inner={args.growth_inner:.3f}  growth_outer={args.growth_outer:.3f}  dense_band={args.dense_band_thickness:.4f}m"
    )
    print(
        f" wake_length={args.wake_length:.4f}m  curvature_weight={args.curvature_weight:.2f}  slope_weight={args.slope_weight:.2f}"
    )
    print("=" * 72)

    bx_all, br_all, bx_body, br_body, L, R_max = load_profile(args.profile)
    x_up_mult = 10.0 if args.supersonic else args.x_up_mult
    r_far = float(args.r_far if args.r_far is not None else 20.0 * R_max)
    x_up = bx_body[0] - x_up_mult * L

    eta, eta_stats = build_eta(
        args.first_cell,
        args.growth_inner,
        args.growth_outer,
        args.dense_band_thickness,
        r_far,
        args.ni_body,
        args.ni_outer,
    )
    print(
        f"[Domain] x_up={x_up:.3f}m  tentative_wake_end={bx_body[-1] + args.wake_length:.3f}m  r_far={r_far:.3f}m"
    )

    X2, R2, bx_b, br_b, block2_stats = build_block2(
        bx_body,
        br_body,
        eta,
        args.nj_body,
        args.curvature_weight,
        args.slope_weight,
        args.smooth_iter,
        args.interface_blend_cols,
    )

    dx0 = max(float(np.min(np.diff(bx_b))), args.first_cell)
    X1, R1 = build_block1(X2, R2, x_up, dx0)
    X4, R4, r_start_block3 = build_block4_wake(
        X2, R2, args.wake_length, args.wake_nx, args.first_cell, args.growth_inner
    )

    x_dn = X4[0, -1] + args.x_dn_mult * L
    X3, R3 = build_block3(r_start_block3, float(X4[0, -1]), x_dn, dx0)

    print("[Quality]")
    named_blocks = [
        ("Block1 upstream", X1, R1),
        ("Block2 body-fitted", X2, R2),
        ("Block4 wake/base", X4, R4),
        ("Block3 downstream", X3, R3),
    ]
    quality_total, _ = summarize_quality(named_blocks)
    intrusion = body_intrusion_count([(X2, R2)], bx_body, br_body)
    connectivity = trailing_edge_connectivity_check(X2, R2, X4, R4, X3, R3)

    print(
        f"  TOTAL: cells={quality_total['cells']:,}  neg={quality_total['negative_cells']}  area=[{quality_total['area_min']:.3e}, {quality_total['area_max']:.3e}]  AR95={quality_total['p95_aspect']:.1f}  AR99={quality_total['p99_aspect']:.1f}"
    )
    print(f"  body_intrusion_count={intrusion}")
    print(
        f"  trailing_edge_connectivity_check={connectivity['ok']}  (B2->B4={connectivity['block2_to_block4']:.3e}, B4->B3={connectivity['block4_to_block3']:.3e})"
    )

    params = {
        "nj_body": args.nj_body,
        "ni_body": args.ni_body,
        "ni_outer": args.ni_outer,
        "first_cell": args.first_cell,
        "growth_inner": args.growth_inner,
        "growth_outer": args.growth_outer,
        "dense_band_thickness": args.dense_band_thickness,
        "wake_length": args.wake_length,
        "wake_nx": args.wake_nx,
        "curvature_weight": args.curvature_weight,
        "slope_weight": args.slope_weight,
        "smooth_iter": args.smooth_iter,
        "r_far": r_far,
        "x_up": x_up,
        "x_dn": x_dn,
        "profile_length": L,
        "profile_r_max": R_max,
        "dense_depth_realized": eta_stats["inner_depth"],
        "tail_r": block2_stats["body_tail_r"],
        "negative_cells": quality_total["negative_cells"],
        "body_intrusion_count": intrusion,
        "te_connectivity_ok": int(connectivity["ok"]),
    }
    save_mesh(
        outpfx,
        X1,
        R1,
        X2,
        R2,
        X3,
        R3,
        X4,
        R4,
        bx_body,
        br_body,
        eta,
        params,
        quality_total,
        connectivity,
    )
    plot_mesh(
        X1,
        R1,
        X2,
        R2,
        X3,
        R3,
        X4,
        R4,
        bx_all,
        br_all,
        bx_body,
        br_body,
        eta_stats,
        quality_total,
        outpfx + "_mesh.png",
    )

    print("[Done]")
    print(f"  mesh_npz={outpfx}.npz")
    print(f"  mesh_png={outpfx}_mesh.png")


if __name__ == "__main__":
    main()
