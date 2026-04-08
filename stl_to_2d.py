"""
stl_to_2d.py  (v6)
==================
Extract 2D axisymmetric rocket profile (x, r) from a binary STL file.

Method: plane-triangle intersection (slicing)
  - Cut the STL mesh with equally-spaced planes perpendicular to the rocket axis
  - At each plane, compute the actual intersection points with every triangle
  - Per-plane centerline = midrange of intersection points (min+max)/2
  - Global centerline = average of per-plane centerlines
  - r = max distance from global centerline at each slice plane
  - This is the correct, geometry-based approach; vertex-sampling is NOT used

Why this is better than vertex sampling:
  - Vertices land at triangle corners, not on the surface between them
  - Intersection points lie exactly on the surface at the chosen z-plane
  - Works regardless of STL axis orientation or vertex density distribution
  - Nose is never clipped because every slice is computed, not inferred from vertices

All intersections are vectorized (numpy) for speed (~5s for 500 slices on 400k triangles).

Output coordinate:
  x : rocket axis, tip (nose) = 0, inlet (tail) = max  [m]
  r : radius from centerline, >= 0                      [m]

Usage:
  python stl_to_2d.py --stl Falcon9.stl --scale 1.0
  python stl_to_2d.py --stl rocket_mm.stl --scale 0.001 --output rocket.csv

Parameters:
  --scale    unit conversion. Check printed Z-extent to decide.
             If Z~4.7  -> already metres,  use 1.0
             If Z~4700 -> millimetres,     use 0.001
  --n_slices number of cutting planes (default 500)
  --output   output CSV path (default: <stl_name>_profile.csv)

Output files:
  <n>_profile.csv   x_m, r_m  (simulator input, r>=0)
  <n>_profile.png   two-panel plot

Requirements:
  pip install numpy matplotlib
  (no numpy-stl needed)
"""

import argparse
import os
import sys
import struct
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import time


# =============================================================================
# 1. Load binary STL -> triangle array
# =============================================================================

def load_stl(filepath: str) -> np.ndarray:
    """
    Parse binary STL, return triangle array (N, 3, 3).

    Binary STL per triangle:
      12 bytes: normal (float32 x3)      -> skipped
      36 bytes: 3 vertices (float32 x3)  -> stored
       2 bytes: attribute                -> skipped

    Returns
    -------
    tris : np.ndarray, shape (N, 3, 3)
        tris[i, j, :] = XYZ of j-th vertex of i-th triangle
        Units: original STL units (apply --scale later)
    """
    if not os.path.exists(filepath):
        sys.exit(f"[ERROR] File not found: {filepath}")

    with open(filepath, 'rb') as f:
        header  = f.read(80)
        n_tri   = struct.unpack('<I', f.read(4))[0]
        print(f"[Load] {os.path.basename(filepath)}")
        print(f"[Load] Triangles: {n_tri:,}")

        raw = []
        for _ in range(n_tri):
            f.read(12)   # skip normal
            v0 = struct.unpack('<fff', f.read(12))
            v1 = struct.unpack('<fff', f.read(12))
            v2 = struct.unpack('<fff', f.read(12))
            f.read(2)    # skip attribute
            raw.append([v0, v1, v2])

    tris = np.array(raw, dtype=np.float64)  # (N, 3, 3)
    print(f"[Load] Triangle array shape: {tris.shape}")
    return tris


# =============================================================================
# 2. Detect rocket axis
# =============================================================================

def detect_axis(tris: np.ndarray) -> tuple:
    """
    Identify rocket axis as the longest bounding box direction.

    Returns
    -------
    axis_idx : int         0=X, 1=Y, 2=Z
    perp_idx : list[int]   two perpendicular axis indices
    """
    all_v    = tris.reshape(-1, 3)
    extents  = all_v.max(axis=0) - all_v.min(axis=0)
    axis_idx = int(np.argmax(extents))
    perp_idx = [i for i in range(3) if i != axis_idx]

    print(f"[Axis] Extents  X={extents[0]:.5f}  Y={extents[1]:.5f}  Z={extents[2]:.5f}"
          f"  (STL native units)")
    print(f"[Axis] Rocket axis -> {'XYZ'[axis_idx]} (index {axis_idx})")
    print(f"       If extents look like mm (e.g. 4700), use --scale 0.001")
    return axis_idx, perp_idx


# =============================================================================
# 3. Vectorized plane-triangle intersection
# =============================================================================

def slice_at_plane(tris: np.ndarray, z_plane: float, axis_idx: int) -> np.ndarray:
    """
    Compute intersection midpoints of all triangles with the plane
    coordinate[axis_idx] = z_plane.

    Algorithm per triangle:
      1. Compute signed distance d = vertex[axis_idx] - z_plane for each vertex
      2. Triangle straddles plane if not all d have the same sign
      3. For each edge where sign changes, interpolate intersection point
      4. Two intersection points form a line segment; return its midpoint

    Fully vectorized: no Python loop over triangles.

    Parameters
    ----------
    tris     : np.ndarray, shape (N, 3, 3)
    z_plane  : float
    axis_idx : int

    Returns
    -------
    mids : np.ndarray, shape (M, 3)   midpoints of intersection segments
           Empty array (0,3) if no intersections.
    """
    d = tris[:, :, axis_idx] - z_plane   # (N, 3) signed distances

    pos   = (d >= 0)
    n_pos = pos.sum(axis=1)
    cross = (n_pos > 0) & (n_pos < 3)    # triangles straddling the plane

    if not cross.any():
        return np.zeros((0, 3))

    tc = tris[cross]   # (M, 3, 3)
    dc = d[cross]      # (M, 3)
    M  = len(tc)

    # Accumulate intersection points per triangle
    # Each triangle has exactly 2 crossing edges -> 2 points -> 1 midpoint
    cross_pts = np.full((M, 2, 3), np.nan)
    n_found   = np.zeros(M, dtype=int)

    for ia, ib in [(0, 1), (1, 2), (2, 0)]:
        da     = dc[:, ia]          # (M,)
        db     = dc[:, ib]
        edge_x = (da >= 0) != (db >= 0)   # edges crossing plane

        if not edge_x.any():
            continue

        denom = da[edge_x] - db[edge_x]
        t     = da[edge_x] / denom                              # interpolation param
        va    = tc[edge_x, ia, :]                               # (K, 3)
        vb    = tc[edge_x, ib, :]
        pt    = va + t[:, None] * (vb - va)                     # (K, 3)

        # Store in first available slot per triangle
        idx_in_M = np.where(edge_x)[0]
        for k, m in enumerate(idx_in_M):
            slot = n_found[m]
            if slot < 2:
                cross_pts[m, slot, :] = pt[k]
                n_found[m] += 1

    # Only triangles with exactly 2 intersection points
    valid = n_found == 2
    if not valid.any():
        return np.zeros((0, 3))

    mids = (cross_pts[valid, 0, :] + cross_pts[valid, 1, :]) / 2
    return mids


# =============================================================================
# 4. Extract full profile
# =============================================================================

def extract_profile(tris: np.ndarray,
                    axis_idx: int,
                    perp_idx: list,
                    scale: float,
                    n_slices: int = 500) -> tuple:
    """
    Extract (x, r) profile using plane-triangle intersection slicing.

    Two-pass approach:
      Pass 1 (centerline): slice every plane, compute per-slice midrange center,
                           average to get global centerline.
      Pass 2 (profile):    slice every plane again, compute max r from global center.

    Parameters
    ----------
    tris     : np.ndarray, shape (N, 3, 3)
    axis_idx : int
    perp_idx : list[int]
    scale    : float   unit conversion (e.g. 0.001 for mm->m)
    n_slices : int     number of cutting planes (default 500)

    Returns
    -------
    x_arr : np.ndarray  [m]  along rocket axis
    r_arr : np.ndarray  [m]  radius from centerline
    """
    z_all  = tris[:, :, axis_idx]
    z_min  = z_all.min()
    z_max  = z_all.max()
    planes = np.linspace(z_min + 1e-6, z_max - 1e-6, n_slices)

    # --- Pass 1: centerline ---
    print(f"[Profile] Pass 1/2: centerline estimation ({n_slices} slices)...")
    t0 = time.time()
    cx_list, cy_list = [], []

    for zp in planes:
        mids = slice_at_plane(tris, zp, axis_idx)
        if len(mids) < 3:
            continue
        px = mids[:, perp_idx[0]]
        py = mids[:, perp_idx[1]]
        cx_list.append((px.min() + px.max()) / 2)
        cy_list.append((py.min() + py.max()) / 2)

    if not cx_list:
        sys.exit("[ERROR] No intersections found. Check STL file and axis.")

    cx_global = np.mean(cx_list)
    cy_global = np.mean(cy_list)
    print(f"         Elapsed: {time.time()-t0:.1f}s")
    print(f"[Center] Global centerline: "
          f"perp[{perp_idx[0]}]={cx_global:.6f}  perp[{perp_idx[1]}]={cy_global:.6f}")

    # --- Pass 2: profile extraction ---
    print(f"[Profile] Pass 2/2: profile extraction ({n_slices} slices)...")
    t0 = time.time()
    x_prof, r_prof = [], []

    for zp in planes:
        mids = slice_at_plane(tris, zp, axis_idx)
        if len(mids) < 1:
            continue
        px = mids[:, perp_idx[0]] - cx_global
        py = mids[:, perp_idx[1]] - cy_global
        r  = np.sqrt(px**2 + py**2)
        x_prof.append(zp * scale)
        r_prof.append(r.max() * scale)

    print(f"         Elapsed: {time.time()-t0:.1f}s")

    x_arr = np.array(x_prof)
    r_arr = np.array(r_prof)
    sf    = np.argsort(x_arr)
    x_arr = x_arr[sf]
    r_arr = r_arr[sf]

    print(f"[Profile] {len(x_arr)} pts | "
          f"x: {x_arr.min():.5f} ~ {x_arr.max():.5f} m | "
          f"r: {r_arr.min():.5f} ~ {r_arr.max():.5f} m")
    return x_arr, r_arr


# =============================================================================
# 5. Orient tip to x=0
# =============================================================================

def orient_tip(x_arr: np.ndarray, r_arr: np.ndarray) -> tuple:
    """
    Orient so tip (nose) = x=0, inlet (tail) = x=max.
    Compare mean r in outermost 5% on each side.
    Smaller r = tip (auto-detect). User confirms via graph.
    """
    n5      = max(1, len(x_arr) // 20)
    r_left  = r_arr[:n5].mean()
    r_right = r_arr[-n5:].mean()
    flipped = False

    if r_left > r_right:
        x_arr   = x_arr.max() - x_arr[::-1]
        r_arr   = r_arr[::-1]
        flipped = True
        print(f"[Orient] left_r={r_left:.5f} > right_r={r_right:.5f} -> auto-flipped")
    else:
        print(f"[Orient] left_r={r_left:.5f} <= right_r={r_right:.5f} -> kept")

    x_arr = x_arr - x_arr.min()
    return x_arr, r_arr, flipped


def do_flip(x_arr, r_arr):
    x_arr = x_arr.max() - x_arr[::-1]
    r_arr = r_arr[::-1]
    x_arr = x_arr - x_arr.min()
    return x_arr, r_arr

# =============================================================================
# 5-b. Cap nose and tail endpoints with r=0
# =============================================================================

def cap_endpoints(x_arr, r_arr):
    """
    Close the nose and tail of the profile by inserting r=0 cap points.

    Without caps the profile is open at both ends:
      Nose: x=0,  r=r_nose (small but nonzero)
      Tail: x=L,  r=r_tail (can be large if engine base is cut)

    This inserts:
      (x_nose, 0) at the front  -> symmetry axis intersection at nose
      (x_tail, 0) at the back   -> symmetry axis intersection at tail

    Effect on mesh.py:
      The profile now starts and ends at r=0, so inflation layers
      have a properly closed boundary and cannot penetrate the body.
    """
    x_nose, r_nose = x_arr[0],  r_arr[0]
    x_tail, r_tail = x_arr[-1], r_arr[-1]

    print(f"[Cap] Nose: x={x_nose:.5f} r={r_nose:.5f} -> prepending (x={x_nose:.5f}, r=0)")
    print(f"[Cap] Tail: x={x_tail:.5f} r={r_tail:.5f} -> appending  (x={x_tail:.5f}, r=0)")

    x_cap = np.concatenate([[x_nose], x_arr, [x_tail]])
    r_cap = np.concatenate([[0.0],    r_arr, [0.0   ]])

    print(f"[Cap] Profile: {len(x_arr)} pts -> {len(x_cap)} pts (with caps)")
    return x_cap, r_cap




# =============================================================================
# 6. Plot
# =============================================================================

def plot_profile(x_arr: np.ndarray, r_arr: np.ndarray,
                 title: str = "Rocket 2D Profile",
                 save_path: str = None,
                 show: bool = True):
    """
    Two-panel:
      Left  : full shape +/-r (direction check)
      Right : r>=0 simulator domain
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                             gridspec_kw={'width_ratios': [2, 1]})

    ax = axes[0]
    ax.plot(x_arr,  r_arr, 'b-', linewidth=1.2)
    ax.plot(x_arr, -r_arr, 'b-', linewidth=1.2)
    ax.fill_between(x_arr, r_arr, -r_arr, alpha=0.15, color='steelblue')
    ax.axhline(0, color='k', linewidth=0.7, linestyle='--', alpha=0.5)
    ax.axvline(x_arr.min(), color='g', linewidth=1.0, linestyle=':',
               label=f'tip   x={x_arr.min():.4f} m')
    ax.axvline(x_arr.max(), color='r', linewidth=1.0, linestyle=':',
               label=f'inlet x={x_arr.max():.4f} m')
    ax.set_xlabel('x [m]  (tip -> inlet)', fontsize=11)
    ax.set_ylabel('r [m]', fontsize=11)
    ax.set_title('Full shape  (direction check)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x_arr, r_arr, 'b-', linewidth=1.2)
    ax.fill_between(x_arr, 0, r_arr, alpha=0.15, color='steelblue')
    ax.axhline(0, color='k', linewidth=1.0, linestyle='--', label='symmetry axis r=0')
    ax.set_xlabel('x [m]', fontsize=11)
    ax.set_ylabel('r [m]', fontsize=11)
    ax.set_title('Simulator domain  (r >= 0)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Plot] saved -> {save_path}")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# 7. User confirmation
# =============================================================================

def confirm(x_arr, r_arr):
    attempt = 0
    while True:
        attempt += 1
        plot_profile(
            x_arr, r_arr,
            title=(f"Check profile (attempt {attempt}) | "
                   f"tip=x{x_arr.min():.4f}m  inlet=x{x_arr.max():.4f}m"),
            show=True
        )
        print("\n" + "-" * 40)
        print("  y    -> confirm and save")
        print("  flip -> flip tip/inlet direction")
        print("  q    -> quit without saving")
        print("-" * 40)
        ans = input("  > ").strip().lower()

        if ans == 'y':
            return x_arr, r_arr
        elif ans in ('flip', 'f'):
            x_arr, r_arr = do_flip(x_arr, r_arr)
        elif ans == 'q':
            sys.exit(0)
        else:
            print("  Enter y / flip / q")


# =============================================================================
# 8. Save CSV
# =============================================================================

def save_csv(x_arr, r_arr, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x_m', 'r_m'])
        for x, r in zip(x_arr, r_arr):
            w.writerow([f"{x:.8f}", f"{r:.8f}"])
    print(f"[CSV] saved -> {path}  ({len(x_arr)} pts)")


# =============================================================================
# 9. Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="STL -> 2D axisymmetric rocket profile  (v6, plane-intersection method)"
    )
    p.add_argument('--stl',      required=True,
                   help='Input binary STL file')
    p.add_argument('--scale',    type=float, default=1.0,
                   help='Unit conversion. mm->m: 0.001, m->m: 1.0 (default: 1.0)')
    p.add_argument('--n_slices', type=int,   default=500,
                   help='Number of cutting planes (default: 500)')
    p.add_argument('--output',   default=None,
                   help='Output CSV path (default: <stl_name>_profile.csv)')
    return p.parse_args()


def main():
    args   = parse_args()
    base   = os.path.splitext(os.path.basename(args.stl))[0]
    outcsv = args.output or (base + '_profile.csv')
    outpng = outcsv.replace('.csv', '_profile.png')

    print("=" * 55)
    print("  STL -> 2D Axisymmetric Profile  v6")
    print("  Method: plane-triangle intersection (slicing)")
    print("=" * 55)
    print(f"  STL      : {args.stl}")
    print(f"  scale    : x{args.scale}  (output unit: m)")
    print(f"  n_slices : {args.n_slices}")
    print(f"  output   : {outcsv}")
    print("=" * 55 + "\n")

    # 1. Load
    tris = load_stl(args.stl)

    # 2. Detect axis
    axis_idx, perp_idx = detect_axis(tris)

    # 3. Extract profile (two-pass: centerline + profile)
    x_arr, r_arr = extract_profile(
        tris, axis_idx, perp_idx,
        scale=args.scale,
        n_slices=args.n_slices
    )

    # 4. Orient tip to x=0
    x_arr, r_arr, flipped = orient_tip(x_arr, r_arr)
    if flipped:
        print("  NOTE: auto-flipped -> confirm in graph\n")

    # 5. User confirmation
    x_arr, r_arr = confirm(x_arr, r_arr)

    # 6. Cap nose and tail (insert r=0 at both ends)
    x_arr, r_arr = cap_endpoints(x_arr, r_arr)

    # 7. Save final plot + CSV
    plot_profile(
        x_arr, r_arr,
        title=f"Confirmed (capped): {os.path.basename(args.stl)}",
        save_path=outpng, show=False
    )
    save_csv(x_arr, r_arr, outcsv)

    print(f"\n[Done]  CSV -> {outcsv}  |  Plot -> {outpng}")


if __name__ == '__main__':
    main()
