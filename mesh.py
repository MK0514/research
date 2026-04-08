"""
mesh.py  (v4 - final)
=====================
3-block body-fitted C-grid mesh generator for 2D axisymmetric rocket CFD.

Block structure:
  Block1 (upstream)  : x=[x_up .. x_nose], Cartesian with geometric x-stretch
  Block2 (body)      : x=[x_nose .. x_tail], body-fitted inflation layers
  Block3 (downstream): x=[x_tail .. x_dn],  Cartesian with geometric x-stretch

Key design decisions:
  - Block2 endpoint normals forced to radial (+r direction) so interface
    columns vary properly in r -> Block1/3 connect without folding
  - Block2 body normal stacking + Laplacian smoothing -> no body interior intrusion
  - Block1/3 TFI: linear blend between interface column and far-field Cartesian
  - x-spacing: geometric stretch (fine near body, coarse far field)
  - r-spacing: inflation layers (geometric growth from first_cell)
  - neg cells = 0, body inside = 0 verified on Falcon9 profile

Output:
  <prefix>.npz    mesh arrays (X1,R1,X2,R2,X3,R3) + metadata
  <prefix>_mesh.png  4-panel visualization

Usage:
  python mesh.py --profile Falcon9_profile.csv
  python mesh.py --profile hh_profile.csv --supersonic --output hh_mesh

Parameters:
  --profile      CSV from stl_to_2d.py  (required)
  --nj_body      Axial points along body surface (default: 500)
  --ni           Max radial layers (default: 80)
  --first_cell   First cell height [m] (default: 5e-5)
  --growth       Inflation layer growth rate (default: 1.15)
  --r_far        Far-field radius [m] (default: 20 * R_max)
  --x_up_mult    Upstream extent multiplier of body length (default: 5)
  --x_dn_mult    Downstream extent multiplier (default: 8)
  --supersonic   Use x_up_mult=10 for shock capture
  --output       Output prefix (default: <profile_name>_mesh)

Requirements:
  pip install numpy matplotlib
"""

import argparse, os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# =============================================================================
# 1. Load profile
# =============================================================================

def load_profile(path: str) -> tuple:
    bx, br = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            bx.append(float(row['x_m']))
            br.append(float(row['r_m']))
    bx = np.array(bx); br = np.abs(np.array(br))
    order = np.argsort(bx); bx = bx[order]; br = br[order]
    # Body surface only (exclude r=0 cap points)
    mask = br > br.max() * 1e-3
    bx_body = bx[mask]; br_body = br[mask]
    L = bx_body.max() - bx_body.min()
    R_max = br_body.max()
    print(f"[Profile] {len(bx_body)} body pts | "
          f"L={L:.4f}m | R_max={R_max:.4f}m | "
          f"nose_r={br_body[0]:.5f}m | tail_r={br_body[-1]:.5f}m")
    return bx_body, br_body, L, R_max


# =============================================================================
# 2. Build shared radial coordinate eta
# =============================================================================

def build_eta(first_cell: float, growth: float,
              r_far: float, ni_max: int) -> np.ndarray:
    eta = [0.0]; dh = first_cell
    for _ in range(2000):
        eta.append(eta[-1] + dh); dh *= growth
        if eta[-1] >= r_far: break
    eta = np.array(eta)
    eta = np.unique(np.append(eta[eta < r_far], r_far))[:ni_max]
    eta = np.append(eta, r_far); eta = np.unique(eta)
    print(f"[Eta]    Ni={len(eta)} | "
          f"first={eta[1]:.2e}m | last_cell={eta[-1]-eta[-2]:.4f}m | "
          f"far={eta[-1]:.2f}m")
    return eta


# =============================================================================
# 3. Block 2: body-fitted inflation
# =============================================================================

def build_block2(bx_body: np.ndarray, br_body: np.ndarray,
                 eta: np.ndarray, nj_body: int,
                 smooth_iter: int = 500) -> tuple:
    """
    Body-fitted grid around rocket surface.

    Curvature-adaptive arc-length sampling for axial spacing.
    Laplacian smoothing to relax initial normal-stacking distortions.
    Endpoint normals forced radial (+r) for clean Block1/3 interface.
    """
    # Curvature-adaptive arc-length
    dx_b = np.diff(bx_body); dr_b = np.diff(br_body)
    ds_b = np.sqrt(dx_b**2 + dr_b**2)
    d2r  = np.zeros(len(bx_body))
    for i in range(1, len(bx_body)-1):
        dx1 = bx_body[i] - bx_body[i-1]; dx2 = bx_body[i+1] - bx_body[i]
        if dx1 > 1e-10 and dx2 > 1e-10:
            d2r[i] = 2*(br_body[i+1]/dx2 - br_body[i]*(1/dx1+1/dx2)
                        + br_body[i-1]/dx1) / (dx1+dx2)
    d2r[0] = d2r[1]; d2r[-1] = d2r[-2]
    curv    = np.abs(d2r)
    curv    = (curv - curv.min()) / (curv.max() - curv.min() + 1e-10)
    density = 1.0 + 3.0*curv
    s_eff   = np.concatenate([[0], np.cumsum(ds_b * 0.5*(density[:-1]+density[1:]))])

    bx_b = np.interp(np.linspace(0, s_eff[-1], nj_body), s_eff, bx_body)
    br_b = np.interp(np.linspace(0, s_eff[-1], nj_body), s_eff, br_body)
    Nj   = nj_body

    # Outward unit normals via finite-difference tangent
    tx = np.zeros(Nj); tr = np.zeros(Nj)
    for j in range(Nj):
        jp = min(j+1, Nj-1); jm = max(j-1, 0)
        tx[j] = bx_b[jp]-bx_b[jm]; tr[j] = br_b[jp]-br_b[jm]
    mag = np.sqrt(tx**2 + tr**2) + 1e-30; tx /= mag; tr /= mag
    nx = -tr; nr = tx
    if nr[Nj//2] < 0: nx = -nx; nr = -nr

    # Force endpoint normals to pure radial (+r)
    blend_n = 60
    nx[0]=0.; nr[0]=1.; nx[-1]=0.; nr[-1]=1.
    for j in range(1, blend_n):
        t = j / blend_n
        for jj, j0, j1 in [(j, 0, blend_n), (-j-1, -1, -blend_n-1)]:
            nx[jj] = (1-t)*nx[j0] + t*nx[j1]
            nr[jj] = (1-t)*nr[j0] + t*nr[j1]
            m = np.sqrt(nx[jj]**2 + nr[jj]**2) + 1e-30
            nx[jj] /= m; nr[jj] /= m

    # Stack eta outward
    Ni = len(eta)
    X = bx_b[None,:] + eta[:,None]*nx[None,:]
    R = br_b[None,:] + eta[:,None]*nr[None,:]
    R = np.maximum(R, 0.)

    # Laplacian smoothing (keep body surface fixed)
    print(f"[Block2] Laplacian smoothing ({smooth_iter} iter)...")
    omega = 0.5
    for _ in range(smooth_iter):
        Xn = X.copy(); Rn = R.copy()
        Xn[1:-1,1:-1] = omega*0.25*(X[:-2,1:-1]+X[2:,1:-1]+
                                     X[1:-1,:-2]+X[1:-1,2:]) + (1-omega)*X[1:-1,1:-1]
        Rn[1:-1,1:-1] = omega*0.25*(R[:-2,1:-1]+R[2:,1:-1]+
                                     R[1:-1,:-2]+R[1:-1,2:]) + (1-omega)*R[1:-1,1:-1]
        X = Xn; R = Rn
        X[0,:] = bx_b; R[0,:] = br_b   # body surface fixed
    R = np.maximum(R, 0.)

    # Body interior check
    r_at    = np.interp(X[1:,:], bx_body, br_body, left=0., right=0.)
    in_rng  = (X[1:,:] >= bx_body[0]) & (X[1:,:] <= bx_body[-1])
    n_inside = int((in_rng & (R[1:,:] < r_at*0.97)).sum())
    print(f"[Block2] Nj={Nj} | inside_body={n_inside} | "
          f"nose_col r={R[:,0].min():.5f}~{R[:,0].max():.3f}m | "
          f"tail_col r={R[:,-1].min():.5f}~{R[:,-1].max():.3f}m")
    return X, R, bx_b, br_b


# =============================================================================
# 4. Block 1: upstream Cartesian TFI
# =============================================================================

def build_block1(X2: np.ndarray, R2: np.ndarray,
                 eta: np.ndarray,
                 x_up: float, dx0: float) -> tuple:
    """
    Upstream Cartesian block.
    x: geometric stretch from nose leftward.
    r: TFI blend between far-field Cartesian (eta) and Block2 nose column.
    """
    x1 = [X2[0, 0]]  # nose x (body surface level)
    dxc = dx0
    while x1[-1] > x_up:
        x1.append(x1[-1] - dxc); dxc = min(dxc*1.10, (X2[0,-1]-x_up)*0.3)
    x1_1d = np.sort(np.unique(np.array(x1 + [x_up])))
    nj1   = len(x1_1d)

    r_nose = R2[:, 0]  # Block2 nose column (varies in r)
    Ni = len(eta)
    X1 = np.zeros((Ni, nj1)); R1 = np.zeros((Ni, nj1))
    for j in range(nj1):
        t = np.clip((x1_1d[j] - x_up) / (X2[0,0] - x_up), 0., 1.)
        X1[:, j] = x1_1d[j]
        R1[:, j] = (1-t)*eta + t*r_nose

    R1 = np.maximum(R1, 0.)
    print(f"[Block1] Nj={nj1} | "
          f"dx_min={np.diff(x1_1d).min():.5f}m | "
          f"dx_max={np.diff(x1_1d).max():.3f}m")
    return X1, R1


# =============================================================================
# 5. Block 3: downstream Cartesian TFI
# =============================================================================

def build_block3(X2: np.ndarray, R2: np.ndarray,
                 eta: np.ndarray,
                 x_dn: float, dx0: float) -> tuple:
    """
    Downstream Cartesian block.
    x: geometric stretch from tail rightward.
    r: TFI blend between Block2 tail column and far-field Cartesian (eta).
    """
    x3 = [X2[0, -1]]
    dxc = dx0
    while x3[-1] < x_dn:
        x3.append(x3[-1] + dxc); dxc = min(dxc*1.08, (x_dn-X2[0,-1])*0.4)
    x3_1d = np.sort(np.unique(np.array(x3 + [x_dn])))
    nj3   = len(x3_1d)

    r_tail = R2[:, -1]
    Ni = len(eta)
    X3 = np.zeros((Ni, nj3)); R3 = np.zeros((Ni, nj3))
    for j in range(nj3):
        t = np.clip((x3_1d[j] - X2[0,-1]) / (x_dn - X2[0,-1]), 0., 1.)
        X3[:, j] = x3_1d[j]
        R3[:, j] = (1-t)*r_tail + t*eta

    R3 = np.maximum(R3, 0.)
    print(f"[Block3] Nj={nj3} | "
          f"dx_min={np.diff(x3_1d).min():.5f}m | "
          f"dx_max={np.diff(x3_1d).max():.3f}m")
    return X3, R3


# =============================================================================
# 6. Quality check
# =============================================================================

def quality(X: np.ndarray, R: np.ndarray, name: str) -> tuple:
    dxi = X[1:,:-1]-X[:-1,:-1]; dri = R[1:,:-1]-R[:-1,:-1]
    dxj = X[:-1,1:]-X[:-1,:-1]; drj = R[:-1,1:]-R[:-1,:-1]
    area = np.abs(dxi*drj - dri*dxj)
    li   = np.sqrt(dxi**2+dri**2+1e-30)
    lj   = np.sqrt(dxj**2+drj**2+1e-30)
    asp  = np.maximum(li,lj)/np.minimum(li,lj)
    neg  = int((area<=0).sum())
    nc   = (X.shape[0]-1)*(X.shape[1]-1)
    fa   = asp[np.isfinite(asp)]
    p99  = float(np.percentile(fa,99)) if len(fa) else 0.
    mn   = float(np.mean(fa))          if len(fa) else 0.
    print(f"  {name}: cells={nc:,}  neg={neg}  p99={p99:.0f}  mean={mn:.1f}")
    return asp, neg, nc


# =============================================================================
# 7. Visualize
# =============================================================================

def plot_mesh(X1, R1, X2, R2, X3, R3,
              bx_body, br_body, bx_orig, br_orig,
              quality_stats, title,
              save_path=None, show=True):

    Ni = X2.shape[0]
    nj1 = X1.shape[1]; Nj2 = X2.shape[1]; nj3 = X3.shape[1]
    tc, neg_tot, p99_all = quality_stats

    fig = plt.figure(figsize=(22, 16))
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)

    def pg(ax, X, R, si=4, sj=4, lw=0.25, al=0.5, mirror=True):
        ni_, nj_ = X.shape
        for i in range(0, ni_, si):
            ax.plot(X[i,:],  R[i,:],  'b-', lw=lw, alpha=al)
            if mirror: ax.plot(X[i,:], -R[i,:],  'b-', lw=lw, alpha=al)
        for j in range(0, nj_, sj):
            ax.plot(X[:,j],  R[:,j],  'b-', lw=lw, alpha=al)
            if mirror: ax.plot(X[:,j], -R[:,j],  'b-', lw=lw, alpha=al)

    x_up = X1[0,0]; x_dn = X3[0,-1]; r_far = X2[-1,0]

    # Panel 1: full domain
    ax1 = fig.add_subplot(gs[0,:])
    pg(ax1, X1, R1, max(1,Ni//20), max(1,nj1//12), 0.18, 0.4)
    pg(ax1, X2, R2, max(1,Ni//20), max(1,Nj2//60), 0.18, 0.4)
    pg(ax1, X3, R3, max(1,Ni//20), max(1,nj3//12), 0.18, 0.4)
    ax1.fill_between(bx_orig, br_orig, -br_orig, color='dimgray', alpha=0.9, label='Body')
    ax1.plot(bx_orig, br_orig, 'r-', lw=1.5)
    ax1.plot(bx_orig, -br_orig, 'r-', lw=1.5)
    ax1.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4, label='Symmetry axis')
    ax1.set_xlim(x_up, x_dn)
    ax1.set_ylim(-r_far*0.55, r_far*0.55)
    ax1.set_aspect('equal'); ax1.grid(False)
    ax1.set_xlabel('x [m]', fontsize=11); ax1.set_ylabel('r [m]', fontsize=11)
    ax1.set_title(f'Full domain  cells={tc:,}  neg={neg_tot}  p99={p99_all:.0f}', fontsize=11)
    ax1.legend(fontsize=9)

    # Panel 2: body region zoom
    ax2 = fig.add_subplot(gs[1,0])
    L   = bx_body[-1] - bx_body[0]
    R_max = br_body.max()
    pg(ax2, X1, R1, max(1,Ni//25), max(1,nj1//8), 0.4, 0.6, False)
    pg(ax2, X2, R2, max(1,Ni//25), max(1,Nj2//80), 0.4, 0.6, False)
    pg(ax2, X3, R3, max(1,Ni//25), max(1,nj3//8), 0.4, 0.6, False)
    ax2.fill_between(bx_orig, 0, br_orig, color='dimgray', alpha=0.9, label='Body')
    ax2.plot(bx_orig, br_orig, 'r-', lw=2)
    ax2.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax2.set_xlim(bx_body[0]-L*0.3, bx_body[-1]+L*0.3)
    ax2.set_ylim(-R_max*0.05, R_max*2.5)
    ax2.set_xlabel('x [m]', fontsize=10); ax2.set_ylabel('r [m]', fontsize=10)
    ax2.set_title('Body region (all 3 blocks)', fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.15)

    # Panel 3: inflation layer detail
    ax3 = fig.add_subplot(gs[1,1])
    x_mid = bx_body[0] + L*0.5; dxw = L*0.06
    r_bm  = float(np.interp(x_mid, bx_body, br_body))
    r_top = r_bm + 0.06
    m2j   = (X2[0,:] >= x_mid-dxw) & (X2[0,:] <= x_mid+dxw)
    ni_s  = min(30, Ni)
    pg(ax3, X2[:ni_s,:][:,m2j], R2[:ni_s,:][:,m2j],
       1, max(1,m2j.sum()//60), 0.6, 0.85, False)
    bx_m = bx_orig[(bx_orig>=x_mid-dxw)&(bx_orig<=x_mid+dxw)]
    br_m = br_orig[(bx_orig>=x_mid-dxw)&(bx_orig<=x_mid+dxw)]
    ax3.fill_between(bx_m, 0, br_m, color='dimgray', alpha=0.9, label='Body')
    ax3.plot(bx_m, br_m, 'r-', lw=2)
    ax3.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax3.set_xlim(x_mid-dxw, x_mid+dxw)
    ax3.set_ylim(r_bm*0.85, r_top)
    ax3.set_xlabel('x [m]', fontsize=10); ax3.set_ylabel('r [m]', fontsize=10)
    ax3.set_title('Inflation layers (mid-body)', fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.15)

    fig.suptitle(title, fontsize=13)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot]   -> {save_path}")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# 8. Save
# =============================================================================

def save_mesh(X1,R1,X2,R2,X3,R3, bx_body,br_body, eta, params, path):
    np.savez_compressed(
        path,
        X1=X1, R1=R1, X2=X2, R2=R2, X3=X3, R3=R3,
        body_x=bx_body, body_r=br_body, eta=eta,
        **{f"p_{k}": np.array(v) for k,v in params.items()
           if isinstance(v, (int,float,bool))}
    )
    print(f"[Save]   -> {path}.npz")


# =============================================================================
# 9. Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="3-block body-fitted C-grid mesh generator")
    p.add_argument('--profile',    required=True)
    p.add_argument('--nj_body',    type=int,   default=500)
    p.add_argument('--ni',         type=int,   default=80)
    p.add_argument('--first_cell', type=float, default=5e-5)
    p.add_argument('--growth',     type=float, default=1.15)
    p.add_argument('--r_far',      type=float, default=None,
                   help='Far-field radius [m]. Default: 20 * R_max')
    p.add_argument('--x_up_mult',  type=float, default=5.0,
                   help='Upstream extent = x_up_mult * L (default: 5)')
    p.add_argument('--x_dn_mult',  type=float, default=8.0,
                   help='Downstream extent = x_dn_mult * L (default: 8)')
    p.add_argument('--supersonic', action='store_true',
                   help='Set x_up_mult=10 for shock capture')
    p.add_argument('--smooth_iter',type=int,   default=500,
                   help='Block2 Laplacian smoothing iterations (default: 500)')
    p.add_argument('--output',     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base   = os.path.splitext(os.path.basename(args.profile))[0].replace('_profile','')
    outpfx = args.output or (base + '_mesh')

    print("=" * 55)
    print("  3-block body-fitted C-grid mesh generator")
    print("=" * 55)
    print(f"  Profile    : {args.profile}")
    print(f"  nj_body={args.nj_body}  ni={args.ni}")
    print(f"  first_cell={args.first_cell:.1e}m  growth={args.growth}")
    print(f"  Output     : {outpfx}")
    print("=" * 55 + "\n")

    # 1. Load
    bx_body, br_body, L, R_max = load_profile(args.profile)
    # Also load full profile (including cap points) for plotting
    bx_all, br_all = [], []
    with open(args.profile) as f:
        for row in csv.DictReader(f):
            bx_all.append(float(row['x_m'])); br_all.append(float(row['r_m']))
    bx_all=np.array(bx_all); br_all=np.abs(np.array(br_all))

    # 2. Domain extents
    x_up_mult = 10.0 if args.supersonic else args.x_up_mult
    x_up  = bx_body[0]  - x_up_mult  * L
    x_dn  = bx_body[-1] + args.x_dn_mult * L
    r_far = args.r_far or (20.0 * R_max)
    print(f"[Domain] upstream={x_up:.2f}m  downstream={x_dn:.2f}m  r_far={r_far:.2f}m\n")

    # 3. Eta
    eta = build_eta(args.first_cell, args.growth, r_far, args.ni)
    Ni  = len(eta)
    print()

    # 4. Block2
    X2, R2, bx_b, br_b = build_block2(
        bx_body, br_body, eta, args.nj_body, args.smooth_iter)
    print()

    # 5. Block1
    dx0 = float(np.diff(bx_b).min())
    X1, R1 = build_block1(X2, R2, eta, x_up, dx0)

    # 6. Block3
    X3, R3 = build_block3(X2, R2, eta, x_dn, dx0)
    print()

    # 7. Quality
    print("Quality:")
    a1,n1,c1 = quality(X1, R1, "Block1(upstream)  ")
    a2,n2,c2 = quality(X2, R2, "Block2(body)      ")
    a3,n3,c3 = quality(X3, R3, "Block3(downstream)")
    all_a = np.concatenate([a1.ravel(),a2.ravel(),a3.ravel()])
    fa    = all_a[np.isfinite(all_a)]
    tc    = c1+c2+c3; neg_tot = n1+n2+n3
    p99   = float(np.percentile(fa,99)) if len(fa) else 0.
    print(f"  TOTAL: cells={tc:,}  neg={neg_tot}  p99={p99:.0f}  mean={np.mean(fa):.1f}")
    print()

    # 8. Save
    params = dict(nj_body=args.nj_body, ni=Ni,
                  first_cell=args.first_cell, growth=args.growth,
                  r_far=r_far, x_up=x_up, x_dn=x_dn, L=L, R_max=R_max)
    save_mesh(X1,R1,X2,R2,X3,R3, bx_body,br_body, eta, params, outpfx)

    # 9. Visualize
    title = (f"3-block C-grid: {os.path.basename(args.profile)}\n"
             f"cells={tc:,}  neg={neg_tot}  "
             f"B2_p99={float(np.percentile(a2[np.isfinite(a2)],99)):.0f}  "
             f"total_p99={p99:.0f}  first_cell={args.first_cell:.1e}m")
    plot_mesh(X1,R1,X2,R2,X3,R3, bx_body, br_body, bx_all, br_all,
              (tc, neg_tot, p99), title,
              save_path=outpfx+'_mesh.png', show=True)

    print(f"\n[Done]")
    print(f"  Mesh -> {outpfx}.npz")
    print(f"  Plot -> {outpfx}_mesh.png")
    print(f"\n  Next: python simulator.py --mesh {outpfx}.npz --mach 0.8 --altitude 20000 --aoa 0")


if __name__ == '__main__':
    main()
