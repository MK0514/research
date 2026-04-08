"""
mesh.py
=======
2D axisymmetric CFD mesh: Inflation + Cartesian background.
Works with any rocket profile CSV from stl_to_2d.py.

Inflation layer count AUTO-COMPUTED from target thickness:
  Default target = 1.0 x R_max  (covers full body radius)
  growth=1.1 (default) -> ~64 layers for Falcon9

Usage:
  python mesh.py --profile Falcon9_profile.csv
  python mesh.py --profile nuri.csv --output nuri_mesh
  python mesh.py --profile rocket.csv --target_thickness 0.5 --supersonic

Parameters:
  --profile          CSV from stl_to_2d.py (required)
  --first_cell       First layer height [m] (default: 5e-5)
  --growth           Growth rate (default: 1.1)
  --target_thickness Inflation thickness [m] (default: 1.0 x R_max)
  --n_inf            Override: fixed layer count
  --nj_body          Body surface points (default: 500)
  --nj_cap           Nose/tail cap points each (default: 60)
  --smooth_iter      Normal smoothing iterations (default: 3000)
  --r_far            Far-field radius [m] (default: 20 x R_max)
  --x_up_mult        Upstream extent in body lengths (default: 5)
  --x_dn_mult        Downstream extent (default: 15)
  --supersonic       Set x_up_mult=10
  --output           Output prefix (default: <profile>_mesh)
"""
import argparse, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def load_profile(path):
    bx, br = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            bx.append(float(row['x_m']))
            br.append(float(row['r_m']))
    bx = np.array(bx); br = np.abs(np.array(br))
    idx = np.argsort(bx); bx = bx[idx]; br = br[idx]
    mask = br > br.max() * 1e-3
    bx_b = bx[mask]; br_b = br[mask]
    L = bx_b.max() - bx_b.min(); R_max = br_b.max()
    print(f"[Profile] {len(bx_b)} pts | L={L:.4f}m | R_max={R_max:.4f}m | "
          f"nose_r={br_b[0]:.5f}m | tail_r={br_b[-1]:.5f}m")
    return bx_b, br_b, bx, br, L, R_max


def compute_n_inf(first_cell, growth, target):
    eta = [0.]; dh = first_cell; n = 0
    while eta[-1] < target:
        eta.append(eta[-1] + dh); dh *= growth; n += 1
        if n > 5000: raise ValueError("Cannot reach target thickness")
    print(f"[Inflate] Auto n_inf={n} | target={target:.4f}m | actual={eta[-1]:.5f}m")
    return n


def build_inflation(bx_b, br_b, first_cell, growth, n_inf,
                    nj_body, nj_cap, smooth_iter):
    # Eta
    eta = [0.]; dh = first_cell
    for _ in range(n_inf): eta.append(eta[-1] + dh); dh *= growth
    eta = np.array(eta); d_inf = eta[-1]
    print(f"[Inflate] Ni={len(eta)} | first={eta[1]:.2e}m | "
          f"thickness={d_inf:.5f}m | last_cell={d_inf-eta[-2]:.5f}m")

    # Nose cap: x=x_nose, r: 0->br_b[0], normal=(-1,0)
    seg_nx  = np.full(nj_cap, bx_b[0])
    seg_nr  = np.linspace(0., br_b[0], nj_cap)
    seg_nnx = np.full(nj_cap, -1.); seg_nnr = np.zeros(nj_cap)

    # Body surface: arc-length uniform, tangent-based normals
    ds = np.sqrt(np.diff(bx_b)**2 + np.diff(br_b)**2)
    s  = np.concatenate([[0], np.cumsum(ds)])
    su = np.linspace(0, s[-1], nj_body)
    bx_s = np.interp(su, s, bx_b); br_s = np.interp(su, s, br_b)
    tx = np.gradient(bx_s); tr = np.gradient(br_s)
    mag = np.sqrt(tx**2 + tr**2) + 1e-30; tx /= mag; tr /= mag
    nx_b = -tr; nr_b = tx
    if nr_b[nj_body // 2] < 0: nx_b = -nx_b; nr_b = -nr_b

    # Tail cap: x=x_tail, r: br_b[-1]->0, normal=(+1,0)
    seg_tx  = np.full(nj_cap, bx_b[-1])
    seg_tr  = np.linspace(br_b[-1], 0., nj_cap)
    seg_tnx = np.full(nj_cap, 1.); seg_tnr = np.zeros(nj_cap)

    # Combine
    all_x  = np.concatenate([seg_nx,   bx_s[1:-1],  seg_tx])
    all_r  = np.concatenate([seg_nr,   br_s[1:-1],  seg_tr])
    all_nx = np.concatenate([seg_nnx,  nx_b[1:-1],  seg_tnx])
    all_nr = np.concatenate([seg_nnr,  nr_b[1:-1],  seg_tnr])
    Nj = len(all_x)

    # Global smooth: blends cap<->body transitions
    print(f"[Inflate] Smoothing normals ({smooth_iter} iter)...")
    for _ in range(smooth_iter):
        nxn = all_nx.copy(); nrn = all_nr.copy()
        nxn[1:-1] = 0.05*all_nx[:-2] + 0.9*all_nx[1:-1] + 0.05*all_nx[2:]
        nrn[1:-1] = 0.05*all_nr[:-2] + 0.9*all_nr[1:-1] + 0.05*all_nr[2:]
        all_nx = nxn; all_nr = nrn
        m = np.sqrt(all_nx**2 + all_nr**2) + 1e-30
        all_nx /= m; all_nr /= m
    all_nx[0] = -1.; all_nr[0] = 0.   # nose bottom: upstream
    all_nx[-1] = 1.; all_nr[-1] = 0.  # tail bottom: downstream

    # Extrude
    XI = all_x[None, :] + eta[:, None] * all_nx[None, :]
    RI = all_r[None, :] + eta[:, None] * all_nr[None, :]
    RI = np.maximum(RI, 0.)

    # Quality
    dxi=XI[1:,:-1]-XI[:-1,:-1]; dri=RI[1:,:-1]-RI[:-1,:-1]
    dxj=XI[:-1,1:]-XI[:-1,:-1]; drj=RI[:-1,1:]-RI[:-1,:-1]
    area = np.abs(dxi*drj - dri*dxj)
    asp  = (np.maximum(np.sqrt(dxi**2+dri**2), np.sqrt(dxj**2+drj**2)) /
            (np.minimum(np.sqrt(dxi**2+dri**2), np.sqrt(dxj**2+drj**2)) + 1e-30))
    neg    = int((area <= 0).sum())
    r_at   = np.interp(XI[1:,:], bx_b, br_b, left=0., right=0.)
    in_x   = (XI[1:,:] >= bx_b[0]) & (XI[1:,:] <= bx_b[-1])
    inside = int((in_x & (RI[1:,:] < r_at * 0.98)).sum())
    fa     = asp[np.isfinite(asp)]
    print(f"[Inflate] Nj={Nj} | neg={neg} | inside={inside} | "
          f"p99={np.percentile(fa,99):.0f} | cells={len(eta)*Nj:,}")
    return XI, RI, eta, d_inf, Nj


def build_cartesian(bx_b, br_b, d_inf, x_up, x_dn, r_far, L, R_max):
    dx_med   = L / 60
    x_fine_l = bx_b[0]  - 2.0 * L
    x_fine_r = bx_b[-1] + 3.0 * L

    def stretch(x0, x1, dx0, ratio=1.08, dx_max=L*0.4):
        pts = [x0]; dx = dx0; sign = 1 if x1 > x0 else -1
        while sign * (x1 - pts[-1]) > 1e-9:
            step = min(dx, abs(x1 - pts[-1]))
            pts.append(pts[-1] + sign * step)
            dx = min(dx * ratio, dx_max)
        return np.array(pts)

    x_far_up = stretch(x_fine_l, x_up, dx_med*2)[::-1]
    x_med_up = np.linspace(x_fine_l, bx_b[0],
                           max(3, int((bx_b[0]-x_fine_l)/dx_med)+2))
    x_fine   = np.linspace(bx_b[0], bx_b[-1], 401)
    x_med_dn = np.linspace(bx_b[-1], x_fine_r,
                           max(3, int((x_fine_r-bx_b[-1])/dx_med)+2))
    x_far_dn = stretch(x_fine_r, x_dn, dx_med*2, 1.08, L*0.5)
    x_bg = np.unique(np.concatenate([x_far_up, x_med_up, x_fine,
                                     x_med_dn, x_far_dn]))

    r_bg = [d_inf]; dr = d_inf * 0.5
    while r_bg[-1] < r_far:
        r_bg.append(r_bg[-1] + dr); dr = min(dr*1.15, R_max)
    r_bg.append(r_far)
    r_bg = np.unique(np.concatenate([[0., d_inf*0.5], np.array(r_bg)]))

    Nx = len(x_bg); Nr = len(r_bg)
    XX, RR = np.meshgrid(x_bg, r_bg, indexing='ij')
    r_body  = np.interp(x_bg, bx_b, br_b, left=0., right=0.)
    is_solid = ((RR < r_body[:, None]) &
                (XX >= bx_b[0]) & (XX <= bx_b[-1]))
    n_fluid = int((~is_solid[:-1,:-1]).sum())
    print(f"[Cartesian] Nx={Nx} Nr={Nr} | x=[{x_bg[0]:.1f},{x_bg[-1]:.1f}]m | "
          f"r=[0,{r_bg[-1]:.2f}]m | fluid_cells={n_fluid:,}")
    return x_bg, r_bg, is_solid


def plot_mesh(XI, RI, x_bg, r_bg, bx_b, br_b, bx_all, br_all,
              eta, d_inf, neg, inside, first_cell, growth, title, save_path):
    Ni, Nj = XI.shape
    L = bx_b.max()-bx_b.min(); R_max = br_b.max()
    Nx = len(x_bg); x_up=x_bg[0]; x_dn=x_bg[-1]; r_far=r_bg[-1]

    fig = plt.figure(figsize=(24, 20))
    gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32)

    def lcp(ax, X, R, si, sj, col='b', lw=0.3, al=0.7, mirror=False):
        ni_, nj_ = X.shape
        segs = ([list(zip(X[i,:],R[i,:])) for i in range(0,ni_,si)] +
                [list(zip(X[:,j],R[:,j])) for j in range(0,nj_,sj)])
        ax.add_collection(LineCollection(segs, colors=col, lw=lw, alpha=al))
        if mirror:
            segsm = ([list(zip(X[i,:],-R[i,:])) for i in range(0,ni_,si)] +
                     [list(zip(X[:,j],-R[:,j])) for j in range(0,nj_,sj)])
            ax.add_collection(LineCollection(segsm, colors=col, lw=lw, alpha=al))

    def bgd(ax, xb, rb, sx=1, sr=1, r_max=None, col='c', lw=0.3, al=0.5, mirror=False):
        rl = rb[rb <= (r_max if r_max else rb[-1])]
        if len(rl) < 2: return
        segs = ([([(xi,0),(xi,rl[-1])]) for xi in xb[::sx]] +
                [([(xb[0],ri),(xb[-1],ri)]) for ri in rl[::sr]])
        ax.add_collection(LineCollection(segs, colors=col, lw=lw, alpha=al))
        if mirror:
            segsm = ([([(xi,0),(xi,-rl[-1])]) for xi in xb[::sx]] +
                     [([(xb[0],-ri),(xb[-1],-ri)]) for ri in rl[::sr]])
            ax.add_collection(LineCollection(segsm, colors=col, lw=lw, alpha=al))

    # P1: Full domain
    ax1 = fig.add_subplot(gs[0,:])
    bgd(ax1, x_bg, r_bg, sx=max(1,Nx//100), sr=1, lw=0.25, al=0.5)
    lcp(ax1, XI, RI, max(1,Ni//5), max(1,Nj//50), lw=0.4, al=0.85)
    ax1.fill_between(bx_all, 0, br_all, color='dimgray', alpha=0.95, label='Body')
    ax1.plot(bx_all, br_all, 'r-', lw=2)
    ax1.axhline(0, color='k', lw=1., ls='--', alpha=0.6, label='Sym. axis')
    ax1.set_xlim(x_up, x_dn); ax1.set_ylim(0, r_far)
    ax1.set_xlabel('x [m]',fontsize=12); ax1.set_ylabel('r [m]',fontsize=12)
    ax1.set_title(f'Full domain | neg={neg} inside={inside}', fontsize=11)
    ax1.legend(fontsize=9)

    # P2: Nose zoom
    ax2 = fig.add_subplot(gs[1,0])
    xz = bx_b[0]+L*0.25; rz = R_max*4.
    xbz = x_bg[(x_bg>=bx_b[0]-L*0.18)&(x_bg<=xz)]
    bgd(ax2, xbz, r_bg, sx=1, sr=1, r_max=rz, lw=0.5, al=0.6)
    lcp(ax2, XI, RI, 1, max(1,Nj//60), lw=0.5, al=0.9)
    ax2.fill_between(bx_all, 0, br_all, color='dimgray', alpha=0.95)
    ax2.plot(bx_all, br_all, 'r-', lw=2)
    ax2.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax2.set_xlim(bx_b[0]-L*0.18, xz); ax2.set_ylim(0, rz); ax2.autoscale_view()
    ax2.set_xlabel('x [m]',fontsize=11); ax2.set_ylabel('r [m]',fontsize=11)
    ax2.set_title('Nose region', fontsize=11); ax2.grid(True, alpha=0.2)

    # P3: Tail zoom
    ax3 = fig.add_subplot(gs[1,1])
    xzl=bx_b[-1]-L*0.25; xzr=bx_b[-1]+L*0.8; rz=R_max*4.
    xbz = x_bg[(x_bg>=xzl)&(x_bg<=xzr)]
    bgd(ax3, xbz, r_bg, sx=1, sr=1, r_max=rz, lw=0.5, al=0.6)
    lcp(ax3, XI, RI, 1, max(1,Nj//60), lw=0.5, al=0.9)
    ax3.fill_between(bx_all, 0, br_all, color='dimgray', alpha=0.95)
    ax3.plot(bx_all, br_all, 'r-', lw=2)
    ax3.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax3.set_xlim(xzl, xzr); ax3.set_ylim(0, rz); ax3.autoscale_view()
    ax3.set_xlabel('x [m]',fontsize=11); ax3.set_ylabel('r [m]',fontsize=11)
    ax3.set_title('Tail region', fontsize=11); ax3.grid(True, alpha=0.2)

    # P4: Mid-body inflation detail
    ax4 = fig.add_subplot(gs[2,:])
    x_mid=bx_b[0]+L*0.5; dxw=L*0.12
    r_bm=float(np.interp(x_mid, bx_b, br_b))
    j_mask=(XI[0,:]>=x_mid-dxw)&(XI[0,:]<=x_mid+dxw)
    if j_mask.sum() > 0:
        Xi=XI[:,j_mask]; Ri=RI[:,j_mask]
        segs=([list(zip(Xi[i,:],Ri[i,:])) for i in range(Ni)] +
              [list(zip(Xi[:,j],Ri[:,j]))
               for j in range(0,Xi.shape[1],max(1,Xi.shape[1]//80))])
        ax4.add_collection(LineCollection(segs, colors='b', lw=0.7, alpha=0.9))
    xbz=x_bg[(x_bg>=x_mid-dxw)&(x_bg<=x_mid+dxw)]
    r_top=r_bm+d_inf*1.3
    bgd(ax4, xbz, r_bg, sx=1, sr=1, r_max=r_top, lw=0.6, al=0.7)
    bx_m=bx_b[(bx_b>=x_mid-dxw)&(bx_b<=x_mid+dxw)]
    br_m=br_b[(bx_b>=x_mid-dxw)&(bx_b<=x_mid+dxw)]
    ax4.fill_between(bx_m, 0, br_m, color='dimgray', alpha=0.95)
    ax4.plot(bx_m, br_m, 'r-', lw=2)
    ax4.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax4.set_xlim(x_mid-dxw, x_mid+dxw); ax4.set_ylim(r_bm*0.80, r_top)
    ax4.autoscale_view()
    ax4.set_xlabel('x [m]',fontsize=11); ax4.set_ylabel('r [m]',fontsize=11)
    ax4.set_title(f'Mid-body inflation | '
                  f'first={first_cell:.1e}m  growth={growth}  '
                  f'thickness={d_inf:.4f}m  ({Ni-1} layers)', fontsize=11)
    ax4.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Plot]   -> {save_path}")
    plt.close('all')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--profile',          required=True)
    p.add_argument('--first_cell',       type=float, default=5e-5)
    p.add_argument('--growth',           type=float, default=1.1)
    p.add_argument('--target_thickness', type=float, default=None)
    p.add_argument('--n_inf',            type=int,   default=None)
    p.add_argument('--nj_body',          type=int,   default=500)
    p.add_argument('--nj_cap',           type=int,   default=60)
    p.add_argument('--smooth_iter',      type=int,   default=3000)
    p.add_argument('--r_far',            type=float, default=None)
    p.add_argument('--x_up_mult',        type=float, default=5.0)
    p.add_argument('--x_dn_mult',        type=float, default=15.0)
    p.add_argument('--supersonic',       action='store_true')
    p.add_argument('--output',           default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base   = os.path.splitext(os.path.basename(args.profile))[0].replace('_profile','')
    outpfx = args.output or (base + '_mesh')

    print("=" * 55)
    print("  Inflation + Cartesian Mesh Generator")
    print("=" * 55)
    print(f"  Profile : {args.profile}")
    print(f"  growth  : {args.growth}  first_cell : {args.first_cell:.1e}m")
    print("=" * 55 + "\n")

    bx_b, br_b, bx_all, br_all, L, R_max = load_profile(args.profile)

    x_up_m = 10. if args.supersonic else args.x_up_mult
    x_up   = bx_b[0]  - x_up_m * L
    x_dn   = bx_b[-1] + args.x_dn_mult * L
    r_far  = args.r_far or (20. * R_max)
    print(f"[Domain] x=[{x_up:.1f},{x_dn:.1f}]m | r=[0,{r_far:.2f}]m\n")

    if args.n_inf is not None:
        n_inf = args.n_inf
        print(f"[Inflate] Fixed n_inf={n_inf}")
    else:
        target = args.target_thickness or R_max
        n_inf  = compute_n_inf(args.first_cell, args.growth, target)

    XI, RI, eta, d_inf, Nj = build_inflation(
        bx_b, br_b, args.first_cell, args.growth, n_inf,
        args.nj_body, args.nj_cap, args.smooth_iter)
    Ni = len(eta)

    print()
    x_bg, r_bg, is_solid = build_cartesian(
        bx_b, br_b, d_inf, x_up, x_dn, r_far, L, R_max)

    # Final quality
    dxi=XI[1:,:-1]-XI[:-1,:-1]; dri=RI[1:,:-1]-RI[:-1,:-1]
    dxj=XI[:-1,1:]-XI[:-1,:-1]; drj=RI[:-1,1:]-RI[:-1,:-1]
    neg    = int((np.abs(dxi*drj-dri*dxj) <= 0).sum())
    r_at   = np.interp(XI[1:,:], bx_b, br_b, left=0., right=0.)
    in_x   = (XI[1:,:] >= bx_b[0]) & (XI[1:,:] <= bx_b[-1])
    inside = int((in_x & (RI[1:,:] < r_at*0.98)).sum())
    print(f"\n[Summary] inflation={Nj*(Ni-1):,}  "
          f"bg_fluid={int((~is_solid[:-1,:-1]).sum()):,}  "
          f"neg={neg}  inside={inside}")

    np.savez_compressed(outpfx+'.npz',
        XI=XI, RI=RI, x_bg=x_bg, r_bg=r_bg, is_solid=is_solid,
        body_x=bx_b, body_r=br_b, eta=eta,
        p_d_inf=np.array(d_inf), p_L=np.array(L), p_R_max=np.array(R_max),
        p_x_up=np.array(x_up), p_x_dn=np.array(x_dn), p_r_far=np.array(r_far))
    print(f"[Save]   -> {outpfx}.npz")

    title = (f"{os.path.basename(args.profile)} | "
             f"Inflation: {Ni-1}L  {d_inf:.4f}m | neg={neg}  inside={inside}")
    plot_mesh(XI, RI, x_bg, r_bg, bx_b, br_b, bx_all, br_all,
              eta, d_inf, neg, inside, args.first_cell, args.growth,
              title, outpfx+'_mesh.png')

    print(f"\n[Done]  {outpfx}.npz  |  {outpfx}_mesh.png")


if __name__ == '__main__':
    main()
