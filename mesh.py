"""
mesh.py  (v7 - Pure Normal Extrusion)
======================================
2D axisymmetric mesh by direct normal extrusion from body surface.

Method:
  1. Resample body surface uniformly in arc-length
  2. Compute smooth outward unit normals
  3. Extrude inflation layers: X[i,j] = surface[j] + eta[i]*normal[j]
     -> body interior intrusion is structurally impossible
  4. Background Cartesian grid fills remaining domain

Parameters:
  --profile      CSV from stl_to_2d.py (required)
  --nj_body      Surface points along body (default: 600)
  --nj_cap       Points per nose/tail cap (default: 40)
  --n_inflate    Inflation layer count (default: 40)
  --first_cell   First layer height [m] (default: 5e-5)
  --growth       Growth rate (default: 1.15)
  --r_far        Far-field radius [m] (default: 12*R_max)
  --x_up_mult    Upstream extent multiplier (default: 5)
  --x_dn_mult    Downstream extent multiplier (default: 7)
  --supersonic   Set x_up_mult=10
  --output       Output prefix

Requirements: pip install numpy matplotlib
"""
import argparse, os, csv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def load_profile(path):
    bx, br = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            bx.append(float(row['x_m'])); br.append(float(row['r_m']))
    bx=np.array(bx); br=np.abs(np.array(br))
    idx=np.argsort(bx); bx=bx[idx]; br=br[idx]
    mask=br>br.max()*1e-3
    bx_b=bx[mask]; br_b=br[mask]
    bx_all=bx; br_all=br
    L=bx_b.max()-bx_b.min(); R_max=br_b.max()
    print(f"[Profile] {len(bx_b)} pts | L={L:.4f}m | R_max={R_max:.4f}m")
    return bx_b, br_b, bx_all, br_all, L, R_max


def build_surface(bx_b, br_b, nj_body, nj_cap):
    """Arc-length uniform resampling + smooth outward normals + caps."""
    ds=np.sqrt(np.diff(bx_b)**2+np.diff(br_b)**2)
    s=np.concatenate([[0],np.cumsum(ds)])
    bx_s=np.interp(np.linspace(0,s[-1],nj_body),s,bx_b)
    br_s=np.interp(np.linspace(0,s[-1],nj_body),s,br_b)

    # Normals
    tx=np.gradient(bx_s); tr=np.gradient(br_s)
    mag=np.sqrt(tx**2+tr**2)+1e-30; tx/=mag; tr/=mag
    nx=-tr; nr=tx
    if nr[nj_body//2]<0: nx=-nx; nr=-nr

    # Smooth normals
    for _ in range(30):
        nxn=nx.copy(); nrn=nr.copy()
        nxn[1:-1]=0.5*(nx[:-2]+nx[2:]); nrn[1:-1]=0.5*(nr[:-2]+nr[2:])
        nx=nxn; nr=nrn
        m=np.sqrt(nx**2+nr**2)+1e-30; nx/=m; nr/=m

    # Blend endpoints to upstream/downstream directions
    blend=min(40,nj_body//4)
    for j in range(blend):
        t=j/blend
        # Nose end -> pure upstream (-1,0)
        nx[j]=(1-t)*(-1)+t*nx[blend]; nr[j]=(1-t)*0+t*nr[blend]
        m=np.sqrt(nx[j]**2+nr[j]**2)+1e-30; nx[j]/=m; nr[j]/=m
        # Tail end -> pure downstream (+1,0)
        jj=nj_body-1-j
        nx[jj]=(1-t)*(1)+t*nx[nj_body-1-blend]
        nr[jj]=(1-t)*0+t*nr[nj_body-1-blend]
        m=np.sqrt(nx[jj]**2+nr[jj]**2)+1e-30; nx[jj]/=m; nr[jj]/=m

    # Caps
    nose_bx=np.full(nj_cap,bx_b[0]); nose_br=np.linspace(0,br_b[0],nj_cap)
    tail_bx=np.full(nj_cap,bx_b[-1]); tail_br=np.linspace(br_b[-1],0,nj_cap)
    nose_nx=np.full(nj_cap,-1.); nose_nr=np.zeros(nj_cap)
    tail_nx=np.full(nj_cap,1.); tail_nr=np.zeros(nj_cap)

    all_bx=np.concatenate([nose_bx,bx_s[1:-1],tail_bx])
    all_br=np.concatenate([nose_br,br_s[1:-1],tail_br])
    all_nx=np.concatenate([nose_nx,nx[1:-1],tail_nx])
    all_nr=np.concatenate([nose_nr,nr[1:-1],tail_nr])
    Nj=len(all_bx)
    print(f"[Surface] Nj={Nj} | cap={nj_cap}+body={nj_body-2}+cap={nj_cap}")
    return all_bx, all_br, all_nx, all_nr


def build_eta(first_cell, growth, n_inflate):
    eta=[0.]; dh=first_cell
    for _ in range(n_inflate): eta.append(eta[-1]+dh); dh*=growth
    eta=np.array(eta)
    print(f"[Eta]    Ni={len(eta)} | first={eta[1]:.2e}m | outer={eta[-1]:.5f}m")
    return eta


def extrude(all_bx, all_br, all_nx, all_nr, eta):
    """Pure normal extrusion - no smoothing, no body intrusion."""
    X=all_bx[None,:]+eta[:,None]*all_nx[None,:]
    R=all_br[None,:]+eta[:,None]*all_nr[None,:]
    R=np.maximum(R,0.)
    print(f"[Extrude] {len(eta)}x{len(all_bx)}={len(eta)*len(all_bx):,} nodes")
    return X,R


def build_background(bx_b, br_b, x_up, x_dn, r_far, eta_outer):
    """Background Cartesian grid outside inflation block."""
    L=bx_b.max()-bx_b.min(); R_max=br_b.max()
    dx_body=L/300; dx_max=R_max*0.6
    x_up_pts=[bx_b[0]]
    dx=dx_body*5
    while x_up_pts[0]>x_up: x_up_pts.insert(0,x_up_pts[0]-dx); dx=min(dx*1.08,dx_max)
    x_dn_pts=[bx_b[-1]]
    dx=dx_body*5
    while x_dn_pts[-1]<x_dn: x_dn_pts.append(x_dn_pts[-1]+dx); dx=min(dx*1.08,dx_max)
    x_bg=np.unique(np.concatenate([x_up_pts,np.linspace(bx_b[0],bx_b[-1],301).tolist(),x_dn_pts]))
    r_bg=[0.,eta_outer]
    dr=eta_outer*0.6
    while r_bg[-1]<r_far: r_bg.append(r_bg[-1]+dr); dr=min(dr*1.15,R_max*0.7)
    r_bg.append(r_far); r_bg=np.unique(np.array(r_bg))
    print(f"[BG]     Nx={len(x_bg)}  Nr={len(r_bg)}")
    return x_bg, r_bg


def check_quality(X, R, bx_b, br_b):
    dxi=X[1:,:-1]-X[:-1,:-1]; dri=R[1:,:-1]-R[:-1,:-1]
    dxj=X[:-1,1:]-X[:-1,:-1]; drj=R[:-1,1:]-R[:-1,:-1]
    area=np.abs(dxi*drj-dri*dxj)
    li=np.sqrt(dxi**2+dri**2+1e-30); lj=np.sqrt(dxj**2+drj**2+1e-30)
    asp=np.maximum(li,lj)/np.minimum(li,lj)
    neg=int((area<=0).sum()); fa=asp[np.isfinite(asp)]
    p99=float(np.percentile(fa,99)); mn=float(np.mean(fa))
    r_at=np.interp(X[1:,:],bx_b,br_b,left=0.,right=0.)
    in_x=(X[1:,:]>=bx_b[0])&(X[1:,:]<=bx_b[-1])
    inside=int((in_x&(R[1:,:]<r_at*0.98)).sum())
    print(f"[Quality] cells={(X.shape[0]-1)*(X.shape[1]-1):,} | neg={neg} | p99={p99:.0f} | mean={mn:.1f} | inside={inside}")
    return dict(neg=neg,p99=p99,mean=mn,inside=inside,nc=(X.shape[0]-1)*(X.shape[1]-1))


def plot_mesh(X,R,x_bg,r_bg,bx_b,br_b,bx_all,br_all,q,eta,
              first_cell,growth,title,save_path=None,show=True):
    Ni,Nj=X.shape
    fig=plt.figure(figsize=(22,16))
    gs=fig.add_gridspec(2,2,hspace=0.38,wspace=0.30)

    def pg(ax,X,R,si,sj,lw,al,mirror=False):
        ni_,nj_=X.shape
        for i in range(0,ni_,si):
            ax.plot(X[i,:],R[i,:],'b-',lw=lw,alpha=al)
            if mirror: ax.plot(X[i,:],-R[i,:],'b-',lw=lw,alpha=al)
        for j in range(0,nj_,sj):
            ax.plot(X[:,j],R[:,j],'b-',lw=lw,alpha=al)
            if mirror: ax.plot(X[:,j],-R[:,j],'b-',lw=lw,alpha=al)

    x_up=x_bg[0]; x_dn=x_bg[-1]; r_far=r_bg[-1]

    ax1=fig.add_subplot(gs[0,:])
    # BG grid
    for xi in x_bg[::max(1,len(x_bg)//60)]:
        ax1.plot([xi,xi],[-r_far,r_far],'c-',lw=0.15,alpha=0.3)
    for ri in r_bg[::2]:
        ax1.plot([x_up,x_dn],[ri,ri],'c-',lw=0.12,alpha=0.3)
        ax1.plot([x_up,x_dn],[-ri,-ri],'c-',lw=0.12,alpha=0.3)
    pg(ax1,X,R,max(1,Ni//10),max(1,Nj//80),0.22,0.7,mirror=True)
    ax1.fill_between(bx_all,br_all,-br_all,color='dimgray',alpha=0.9,label='Body')
    ax1.plot(bx_all,br_all,'r-',lw=1.5); ax1.plot(bx_all,-br_all,'r-',lw=1.5)
    ax1.axhline(0,color='k',lw=0.8,ls='--',alpha=0.4,label='Sym. axis')
    ax1.set_xlim(x_up,x_dn); ax1.set_ylim(-r_far*0.65,r_far*0.65)
    ax1.set_aspect('equal'); ax1.grid(False)
    ax1.set_xlabel('x [m]',fontsize=11); ax1.set_ylabel('r [m]',fontsize=11)
    ax1.set_title(f'Full domain  neg={q["neg"]}  inside={q["inside"]}  p99={q["p99"]:.0f}',fontsize=11)
    ax1.legend(fontsize=9)

    L=bx_b.max()-bx_b.min(); R_max=br_b.max()
    ax2=fig.add_subplot(gs[1,0])
    pg(ax2,X,R,max(1,Ni//12),max(1,Nj//100),0.3,0.7,False)
    ax2.fill_between(bx_all,0,br_all,color='dimgray',alpha=0.9,label='Body')
    ax2.plot(bx_all,br_all,'r-',lw=2)
    ax2.axhline(0,color='k',lw=0.8,ls='--',alpha=0.5)
    ax2.set_xlim(bx_b[0]-L*0.08,bx_b[-1]+L*0.1)
    ax2.set_ylim(-R_max*0.01,R_max*3.5)
    ax2.set_xlabel('x [m]',fontsize=10); ax2.set_ylabel('r [m]',fontsize=10)
    ax2.set_title(f'Inflation block ({Ni} layers, outer={eta[-1]:.4f}m)',fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True,alpha=0.15)

    ax3=fig.add_subplot(gs[1,1])
    x_mid=bx_b[0]+L*0.5; dxw=L*0.05
    r_bm=float(np.interp(x_mid,bx_b,br_b)); r_top=r_bm+eta[-1]*1.3
    j_mask=(X[0,:]>=x_mid-dxw)&(X[0,:]<=x_mid+dxw)
    if j_mask.sum()>0:
        pg(ax3,X[:,j_mask],R[:,j_mask],1,max(1,j_mask.sum()//60),0.5,0.9,False)
    bx_m=bx_b[(bx_b>=x_mid-dxw)&(bx_b<=x_mid+dxw)]
    br_m=br_b[(bx_b>=x_mid-dxw)&(bx_b<=x_mid+dxw)]
    ax3.fill_between(bx_m,0,br_m,color='dimgray',alpha=0.9,label='Body')
    ax3.plot(bx_m,br_m,'r-',lw=2)
    ax3.axhline(0,color='k',lw=0.8,ls='--',alpha=0.5)
    ax3.set_xlim(x_mid-dxw,x_mid+dxw); ax3.set_ylim(r_bm*0.87,r_top)
    ax3.set_title(f'ALL {Ni} inflation layers  first={first_cell:.1e}m  growth={growth}',fontsize=10)
    ax3.set_xlabel('x [m]',fontsize=10); ax3.set_ylabel('r [m]',fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(True,alpha=0.15)

    fig.suptitle(title,fontsize=12)
    if save_path:
        plt.savefig(save_path,dpi=150,bbox_inches='tight')
        print(f"[Plot]   -> {save_path}")
    if show: plt.show()
    plt.close()


def save_mesh(X,R,x_bg,r_bg,all_bx,all_br,bx_b,br_b,eta,params,path):
    np.savez_compressed(path,
        X_inf=X, R_inf=R,
        x_bg=x_bg, r_bg=r_bg,
        surface_x=all_bx, surface_r=all_br,
        body_x=bx_b, body_r=br_b, eta=eta,
        **{f"p_{k}":np.array(v) for k,v in params.items() if isinstance(v,(int,float,bool))})
    print(f"[Save]   -> {path}.npz")


def parse_args():
    p=argparse.ArgumentParser(description="Normal extrusion mesh v7")
    p.add_argument('--profile',    required=True)
    p.add_argument('--nj_body',    type=int,   default=600)
    p.add_argument('--nj_cap',     type=int,   default=40)
    p.add_argument('--n_inflate',  type=int,   default=40)
    p.add_argument('--first_cell', type=float, default=5e-5)
    p.add_argument('--growth',     type=float, default=1.15)
    p.add_argument('--r_far',      type=float, default=None)
    p.add_argument('--x_up_mult',  type=float, default=5.0)
    p.add_argument('--x_dn_mult',  type=float, default=7.0)
    p.add_argument('--supersonic', action='store_true')
    p.add_argument('--output',     default=None)
    return p.parse_args()


def main():
    args=parse_args()
    base=os.path.splitext(os.path.basename(args.profile))[0].replace('_profile','')
    outpfx=args.output or (base+'_mesh')

    print("="*55)
    print("  Normal Extrusion Mesh Generator  v7")
    print("="*55)
    print(f"  Profile    : {args.profile}")
    print(f"  n_inflate={args.n_inflate}  first={args.first_cell:.1e}m  growth={args.growth}")
    print(f"  nj_body={args.nj_body}  nj_cap={args.nj_cap}")
    print("="*55+"\n")

    bx_b,br_b,bx_all,br_all,L,R_max=load_profile(args.profile)

    x_up_m=10. if args.supersonic else args.x_up_mult
    x_up=bx_b[0]-x_up_m*L; x_dn=bx_b[-1]+args.x_dn_mult*L
    r_far=args.r_far or (12.*R_max)
    print(f"[Domain] x_up={x_up:.2f}m  x_dn={x_dn:.2f}m  r_far={r_far:.2f}m\n")

    all_bx,all_br,all_nx,all_nr = build_surface(bx_b,br_b,args.nj_body,args.nj_cap)
    eta = build_eta(args.first_cell,args.growth,args.n_inflate)
    X,R = extrude(all_bx,all_br,all_nx,all_nr,eta)
    x_bg,r_bg = build_background(bx_b,br_b,x_up,x_dn,r_far,eta[-1])
    print()
    q = check_quality(X,R,bx_b,br_b)
    print()

    params=dict(nj_body=args.nj_body,nj_cap=args.nj_cap,n_inflate=args.n_inflate,
                first_cell=args.first_cell,growth=args.growth,
                r_far=r_far,x_up=x_up,x_dn=x_dn,L=L,R_max=R_max)
    save_mesh(X,R,x_bg,r_bg,all_bx,all_br,bx_b,br_b,eta,params,outpfx)

    title=(f"Normal Extrusion v7: {os.path.basename(args.profile)}\n"
           f"Ni={len(eta)}xNj={len(all_bx)}={len(eta)*len(all_bx):,}  "
           f"neg={q['neg']}  inside={q['inside']}  p99={q['p99']:.0f}")
    plot_mesh(X,R,x_bg,r_bg,bx_b,br_b,bx_all,br_all,q,eta,
              args.first_cell,args.growth,title,
              save_path=outpfx+'_mesh.png',show=True)

    print(f"\n[Done]  Mesh->{outpfx}.npz  Plot->{outpfx}_mesh.png")


if __name__=='__main__':
    main()