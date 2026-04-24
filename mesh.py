"""
2D axisymmetric hybrid mesh generator.

Method:
  1. Build inflation layers directly with NumPy by offsetting profile points
     along smoothed outward normals.
  2. Fill the outer region with a Gmsh Delaunay triangular mesh.

Output NPZ fields:
  nodes      (N, 2)   all node coordinates [x, r]
  quad_cells (M, 4)   inflation-layer quads
  tri_cells  (K, 3)   outer triangular cells
  face_nodes (F, 2, 2) boundary segments as coordinates
  face_tag   (F,)     1=body 2=axis 3=inlet 4=outlet 5=farfield
  bl_first_h, bl_layers, bl_growth
"""

import argparse
import math
import os
import sys

import numpy as np

try:
    import gmsh
except ImportError:
    sys.exit("[ERROR] pip install gmsh")


def load_profile(path):
    """Load CSV profile, collapse duplicate x values, and enforce axis endpoints."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"Invalid profile CSV: {path}")

    raw = raw[:, :2]
    raw = raw[np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1])]
    if len(raw) < 2:
        raise ValueError(f"Profile has too few valid points: {path}")

    raw = raw[np.argsort(raw[:, 0])]
    raw[:, 1] = np.maximum(raw[:, 1], 0.0)

    collapsed = []
    i = 0
    while i < len(raw):
        j = i + 1
        while j < len(raw) and abs(raw[j, 0] - raw[i, 0]) < 1e-9:
            j += 1
        group = raw[i:j]
        zeros = group[group[:, 1] < 1e-10]
        collapsed.append(zeros[0] if len(zeros) else group[np.argmin(group[:, 1])])
        i = j

    pts = np.asarray(collapsed, dtype=np.float64)
    _, idx = np.unique(np.round(pts, 12), axis=0, return_index=True)
    pts = pts[np.sort(idx)]

    if pts[0, 1] > 1e-8:
        pts = np.vstack(([pts[0, 0], 0.0], pts))
        print("  [auto] added nose axis point")
    if pts[-1, 1] > 1e-8:
        pts = np.vstack((pts, [pts[-1, 0], 0.0]))
        print("  [auto] added tail axis point")

    step = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    keep = np.concatenate(([True], step > 1e-12))
    pts = pts[keep]
    if len(pts) < 3:
        raise ValueError("Profile collapsed to fewer than 3 unique points")

    print(
        f"  profile: {len(pts)} pt  "
        f"x=[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]  "
        f"R={pts[:,1].max():.4f} m"
    )
    print(
        f"  nose=({pts[0,0]:.5f},{pts[0,1]:.5f})  "
        f"tail=({pts[-1,0]:.5f},{pts[-1,1]:.5f})"
    )
    return pts


def domain_params(pts, mach):
    x0 = pts[:, 0].min()
    x1 = pts[:, 0].max()
    R = pts[:, 1].max()
    L = x1 - x0

    if mach >= 2.0:
        up = max(3.0 * L, 10.0 * R)
    elif mach >= 1.0:
        up = max(5.0 * L, 20.0 * R)
    else:
        up = max(12.0 * L, 60.0 * R)

    down = max(10.0 * L, 30.0 * R)
    rfar = max(80.0 * R, 5.0 * L)
    print(f"  domain: x=[{x0 - up:.1f},{x1 + down:.1f}]  r=[0,{rfar:.1f}] m")
    return {
        "x0": x0,
        "x1": x1,
        "R": R,
        "L": L,
        "x_in": x0 - up,
        "x_out": x1 + down,
        "r_far": rfar,
    }


def is_transonic(mach):
    return 0.8 <= mach <= 1.2


def inflation_params(pts, mach, h0_ratio=0.005, gr=1.1):
    R = pts[:, 1].max()
    rf = 1.0
    if is_transonic(mach):
        t = (mach - 1.0) / 0.2
        rf = 0.65 - 0.15 * 0.5 * (1.0 + math.cos(math.pi * t))

    h0 = h0_ratio * R * rf
    t_bl = R
    n = max(25, math.ceil(math.log(t_bl * (gr - 1.0) / h0 + 1.0) / math.log(gr)))
    t_actual = h0 * (gr**n - 1.0) / (gr - 1.0)
    print(
        f"  inflation: h0={h0:.5f}m ({h0 / R * 100:.3f}%R)  gr={gr}  "
        f"n={n}  t={t_actual:.4f}m ({t_actual / R * 100:.1f}%R)"
    )
    return {"h0": h0, "gr": gr, "n": n, "t": t_actual, "rf": rf}


def smooth_polyline(points, passes=20, weight_center=0.8):
    """Lightly smooth a polyline without moving endpoints."""
    if len(points) <= 2:
        return points

    points = points.copy()
    w_side = 0.5 * (1.0 - weight_center)
    for _ in range(passes):
        points[1:-1] = (
            w_side * points[:-2]
            + weight_center * points[1:-1]
            + w_side * points[2:]
        )
    return points


def build_inflation(pts, infl):
    """
    Create inflation layers by offsetting the body profile along smoothed normals.
    The method stays explicit and geometric; only implementation details are improved.
    """
    N = len(pts)
    h0 = infl["h0"]
    gr = infl["gr"]
    n = infl["n"]

    tx = np.gradient(pts[:, 0])
    tr = np.gradient(pts[:, 1])
    mag = np.sqrt(tx**2 + tr**2) + 1e-30
    nx = tr / mag
    nr = -tx / mag
    if nr.mean() < 0.0:
        nx = -nx
        nr = -nr

    x_nose = pts[:, 0].min()
    x_tail = pts[:, 0].max()
    zero_mask = pts[:, 1] < 1e-8
    nose_mask = zero_mask & (pts[:, 0] <= x_nose + 1e-6)
    tail_mask = zero_mask & (pts[:, 0] >= x_tail - 1e-6)
    mid_mask = zero_mask & ~nose_mask & ~tail_mask

    nx[nose_mask] = -1.0
    nr[nose_mask] = 0.0
    nx[tail_mask] = 1.0
    nr[tail_mask] = 0.0
    nx[mid_mask] = 0.0
    nr[mid_mask] = 1.0

    for _ in range(200):
        nx_s = nx.copy()
        nr_s = nr.copy()
        nx_s[1:-1] = 0.15 * nx[:-2] + 0.70 * nx[1:-1] + 0.15 * nx[2:]
        nr_s[1:-1] = 0.15 * nr[:-2] + 0.70 * nr[1:-1] + 0.15 * nr[2:]
        nx_s[nose_mask] = -1.0
        nr_s[nose_mask] = 0.0
        nx_s[tail_mask] = 1.0
        nr_s[tail_mask] = 0.0
        nx_s[mid_mask] = 0.0
        nr_s[mid_mask] = 1.0
        m = np.sqrt(nx_s**2 + nr_s**2) + 1e-30
        nx = nx_s / m
        nr = nr_s / m

    distances = h0 * (gr ** np.arange(1, n + 1) - 1.0) / (gr - 1.0)
    normals = np.stack((nx, nr), axis=1)

    layers = np.zeros((n + 1, N, 2), dtype=np.float64)
    layers[0] = pts
    for k, d in enumerate(distances, start=1):
        raw = pts + d * normals
        raw[:, 1] = np.maximum(raw[:, 1], 0.0)
        raw = smooth_polyline(raw, passes=20, weight_center=0.8)
        raw[:, 1] = np.maximum(raw[:, 1], 0.0)
        raw[0] = pts[0] + d * normals[0]
        raw[-1] = pts[-1] + d * normals[-1]
        raw[0, 1] = max(raw[0, 1], 0.0)
        raw[-1, 1] = max(raw[-1, 1], 0.0)
        layers[k] = raw

    print(f"  inflation layers built: {n} x {N} points")
    return layers, normals


def build_outer_mesh(pts, layers, dom, infl, h_far_ratio, out_msh, preview):
    """
    Generate the outer Delaunay mesh in Gmsh around the last inflation layer.
    """
    gmsh.initialize()
    gmsh.model.add("outer")
    geo = gmsh.model.geo

    R = dom["R"]
    x_in = dom["x_in"]
    x_out = dom["x_out"]
    r_far = dom["r_far"]
    x0 = dom["x0"]
    x1 = dom["x1"]

    h_far = h_far_ratio * R
    h_last = infl["h0"] * infl["gr"] ** (infl["n"] - 1)
    h_tr = h_last * 2.0

    outer_layer = layers[-1]
    N = len(outer_layer)
    outer_tags = [geo.addPoint(x, r, 0.0, meshSize=h_tr) for x, r in outer_layer]
    outer_curves = [geo.addLine(outer_tags[i], outer_tags[i + 1]) for i in range(N - 1)]

    p_in_ax = geo.addPoint(x_in, 0.0, 0.0, meshSize=h_far)
    p_out_ax = geo.addPoint(x_out, 0.0, 0.0, meshSize=h_far)
    p_in_top = geo.addPoint(x_in, r_far, 0.0, meshSize=h_far)
    p_out_top = geo.addPoint(x_out, r_far, 0.0, meshSize=h_far)
    p_nose_top = geo.addPoint(x0, r_far, 0.0, meshSize=h_far)
    p_base_top = geo.addPoint(x1, r_far, 0.0, meshSize=h_far)

    p_ol_nose = outer_tags[0]
    p_ol_base = outer_tags[-1]

    l_ax_in = geo.addLine(p_in_ax, p_ol_nose)
    l_ax_out = geo.addLine(p_ol_base, p_out_ax)
    l_inlet = geo.addLine(p_in_ax, p_in_top)
    l_outlet = geo.addLine(p_out_top, p_out_ax)
    l_far_1 = geo.addLine(p_in_top, p_nose_top)
    l_far_2 = geo.addLine(p_nose_top, p_base_top)
    l_far_3 = geo.addLine(p_base_top, p_out_top)
    l_v_nose = geo.addLine(p_ol_nose, p_nose_top)
    l_v_base = geo.addLine(p_ol_base, p_base_top)

    lp1 = geo.addCurveLoop([-l_ax_in, l_inlet, l_far_1, -l_v_nose])
    s1 = geo.addPlaneSurface([lp1])

    mid = [l_v_nose, l_far_2, -l_v_base, *[-c for c in reversed(outer_curves)]]
    lp2 = geo.addCurveLoop(mid)
    s2 = geo.addPlaneSurface([lp2])

    lp3 = geo.addCurveLoop([l_v_base, l_far_3, l_outlet, -l_ax_out])
    s3 = geo.addPlaneSurface([lp3])

    geo.synchronize()

    gmsh.model.addPhysicalGroup(1, outer_curves, tag=1, name="bl_outer")
    gmsh.model.addPhysicalGroup(1, [l_ax_in, l_ax_out], tag=2, name="axis")
    gmsh.model.addPhysicalGroup(1, [l_inlet], tag=3, name="inlet")
    gmsh.model.addPhysicalGroup(1, [l_outlet], tag=4, name="outlet")
    gmsh.model.addPhysicalGroup(1, [l_far_1, l_far_2, l_far_3], tag=5, name="farfield")
    gmsh.model.addPhysicalGroup(2, [s1, s2, s3], tag=10, name="fluid")

    h_mid = h_tr * 8.0
    h_far_eff = max(h_far, h_mid * 2.5)

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", outer_curves)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", max(200, min(2000, 2 * N)))

    f_thr1 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr1, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thr1, "SizeMin", h_tr)
    gmsh.model.mesh.field.setNumber(f_thr1, "SizeMax", h_mid)
    gmsh.model.mesh.field.setNumber(f_thr1, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thr1, "DistMax", R * 5.0)

    f_thr2 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr2, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thr2, "SizeMin", h_mid)
    gmsh.model.mesh.field.setNumber(f_thr2, "SizeMax", h_far_eff)
    gmsh.model.mesh.field.setNumber(f_thr2, "DistMin", R * 5.0)
    gmsh.model.mesh.field.setNumber(f_thr2, "DistMax", R * 40.0)

    f_box = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f_box, "VIn", h_mid)
    gmsh.model.mesh.field.setNumber(f_box, "VOut", h_far_eff)
    gmsh.model.mesh.field.setNumber(f_box, "XMin", dom["x0"] - 1.5 * dom["L"])
    gmsh.model.mesh.field.setNumber(f_box, "XMax", dom["x1"] + 2.5 * dom["L"])
    gmsh.model.mesh.field.setNumber(f_box, "YMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_box, "YMax", min(dom["r_far"], max(6.0 * R, 2.5 * infl["t"])))
    gmsh.model.mesh.field.setNumber(f_box, "Thickness", max(0.5 * R, infl["t"]))

    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thr1, f_thr2, f_box])
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("Mesh.Smoothing", 3)
    gmsh.option.setNumber("Mesh.MinimumCurveNodes", 5)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)

    print("  [Gmsh] generating outer Delaunay mesh...")
    print(f"         h_tr={h_tr:.5f}m  h_mid={h_mid:.4f}m  h_far={h_far_eff:.4f}m")
    gmsh.model.mesh.generate(2)

    node_tags, _, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
    n_elems = sum(len(tags) for tags in elem_tags)
    print(f"  [Gmsh] nodes={len(node_tags):,}  elems={n_elems:,}  types={elem_types}")

    gmsh.write(out_msh)
    if preview:
        gmsh.fltk.run()
    gmsh.finalize()


def assemble_npz(pts, layers, infl, msh_path, out_npz):
    """Combine inflation quads and outer Gmsh triangles into one NPZ dataset."""
    n_lay = infl["n"]
    N = len(pts)

    bl_nodes = layers.reshape(-1, 2)

    k_idx = np.arange(n_lay, dtype=np.int32)[:, None]
    i_idx = np.arange(N - 1, dtype=np.int32)[None, :]
    base = (k_idx * N + i_idx).reshape(-1)
    bl_cells = np.column_stack((base, base + 1, base + N + 1, base + N)).astype(np.int32)
    n_bl = len(bl_nodes)

    gmsh.initialize()
    gmsh.open(msh_path)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    t2i = {tag: i for i, tag in enumerate(node_tags)}
    ext_nodes = coords[:, :2].astype(np.float64)

    try:
        elem_tags, elem_nodes = gmsh.model.mesh.getElementsByType(2)
        del elem_tags
        tri_conn = elem_nodes.reshape(-1, 3)
        flat = np.fromiter((t2i[tag] for tag in tri_conn.ravel()), dtype=np.int32)
        tri_cells = flat.reshape(-1, 3)[:, ::-1]
    except Exception:
        tri_cells = np.empty((0, 3), dtype=np.int32)

    face_coords = []
    face_tags = []
    for ptag in (1, 2, 3, 4, 5):
        try:
            ents = gmsh.model.getEntitiesForPhysicalGroup(1, ptag)
        except Exception:
            continue
        for ent in ents:
            try:
                _, _, elem_node_sets = gmsh.model.mesh.getElements(1, ent)
            except Exception:
                continue
            for elem_nodes in elem_node_sets:
                pairs = elem_nodes.reshape(-1, 2)
                coords_pair = np.array(
                    [[ext_nodes[t2i[a]], ext_nodes[t2i[b]]] for a, b in pairs],
                    dtype=np.float64,
                )
                face_coords.append(coords_pair)
                face_tags.append(np.full(len(coords_pair), ptag, dtype=np.int8))

    gmsh.finalize()

    if len(tri_cells):
        tri_cells = tri_cells + n_bl

    all_nodes = np.vstack((bl_nodes, ext_nodes))

    if face_coords:
        face_coords = np.vstack(face_coords)
        face_tags = np.concatenate(face_tags)
    else:
        face_coords = np.empty((0, 2, 2), dtype=np.float64)
        face_tags = np.empty((0,), dtype=np.int8)

    body_arr = np.stack((bl_nodes[: N - 1], bl_nodes[1:N]), axis=1)
    body_tag = np.ones(len(body_arr), dtype=np.int8)
    face_coords = np.vstack((body_arr, face_coords))
    face_tags = np.concatenate((body_tag, face_tags))

    np.savez_compressed(
        out_npz,
        nodes=all_nodes.astype(np.float64),
        quad_cells=bl_cells,
        tri_cells=tri_cells,
        face_nodes=face_coords,
        face_tag=face_tags,
        bl_first_h=np.float64(infl["h0"]),
        bl_layers=np.int32(infl["n"]),
        bl_growth=np.float64(infl["gr"]),
    )
    print(f"  [NPZ] {out_npz}")
    print(
        f"        all nodes={len(all_nodes):,}  "
        f"quad(inflation)={len(bl_cells):,}  tri(outer)={len(tri_cells):,}"
    )
    tag_name = {1: "body", 2: "axis", 3: "inlet", 4: "outlet", 5: "farfield"}
    for tag, name in tag_name.items():
        print(f"        tag{tag}({name})={int((face_tags == tag).sum())}")


def visualize(npz_path, out_png):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.collections as mc
        import matplotlib.pyplot as plt
    except ImportError:
        print("  pip install matplotlib required")
        return

    data = np.load(npz_path, allow_pickle=False)
    nodes = data["nodes"]
    quads = data["quad_cells"]
    tris = data["tri_cells"]
    face_tag = data["face_tag"]
    face_nodes = data["face_nodes"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor="#0d1117")
    tag_color = {1: "#ff4c4c", 2: "#3a8fff", 3: "#44ff88", 4: "#ffaa00", 5: "#cc44ff"}
    tag_label = {1: "body", 2: "axis", 3: "inlet", 4: "outlet", 5: "farfield"}

    body_segments = face_nodes[face_tag == 1]
    bx = body_segments[:, :, 0]
    br = body_segments[:, :, 1]
    x0 = bx.min()
    x1 = bx.max()
    rmax = br.max()

    def draw(ax, xl, xr, rl, rr, maxc=60000):
        for cells, ncorner in ((quads, 4), (tris, 3)):
            if not len(cells):
                continue
            c0 = nodes[cells[:, 0]]
            mask = (c0[:, 0] >= xl) & (c0[:, 0] <= xr) & (c0[:, 1] >= rl) & (c0[:, 1] <= rr)
            sub = cells[mask]
            step = max(1, len(sub) // maxc)
            segs = []
            for cell in sub[::step]:
                for k in range(ncorner):
                    segs.append([nodes[cell[k]], nodes[cell[(k + 1) % ncorner]]])
            if segs:
                color = "#1e5a9a" if ncorner == 4 else "#2a6a3a"
                ax.add_collection(mc.LineCollection(segs, colors=color, linewidths=0.2, alpha=0.8))

    ax = axes[0]
    ax.set_facecolor("#0d1117")
    xl = x0 - rmax * 0.5
    xr = x1 + rmax * 0.5
    rl = 0.0
    rr = rmax * 4.0
    draw(ax, xl, xr, rl, rr)
    for tag in (2, 1):
        seg = face_nodes[face_tag == tag]
        in_range = ((seg[:, :, 0] >= xl) & (seg[:, :, 0] <= xr)).any(axis=1)
        if in_range.sum():
            ax.add_collection(
                mc.LineCollection(
                    seg[in_range],
                    colors=tag_color[tag],
                    linewidths=2.0 if tag == 1 else 1.0,
                    label=tag_label[tag],
                )
            )
    ax.set_xlim(xl, xr)
    ax.set_ylim(rl, rr)
    ax.set_aspect("equal")
    ax.set_title("Body + Inflation Layers", color="white", fontsize=12)
    ax.set_xlabel("x [m]", color="#aaa")
    ax.set_ylabel("r [m]", color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    ax = axes[1]
    ax.set_facecolor("#0d1117")
    zr = rmax * 0.8
    xl2 = x0 - zr * 0.15
    xr2 = x0 + zr * 1.5
    rl2 = 0.0
    rr2 = zr
    draw(ax, xl2, xr2, rl2, rr2, maxc=80000)
    for tag in (2, 1):
        seg = face_nodes[face_tag == tag]
        in_range = ((seg[:, :, 0] >= xl2) & (seg[:, :, 0] <= xr2)).any(axis=1)
        if in_range.sum():
            ax.add_collection(
                mc.LineCollection(
                    seg[in_range],
                    colors=tag_color[tag],
                    linewidths=2.0 if tag == 1 else 1.0,
                    label=tag_label[tag],
                )
            )
    ax.set_xlim(xl2, xr2)
    ax.set_ylim(rl2, rr2)
    ax.set_aspect("equal")
    ax.set_title("Nose Close-up (Inflation Layers)", color="white", fontsize=12)
    ax.set_xlabel("x [m]", color="#aaa")
    ax.set_ylabel("r [m]", color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [PNG] {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid mesh: NumPy inflation layers + Gmsh Delaunay outer mesh"
    )
    parser.add_argument("--profile", default="Falcon9_profile.csv")
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--h0ratio", type=float, default=0.005, help="h0 / R_ref")
    parser.add_argument("--farfield", type=float, default=0.35, help="farfield mesh size / R_ref")
    parser.add_argument("--out", default=None)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    workdir = os.path.dirname(os.path.abspath(__file__))
    if args.out is None:
        base_name = os.path.splitext(os.path.basename(args.profile))[0] + "_mesh"
        args.out = os.path.join(workdir, base_name)
    elif not os.path.isabs(args.out):
        args.out = os.path.join(workdir, args.out)

    print(f"\n{'=' * 52}")
    print(f"  Hybrid Mesh Generator  Mach={args.mach}  output:{args.out}")
    if is_transonic(args.mach):
        print("  transonic mode: first-layer height reduced automatically")
    print(f"{'=' * 52}\n")

    print("[1/5] Load profile")
    pts = load_profile(args.profile)

    print("\n[2/5] Domain and inflation parameters")
    dom = domain_params(pts, args.mach)
    infl = inflation_params(pts, args.mach, h0_ratio=args.h0ratio)

    print("\n[3/5] Build inflation layers (NumPy)")
    layers, _ = build_inflation(pts, infl)

    msh = args.out + ".msh"
    print("\n[4/5] Build outer Delaunay mesh (Gmsh)")
    build_outer_mesh(pts, layers, dom, infl, args.farfield, msh, args.preview)

    npz = args.out + ".npz"
    print("\n[5/5] Assemble NPZ and preview image")
    assemble_npz(pts, layers, infl, msh, npz)
    visualize(npz, args.out + "_preview.png")

    print("\nOutput files:")
    print(f"  {msh}")
    print(f"  {npz}")
    print(f"  {args.out}_preview.png")


if __name__ == "__main__":
    main()
