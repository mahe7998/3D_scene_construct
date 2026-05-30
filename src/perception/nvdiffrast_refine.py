"""
nvdiffrast render-and-compare prototype.

Proves the core of the refinement idea on ONE object: render a target view's
GEOMETRIC buffers (depth + silhouette + normals) at a known pose, perturb the
pose (dx, dy, dyaw), then recover it by gradient descent through the
differentiable rasterizer on a geometry-only loss (no RGB).

Level-1 DOF: object rests on the ground, so we optimize (x, y, yaw).
Self-contained: torch + nvdiffrast + trimesh only (no cv2/perception __init__).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr


# ----------------------------------------------------------------------------- camera math (OpenGL convention, Z-up world)
def look_at(eye, center, up):
    eye, center, up = map(lambda a: np.asarray(a, np.float32), (eye, center, up))
    f = center - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def perspective(fovy_rad, aspect, near, far):
    t = math.tan(fovy_rad / 2.0)
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = 1.0 / (aspect * t)
    P[1, 1] = 1.0 / t
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def eye_from_spherical(distance, azim_deg, elev_deg, look_at_pt):
    a, e = math.radians(azim_deg), math.radians(elev_deg)
    x = distance * math.cos(e) * math.sin(a)
    y = -distance * math.cos(e) * math.cos(a)
    z = distance * math.sin(e)
    return np.array([x + look_at_pt[0], y + look_at_pt[1], z + look_at_pt[2]], np.float32)


# ----------------------------------------------------------------------------- mesh
def load_normalized_mesh(path, target_size=1.0):
    """Load glb -> centered in xy, bottom on z=0, longest dim == target_size."""
    m = trimesh.load(path, force="mesh")
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.faces, np.int64)
    lo, hi = v.min(0), v.max(0)
    dims = hi - lo
    s = target_size / float(dims.max())
    v[:, 0] -= (lo[0] + hi[0]) / 2.0
    v[:, 1] -= (lo[1] + hi[1]) / 2.0
    v[:, 2] -= lo[2]
    v *= s
    return v, f


def vertex_normals(verts, faces_i64):
    v0 = verts[faces_i64[:, 0]]; v1 = verts[faces_i64[:, 1]]; v2 = verts[faces_i64[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)  # area-weighted face normals
    vn = torch.zeros_like(verts)
    for k in range(3):
        vn.index_add_(0, faces_i64[:, k], fn)
    return vn / (vn.norm(dim=1, keepdim=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser(description="nvdiffrast pose-recovery prototype")
    ap.add_argument("--object-id", default="fc1339e225b7408caec82681be2746c5")
    ap.add_argument("--assets", default="/data/assets")
    ap.add_argument("--db", default="/data/database/assets.db")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--dx", type=float, default=0.25)
    ap.add_argument("--dy", type=float, default=-0.20)
    ap.add_argument("--dyaw", type=float, default=35.0, help="perturbation in degrees")
    ap.add_argument("--out", default="/data/scenes/refine_demo")
    args = ap.parse_args()

    dev = "cuda"
    H = W = args.res

    # Resolve the asset path from the DB (fall back to first object).
    import sqlite3
    con = sqlite3.connect(args.db); con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM objects WHERE id=?", (args.object_id,)).fetchone()
    if row is None:
        row = con.execute("SELECT * FROM objects LIMIT 1").fetchone()
    fp = Path(args.assets) / row["file_path"]
    print(f"Object: {row['id']} ({row['category']})  ->  {fp}")

    v_np, f_np = load_normalized_mesh(str(fp))
    print(f"Mesh: {v_np.shape[0]} verts, {f_np.shape[0]} faces")
    verts0 = torch.tensor(v_np, device=dev)              # canonical (V,3)
    faces_i = torch.tensor(f_np, dtype=torch.int32, device=dev)
    faces_l = torch.tensor(f_np, dtype=torch.int64, device=dev)

    # Camera: matches our stereo intrinsics (50mm / 36mm sensor -> ~39.6deg vfov).
    look_pt = np.array([0, 0, 0.5], np.float32)
    eye = eye_from_spherical(distance=4.0, azim_deg=35.0, elev_deg=25.0, look_at_pt=look_pt)
    fovy = 2.0 * math.atan((36.0 / 2.0) / 50.0)
    V = torch.tensor(look_at(eye, look_pt, [0, 0, 1]), device=dev)
    P = torch.tensor(perspective(fovy, 1.0, 0.1, 100.0), device=dev)
    PV = P @ V

    glctx = dr.RasterizeCudaContext()

    def render(tx, ty, yaw):
        c, s = torch.cos(yaw), torch.sin(yaw)
        Rz = torch.stack([
            torch.stack([c, -s, torch.zeros_like(c)]),
            torch.stack([s,  c, torch.zeros_like(c)]),
            torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)]),
        ])
        vw = verts0 @ Rz.T + torch.stack([tx, ty, torch.zeros_like(tx)])   # posed world verts
        vh = torch.cat([vw, torch.ones_like(vw[:, :1])], dim=1)            # (V,4)
        clip = (vh @ PV.T).contiguous()
        rast, _ = dr.rasterize(glctx, clip[None], faces_i, resolution=[H, W])
        mask = (rast[..., 3:4] > 0).float()
        mask_aa = dr.antialias(mask, rast, clip[None], faces_i)
        vview = (vh @ V.T)[:, 2:3]                                         # view-space z (neg in front)
        depth, _ = dr.interpolate((-vview)[None], rast, faces_i)           # positive distance
        nrm = vertex_normals(vw, faces_l)
        normals, _ = dr.interpolate(nrm[None], rast, faces_i)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
        return depth, mask_aa, normals

    # Target at the known pose.
    yaw_true = math.radians(20.0)
    with torch.no_grad():
        d_t, m_t, n_t = render(torch.tensor(0.0, device=dev),
                               torch.tensor(0.0, device=dev),
                               torch.tensor(yaw_true, device=dev))
    d_t, m_t, n_t = d_t.detach(), m_t.detach(), n_t.detach()

    # Separable Gaussian blur on NHWC buffers - scale-space continuation widens
    # the convergence basin (a blurred silhouette/depth has gradient support far
    # from the true pose, instead of a razor-thin edge).
    def gaussian_blur(x, sigma):
        if sigma <= 0:
            return x
        rad = max(1, int(3 * sigma))
        xs = torch.arange(-rad, rad + 1, device=x.device, dtype=torch.float32)
        k = torch.exp(-(xs ** 2) / (2 * sigma * sigma)); k = k / k.sum()
        xc = x.permute(0, 3, 1, 2)
        C = xc.shape[1]
        kh = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        kw = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        xc = torch.nn.functional.conv2d(xc, kh, padding=(rad, 0), groups=C)
        xc = torch.nn.functional.conv2d(xc, kw, padding=(0, rad), groups=C)
        return xc.permute(0, 2, 3, 1)

    def geom_loss(tx, ty, yaw, sigma, dw, wn):
        """Geometric loss at blur scale `sigma` with depth/normal weights dw/wn."""
        d, m, n = render(tx, ty, yaw)
        mb, mtb = gaussian_blur(m, sigma), gaussian_blur(m_t, sigma)
        sil = torch.mean((mb - mtb) ** 2)
        inter = mtb * mb
        depth = torch.zeros((), device=dev)
        nrm = torch.zeros((), device=dev)
        if dw > 0:
            db, dtb = gaussian_blur(d, sigma), gaussian_blur(d_t, sigma)
            depth = (torch.abs(db - dtb) * inter).sum() / (inter.sum() + 1e-6)
        if wn > 0:
            nb, ntb = gaussian_blur(n, sigma), gaussian_blur(n_t, sigma)
            nrm = ((1.0 - (nb * ntb).sum(-1, keepdim=True)) * inter).sum() / (inter.sum() + 1e-6)
        return SIL_W * sil + dw * depth + wn * nrm

    # Coarse-to-fine schedule: (blur sigma px, depth weight, normal weight, lr).
    # The COARSE phase is silhouette-ONLY (depth/normals off): silhouette MSE over
    # the whole image rises when the object moves away, so it can't "escape" the
    # way intersection-normalized depth let it earlier. Depth then normals come in
    # as it locks on; lr decays so it settles instead of dithering.
    phases = [
        (6.0, 0.0, 0.0, args.lr),
        (3.0, 1.0, 0.0, args.lr * 0.5),
        (1.5, 1.0, 0.5, args.lr * 0.25),
        (0.0, 1.0, 1.0, args.lr * 0.1),
    ]
    SIL_W = 20.0  # keep silhouette dominant throughout so the object can't run off
    iters_per = max(1, args.iters // len(phases))

    def refine(tx0, ty0, yaw0):
        """Coarse-to-fine refine from one init; return (x, y, yaw, final_score)."""
        tx = torch.tensor(tx0, device=dev, requires_grad=True)
        ty = torch.tensor(ty0, device=dev, requires_grad=True)
        yaw = torch.tensor(yaw0, device=dev, requires_grad=True)
        opt = torch.optim.Adam([tx, ty, yaw], lr=args.lr)
        for sigma, dw, wn, lr in phases:
            for g in opt.param_groups:
                g["lr"] = lr
            for _ in range(iters_per):
                opt.zero_grad()
                loss = geom_loss(tx, ty, yaw, sigma, dw, wn)
                loss.backward()
                opt.step()
        # Score each start at full resolution with all terms, so they're comparable.
        with torch.no_grad():
            score = geom_loss(tx, ty, yaw, 0.0, 1.0, 1.0).item()
        return tx.item(), ty.item(), yaw.item(), score

    # Multi-start over yaw (the ambiguous DOF): refine from headings around the
    # full circle (same translation estimate) and keep the lowest-loss result -
    # the standard fix for rotation's narrow basin / local minima.
    starts = [math.radians(a) for a in range(0, 360, 45)]
    print(f"\ninit translation estimate ({args.dx:+.3f}, {args.dy:+.3f}); "
          f"{len(starts)} yaw starts every 45 deg")
    best = None
    for yc in starts:
        rx, ry, ryaw, score = refine(args.dx, args.dy, yc)
        print(f"  start yaw={math.degrees(yc):6.1f}  ->  x={rx:+.3f} y={ry:+.3f} "
              f"yaw={math.degrees(ryaw) % 360:6.2f}  score={score:.5f}")
        if best is None or score < best[3]:
            best = (rx, ry, ryaw, score)

    rx, ry, ryaw, score = best
    rec_yaw = math.degrees(ryaw) % 360.0
    tgt_yaw = math.degrees(yaw_true) % 360.0
    print(f"\nRESULT (best of {len(starts)} starts, score={score:.5f})")
    print(f"  position: recovered ({rx:+.3f}, {ry:+.3f})  target (0.000, 0.000)  "
          f"err={math.hypot(rx, ry):.4f} m")
    print(f"  yaw:      recovered {rec_yaw:.2f} deg  target {tgt_yaw:.2f} deg  "
          f"err={abs((rec_yaw - tgt_yaw + 180) % 360 - 180):.2f} deg")

    # Save target vs recovered depth for visual confirmation.
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    def depth_png(d):
        a = d[0, ..., 0].detach().cpu().numpy()
        v = a[a > 0]
        if v.size:
            a = np.clip((a - v.min()) / (v.max() - v.min() + 1e-6), 0, 1)
        return (a * 255).astype(np.uint8)

    with torch.no_grad():
        d_r, _, _ = render(torch.tensor(rx, device=dev),
                           torch.tensor(ry, device=dev),
                           torch.tensor(ryaw, device=dev))
    imageio.imwrite(out / "depth_target.png", depth_png(d_t))
    imageio.imwrite(out / "depth_recovered.png", depth_png(d_r))
    print(f"\nSaved depth_target.png / depth_recovered.png to {out}")


if __name__ == "__main__":
    main()
