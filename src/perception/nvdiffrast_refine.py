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

    # Perturbed initial estimate.
    tx = torch.tensor(args.dx, device=dev, requires_grad=True)
    ty = torch.tensor(args.dy, device=dev, requires_grad=True)
    yaw = torch.tensor(yaw_true + math.radians(args.dyaw), device=dev, requires_grad=True)

    opt = torch.optim.Adam([tx, ty, yaw], lr=args.lr)
    print(f"\ninit error:  dx={args.dx:+.3f} dy={args.dy:+.3f} dyaw={args.dyaw:+.1f} deg")
    for it in range(args.iters):
        opt.zero_grad()
        d, m, n = render(tx, ty, yaw)
        inter = (m_t * m)                                   # where both have geometry
        depth_loss = (torch.abs(d - d_t) * inter).sum() / (inter.sum() + 1e-6)
        sil_loss = torch.mean((m - m_t) ** 2)               # silhouette alignment
        nrm_loss = ((1.0 - (n * n_t).sum(-1, keepdim=True)) * inter).sum() / (inter.sum() + 1e-6)
        loss = depth_loss + sil_loss + nrm_loss
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == args.iters - 1:
            print(f"  it {it:4d}  loss={loss.item():.5f}  "
                  f"depth={depth_loss.item():.5f} sil={sil_loss.item():.5f} nrm={nrm_loss.item():.5f}  "
                  f"| x={tx.item():+.3f} y={ty.item():+.3f} yaw={math.degrees(yaw.item()):.2f}")

    rec_yaw = math.degrees(yaw.item()) % 360.0
    tgt_yaw = math.degrees(yaw_true) % 360.0
    print(f"\nRESULT")
    print(f"  position: recovered ({tx.item():+.3f}, {ty.item():+.3f})  target (0.000, 0.000)  "
          f"err={math.hypot(tx.item(), ty.item()):.4f} m")
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
        d_r, _, _ = render(tx, ty, yaw)
    imageio.imwrite(out / "depth_target.png", depth_png(d_t))
    imageio.imwrite(out / "depth_recovered.png", depth_png(d_r))
    print(f"\nSaved depth_target.png / depth_recovered.png to {out}")


if __name__ == "__main__":
    main()
