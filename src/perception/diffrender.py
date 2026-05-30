"""
Shared nvdiffrast geometric renderer.

Renders appearance-invariant buffers (depth, silhouette, surface normals) for a
mesh at a given ground pose (x, y, yaw) under a fixed camera. Used to generate
training data and to run the render-and-compare loop. No RGB - proxies match
real objects only at the category/shape level, so comparison is geometric.
"""

import math
from pathlib import Path

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr


# ----------------------------------------------------------------------------- camera (OpenGL convention, Z-up world)
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
    """Load glb -> centered in xy, bottom on z=0, longest dim == target_size.

    Returns (verts, faces, uv, tex): uv (V,2) and an RGB texture (Ht,Wt,3) in
    [0,1] for the appearance/yaw cue. Falls back to a flat 2x2 base-color texture
    (uv=0) when an asset has no baseColorTexture - those just render a solid color.
    """
    m = trimesh.load(path, force="mesh")
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.faces, np.int64)
    lo, hi = v.min(0), v.max(0)
    s = target_size / float((hi - lo).max())
    v[:, 0] -= (lo[0] + hi[0]) / 2.0
    v[:, 1] -= (lo[1] + hi[1]) / 2.0
    v[:, 2] -= lo[2]
    v *= s

    vis = m.visual
    uv = None
    try:
        if getattr(vis, "uv", None) is not None and len(vis.uv) == len(v):
            uv = np.asarray(vis.uv, np.float32)
    except Exception:
        uv = None
    img = None
    try:
        img = vis.material.baseColorTexture
    except Exception:
        img = None
    if uv is not None and img is not None:
        tex = np.asarray(img.convert("RGB"), np.float32) / 255.0
    else:
        col = np.array([0.6, 0.6, 0.6], np.float32)
        try:
            bc = vis.material.baseColorFactor
            if bc is not None:
                bc = np.asarray(bc[:3], np.float32)
                col = bc / 255.0 if bc.max() > 1.0 else bc
        except Exception:
            pass
        tex = np.tile(col.reshape(1, 1, 3), (2, 2, 1)).astype(np.float32)
        uv = np.zeros((len(v), 2), np.float32)
    return v, f, uv, tex


def vertex_normals(verts, faces_l):
    v0 = verts[faces_l[:, 0]]; v1 = verts[faces_l[:, 1]]; v2 = verts[faces_l[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    vn = torch.zeros_like(verts)
    for k in range(3):
        vn.index_add_(0, faces_l[:, k], fn)
    return vn / (vn.norm(dim=1, keepdim=True) + 1e-8)


class DiffRenderer:
    """Differentiable geometric renderer with a fixed camera."""

    def __init__(self, res=128, device="cuda",
                 distance=4.0, azim=35.0, elev=25.0, look_h=0.5,
                 focal_mm=50.0, sensor_mm=36.0):
        self.res = res
        self.device = device
        look_pt = np.array([0, 0, look_h], np.float32)
        eye = eye_from_spherical(distance, azim, elev, look_pt)
        fovy = 2.0 * math.atan((sensor_mm / 2.0) / focal_mm)
        self.V = torch.tensor(look_at(eye, look_pt, [0, 0, 1]), device=device)
        P = torch.tensor(perspective(fovy, 1.0, 0.1, 100.0), device=device)
        self.PV = P @ self.V
        self.glctx = dr.RasterizeCudaContext()

    def render(self, verts0, faces_i, faces_l, tx, ty, yaw, uv=None, tex=None):
        """verts0: canonical (V,3). tx,ty,yaw: scalar tensors.
        Returns depth, mask, normals, rgb (each (1,H,W,C)); rgb is zeros if no uv/tex."""
        c, s = torch.cos(yaw), torch.sin(yaw)
        z0, o1 = torch.zeros_like(c), torch.ones_like(c)
        Rz = torch.stack([
            torch.stack([c, -s, z0]),
            torch.stack([s, c, z0]),
            torch.stack([z0, z0, o1]),
        ])
        vw = verts0 @ Rz.T + torch.stack([tx, ty, torch.zeros_like(tx)])
        vh = torch.cat([vw, torch.ones_like(vw[:, :1])], dim=1)
        clip = (vh @ self.PV.T).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip[None], faces_i, resolution=[self.res, self.res])
        mask = (rast[..., 3:4] > 0).float()
        mask = dr.antialias(mask, rast, clip[None], faces_i)
        depth, _ = dr.interpolate((-(vh @ self.V.T)[:, 2:3])[None], rast, faces_i)
        nrm = vertex_normals(vw, faces_l)
        normals, _ = dr.interpolate(nrm[None], rast, faces_i)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
        if uv is not None and tex is not None:
            uv_pix, _ = dr.interpolate(uv[None], rast, faces_i)
            rgb = dr.texture(tex[None], uv_pix, filter_mode="linear") * mask
        else:
            rgb = torch.zeros_like(normals)
        return depth, mask, normals, rgb
