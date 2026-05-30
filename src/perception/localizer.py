"""
Analytic 3D localizer.

Given a 2D image point (later: a VLM detection; for now: a projected GT point),
recover the object's 3D world position using the scene's stored camera geometry.

Two methods:
  - ray_ground(): cast a ray through the pixel and intersect the KNOWN ground
    plane (z=0). Pure geometry, exact, no depth needed - the best estimate of
    "where on the ground the object sits", which is the task's objective.
  - backproject_depth(): use the SGBM depth map at the pixel. Needed for elevated
    / non-ground points and as a cross-check; noisier than ray-ground.

Conventions match renderer._scene_camera_gt: world->cam (R, t) in OpenCV frame
(x=right, y=down, z=forward); X_cam = R @ X_world + t.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class CameraGeometry:
    """Projection / back-projection for one camera (the left/depth-reference eye)."""

    def __init__(self, K, R, t):
        self.K = np.asarray(K, dtype=float)
        self.Kinv = np.linalg.inv(self.K)
        self.R = np.asarray(R, dtype=float)          # world -> cam
        self.t = np.asarray(t, dtype=float).reshape(3)
        self.C = -self.R.T @ self.t                  # camera center (world)

    @classmethod
    def from_gt(cls, gt_camera: Dict, eye: str = "left") -> "CameraGeometry":
        e = gt_camera[eye]
        return cls(gt_camera["intrinsics"], e["R_world_to_cam"], e["t_world_to_cam"])

    def project(self, xw) -> Tuple[np.ndarray, float]:
        """World point -> (pixel [u,v], depth z_cam). z_cam<=0 means behind camera."""
        xc = self.R @ np.asarray(xw, dtype=float) + self.t
        p = self.K @ xc
        return np.array([p[0] / p[2], p[1] / p[2]]), float(xc[2])

    def backproject_depth(self, u: float, v: float, z_cam: float) -> np.ndarray:
        """Pixel + camera-frame depth -> world point."""
        xc = z_cam * (self.Kinv @ np.array([u, v, 1.0]))
        return self.R.T @ (xc - self.t)

    def ray_ground(self, u: float, v: float, ground_z: float = 0.0) -> Optional[np.ndarray]:
        """Pixel -> intersection of its viewing ray with the plane z=ground_z."""
        d = self.R.T @ (self.Kinv @ np.array([u, v, 1.0]))   # ray direction (world)
        if abs(d[2]) < 1e-9:
            return None
        s = (ground_z - self.C[2]) / d[2]
        if s <= 0:
            return None
        return self.C + s * d


def localize_scene(gt: Dict, depth: Optional[np.ndarray] = None) -> Dict:
    """
    Validate localization against ground truth for one scene.

    For each GT object, project its ground-contact point to the left image (this
    stands in for a perfect detection), then recover the world position via
    ray-ground and (if a depth map is given) via depth back-projection. Reports
    per-object errors vs GT.
    """
    geom = CameraGeometry.from_gt(gt["camera"], "left")
    # Sanity: recovered camera center should equal the stored left eye position.
    c_err = float(np.linalg.norm(geom.C - np.array(gt["camera"]["left"]["position"])))

    res = []
    H = W = gt["camera"].get("resolution", 512)
    for obj in gt["objects"]:
        Xw = np.array(obj["position"], dtype=float)   # ground contact (x, y, 0)
        (u, v), z_cam = geom.project(Xw)

        rg = geom.ray_ground(u, v)
        rg_err = float(np.linalg.norm(rg[:2] - Xw[:2])) if rg is not None else None

        bp = bp_err = z_used = None
        if depth is not None:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                z_used = float(depth[vi, ui])
                if z_used > 0:
                    bp = geom.backproject_depth(u, v, z_used)
                    bp_err = float(np.linalg.norm(bp[:2] - Xw[:2]))

        res.append({
            "object_id": obj["object_id"],
            "gt_position": Xw.tolist(),
            "pixel": [round(float(u), 1), round(float(v), 1)],
            "z_cam_gt": round(z_cam, 3),
            "ray_ground": None if rg is None else [round(float(c), 3) for c in rg],
            "ray_ground_xy_err": None if rg_err is None else round(rg_err, 4),
            "depth_at_pixel": None if z_used is None else round(z_used, 3),
            "depth_backproj": None if bp is None else [round(float(c), 3) for c in bp],
            "depth_backproj_xy_err": None if bp_err is None else round(bp_err, 4),
        })

    return {"scene_id": gt.get("scene_id"), "camera_center_err": round(c_err, 6), "objects": res}


def main():
    parser = argparse.ArgumentParser(description="Validate the analytic localizer on a scene")
    parser.add_argument("--scene", type=str, default=None,
                        help="scene_id under /data/scenes (default: first found)")
    parser.add_argument("--scenes-dir", type=str, default="/data/scenes")
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir)
    if args.scene:
        scene_dir = scenes_dir / args.scene
    else:
        candidates = sorted(p.parent for p in scenes_dir.glob("*/ground_truth.json"))
        if not candidates:
            print(f"No scenes with ground_truth.json under {scenes_dir}")
            return
        scene_dir = candidates[0]

    gt = json.loads((scene_dir / "ground_truth.json").read_text())
    print(f"Scene: {gt.get('scene_id')}  ({len(gt['objects'])} objects)")

    # Compute (and cache) the SGBM depth map from the stereo pair.
    from src.stereo.disparity import DisparityEstimator
    cam = gt["camera"]
    est = DisparityEstimator()
    result = est.process_stereo_pair(
        gt["images"]["left"], gt["images"]["right"],
        {
            "focal_length": 50.0, "sensor_width": 36.0,
            "baseline": cam["baseline"], "resolution": cam["resolution"],
        },
    )
    depth = result["depth"]
    est.save_depth(depth, str(scene_dir / "depth.npy"))
    print(f"Depth: min={result['depth_min']:.3f}m max={result['depth_max']:.3f}m "
          f"(saved {scene_dir / 'depth.npy'})")

    report = localize_scene(gt, depth)
    print(f"Camera-center reconstruction error: {report['camera_center_err']:.2e} m (should be ~0)\n")
    print(f"{'object':10} {'GT (x,y)':>16} {'ray-ground (x,y)':>20} {'err(m)':>8} "
          f"{'depthZ':>8} {'bp err(m)':>10}")
    for o in report["objects"]:
        gx, gy = o["gt_position"][0], o["gt_position"][1]
        rg = o["ray_ground"]
        rgs = f"({rg[0]:.2f},{rg[1]:.2f})" if rg else "n/a"
        dz = o["depth_at_pixel"]
        print(f"{o['object_id'][:10]:10} ({gx:6.2f},{gy:6.2f})    {rgs:>20} "
              f"{o['ray_ground_xy_err'] if o['ray_ground_xy_err'] is not None else -1:8.4f} "
              f"{dz if dz is not None else -1:8.3f} "
              f"{o['depth_backproj_xy_err'] if o['depth_backproj_xy_err'] is not None else -1:10.4f}")


if __name__ == "__main__":
    main()
