"""
Learned pose corrector (DeepIM-style render-and-compare).

Trains a CNN that, given the GEOMETRIC buffers (depth + silhouette + normals) of
an observed object and of the current hypothesis, predicts the pose correction
(Δx, Δy, Δyaw) that maps the hypothesis toward the observation. Supervised with
GT during training; at inference it only sees the two buffer sets, so it works
GT-free. Applied iteratively (predict → apply → re-render → repeat).

Training data is generated on the fly from the asset DB + the differentiable
renderer (no pre-rendered scene bank needed): sample asset + GT pose +
perturbation, render both, supervise the correction = -perturbation.
"""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn

from src.perception.diffrender import DiffRenderer, load_normalized_mesh

# Perturbation ranges = the regime the corrector is trained to fix, and the
# normalization that puts each target component in ~[-1, 1].
POS_RANGE = 0.30          # meters
YAW_RANGE = math.radians(50.0)


class MeshBank:
    """Loads + normalizes DB assets onto the GPU for fast repeated rendering."""

    def __init__(self, db_path, assets_root, device="cuda", limit=None, target_size=1.0):
        import sqlite3
        con = sqlite3.connect(db_path); con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM objects").fetchall()
        self.meshes = []
        for r in rows:
            if limit and len(self.meshes) >= limit:
                break
            try:
                v, f, uv, tex = load_normalized_mesh(f"{assets_root}/{r['file_path']}", target_size)
                self.meshes.append((
                    torch.tensor(v, device=device),
                    torch.tensor(f, dtype=torch.int32, device=device),
                    torch.tensor(f, dtype=torch.int64, device=device),
                    torch.tensor(uv, device=device),
                    torch.tensor(tex, device=device).contiguous(),
                ))
            except Exception as e:
                print(f"  skip {r['id'][:8]}: {e}")
        if not self.meshes:
            raise RuntimeError("No meshes loaded")
        print(f"MeshBank: {len(self.meshes)} assets loaded")

    def __len__(self):
        return len(self.meshes)


def _buffers(renderer, mesh, x, y, yaw, device):
    """Render and stack [depth, mask, normals, rgb] -> (H,W,8)."""
    v0, fi, fl, uv, tex = mesh
    d, m, n, rgb = renderer.render(
        v0, fi, fl,
        torch.tensor(float(x), device=device),
        torch.tensor(float(y), device=device),
        torch.tensor(float(yaw), device=device),
        uv, tex,
    )
    return torch.cat([d[0], m[0], n[0], rgb[0]], dim=-1)  # (H,W,8)


def sample_batch(renderer, bank, B, device, rng):
    """Return (input (B,10,H,W), target (B,3) normalized, meta) for one batch."""
    obs_cur, targets = [], []
    for _ in range(B):
        mesh = bank.meshes[rng.randrange(len(bank))]
        gx, gy = rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35)
        gyaw = rng.uniform(0, 2 * math.pi)
        dx = rng.uniform(-POS_RANGE, POS_RANGE)
        dy = rng.uniform(-POS_RANGE, POS_RANGE)
        dyaw = rng.uniform(-YAW_RANGE, YAW_RANGE)
        with torch.no_grad():
            obs = _buffers(renderer, mesh, gx, gy, gyaw, device)
            cur = _buffers(renderer, mesh, gx + dx, gy + dy, gyaw + dyaw, device)
        obs_cur.append(torch.cat([obs, cur], dim=-1))                  # (H,W,10)
        targets.append([-dx / POS_RANGE, -dy / POS_RANGE, -dyaw / YAW_RANGE])
    x = torch.stack(obs_cur).permute(0, 3, 1, 2).contiguous()          # (B,10,H,W)
    t = torch.tensor(targets, device=device, dtype=torch.float32)
    return x, t


class PoseCorrector(nn.Module):
    def __init__(self, in_ch=16):  # depth+mask+normals+rgb (8) x {observed, current}
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.feat = nn.Sequential(
            blk(in_ch, 32), blk(32, 64), blk(64, 128), blk(128, 128), blk(128, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))

    def forward(self, x):
        return self.head(self.feat(x))


@torch.no_grad()
def evaluate(model, renderer, bank, device, rng, n=64, iters=5):
    """Iterative refinement on fresh samples; report mean pose error vs GT."""
    model.eval()
    init_pos, init_yaw, fin_pos, fin_yaw = [], [], [], []
    for _ in range(n):
        mesh = bank.meshes[rng.randrange(len(bank))]
        gx, gy = rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35)
        gyaw = rng.uniform(0, 2 * math.pi)
        cx = gx + rng.uniform(-POS_RANGE, POS_RANGE)
        cy = gy + rng.uniform(-POS_RANGE, POS_RANGE)
        cyaw = gyaw + rng.uniform(-YAW_RANGE, YAW_RANGE)
        init_pos.append(math.hypot(cx - gx, cy - gy))
        init_yaw.append(abs((math.degrees(cyaw - gyaw) + 180) % 360 - 180))
        obs = _buffers(renderer, mesh, gx, gy, gyaw, device)
        for _ in range(iters):
            cur = _buffers(renderer, mesh, cx, cy, cyaw, device)
            x = torch.cat([obs, cur], dim=-1).permute(2, 0, 1)[None].contiguous()
            p = model(x)[0]
            cx += float(p[0]) * POS_RANGE
            cy += float(p[1]) * POS_RANGE
            cyaw += float(p[2]) * YAW_RANGE
        fin_pos.append(math.hypot(cx - gx, cy - gy))
        fin_yaw.append(abs((math.degrees(cyaw - gyaw) + 180) % 360 - 180))
    model.train()
    return (np.mean(init_pos), np.mean(init_yaw), np.mean(fin_pos), np.mean(fin_yaw))


def main():
    ap = argparse.ArgumentParser(description="Train the learned pose corrector")
    ap.add_argument("--db", default="/data/database/assets.db")
    ap.add_argument("--assets", default="/data/assets")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-assets", type=int, default=None)
    ap.add_argument("--eval-iters", type=int, default=5)
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--out", default="/data/models/pose_corrector.pt")
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    rng = __import__("random").Random(0)

    renderer = DiffRenderer(res=args.res, device=dev)
    bank = MeshBank(args.db, args.assets, device=dev, limit=args.num_assets)

    model = PoseCorrector().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.MSELoss()

    print(f"\nTraining {args.steps} steps, batch {args.batch}, res {args.res}")
    run = 0.0
    for step in range(1, args.steps + 1):
        x, t = sample_batch(renderer, bank, args.batch, dev, rng)
        opt.zero_grad()
        loss = lossf(model(x), t)
        loss.backward()
        opt.step()
        run += loss.item()
        if step % 100 == 0:
            print(f"  step {step:5d}  loss={run / 100:.4f}")
            run = 0.0
        if step % 500 == 0:
            ip, iy, fp, fy = evaluate(model, renderer, bank, dev, rng,
                                      n=args.eval_n, iters=args.eval_iters)
            print(f"    [eval] init {ip:.3f}m/{iy:.1f}deg  ->  after {args.eval_iters} iters "
                  f"{fp:.3f}m/{fy:.1f}deg")

    torch.save(model.state_dict(), args.out)
    print(f"\nSaved corrector to {args.out}")
    ip, iy, fp, fy = evaluate(model, renderer, bank, dev, rng, n=256, iters=args.eval_iters)
    print(f"FINAL (256 samples): init {ip:.3f}m/{iy:.1f}deg  ->  "
          f"refined {fp:.3f}m/{fy:.1f}deg  (baseline GD was ~0.12m/12deg)")


if __name__ == "__main__":
    main()
