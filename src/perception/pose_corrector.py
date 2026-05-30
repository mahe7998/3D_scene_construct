"""
Learned pose corrector (DeepIM-style render-and-compare).

Trains a CNN that, given the GEOMETRIC buffers (depth + silhouette + normals)
plus an RGB appearance channel of an observed object and of the current
hypothesis, predicts the pose correction (Δx, Δy, Δyaw) that maps the hypothesis
toward the observation. Supervised with GT during training; at inference it only
sees the two buffer sets, so it works GT-free. Applied iteratively
(predict → apply → re-render → repeat).

Position is a continuous regression. Yaw is a CLASSIFICATION over bins of the
Δyaw correction + a within-bin residual regression (PoseCNN/DeepIM-style), NOT a
single MSE regression: single-view geometry is heading-ambiguous, so a plain
regression collapses toward the mean when two yaws look alike. Softmax-over-bins
picks one mode (argmax) instead of averaging, and the residual recovers sub-bin
precision.

Training data is generated on the fly from the asset DB + the differentiable
renderer (no pre-rendered scene bank needed): sample asset + GT pose +
perturbation, render both, supervise the correction = -perturbation.
"""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.perception.diffrender import DiffRenderer, load_normalized_mesh

# Perturbation ranges = the regime the corrector is trained to fix, and the
# normalization that puts the position target in ~[-1, 1].
POS_RANGE = 0.30          # meters
YAW_RANGE = math.radians(50.0)

# Yaw is binned over the correction range [-YAW_RANGE, +YAW_RANGE]. K bins of
# width BIN_W; the residual head refines within a bin. K=20 -> 5deg bins, so the
# residual only has to resolve <2.5deg.
YAW_BINS = 20
BIN_W = 2.0 * YAW_RANGE / YAW_BINS
# Loss weights (tunable): position MSE keeps full weight; yaw cls/residual are
# scaled down so the cross-entropy magnitude (~ln K) doesn't swamp the backbone.
W_CLS = 0.5
W_RES = 0.5


def yaw_to_bin_res(dyaw):
    """Map a Δyaw correction (B,) in radians -> (bin index (B,), residual (B,)).

    Residual is normalized to ~[-1, 1] = offset from the bin center / (BIN_W/2).
    """
    t = dyaw.clamp(-YAW_RANGE, YAW_RANGE - 1e-6)
    b = ((t + YAW_RANGE) / BIN_W).floor().long().clamp(0, YAW_BINS - 1)
    center = -YAW_RANGE + (b.float() + 0.5) * BIN_W
    res = (t - center) / (BIN_W / 2.0)
    return b, res


def decode_yaw(logits, res):
    """argmax bin + its residual -> Δyaw correction (B,) in radians."""
    b = logits.argmax(dim=1)
    center = -YAW_RANGE + (b.float() + 0.5) * BIN_W
    r = res.gather(1, b[:, None]).squeeze(1) * (BIN_W / 2.0)
    return center + r


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
    """Return (input (B,16,H,W), pos_target (B,2), dyaw_correction (B,)) for a batch.

    pos_target is normalized by POS_RANGE; dyaw_correction is the raw correction
    in radians (= -perturbation), binned inside the loss.
    """
    obs_cur, pos_t, yaw_t = [], [], []
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
        obs_cur.append(torch.cat([obs, cur], dim=-1))                  # (H,W,16)
        pos_t.append([-dx / POS_RANGE, -dy / POS_RANGE])
        yaw_t.append(-dyaw)
    x = torch.stack(obs_cur).permute(0, 3, 1, 2).contiguous()          # (B,16,H,W)
    pos = torch.tensor(pos_t, device=device, dtype=torch.float32)
    yaw = torch.tensor(yaw_t, device=device, dtype=torch.float32)
    return x, pos, yaw


class PoseCorrector(nn.Module):
    """Shared CNN backbone -> position regression + yaw (classification + residual).

    forward returns (pos (B,2), yaw_logits (B,K), yaw_res (B,K)). Decode the yaw
    correction with decode_yaw(logits, res).
    """

    def __init__(self, in_ch=16):  # depth+mask+normals+rgb (8) x {observed, current}
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.feat = nn.Sequential(
            blk(in_ch, 32), blk(32, 64), blk(64, 128), blk(128, 128), blk(128, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.trunk = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.pos_head = nn.Linear(128, 2)
        self.yaw_cls = nn.Linear(128, YAW_BINS)
        self.yaw_res = nn.Linear(128, YAW_BINS)

    def forward(self, x):
        h = self.trunk(self.feat(x))
        return self.pos_head(h), self.yaw_cls(h), self.yaw_res(h)


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
            pos, logits, res = model(x)
            cx += float(pos[0, 0]) * POS_RANGE
            cy += float(pos[0, 1]) * POS_RANGE
            cyaw += float(decode_yaw(logits, res)[0])
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
    ap.add_argument("--logdir", default="/data/logs/pose_corrector",
                    help="TensorBoard base dir; each run goes in <logdir>/<run-name>")
    ap.add_argument("--run-name", default="binned",
                    help="Subdir name so runs show as separate curves in TensorBoard")
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    rng = __import__("random").Random(0)

    renderer = DiffRenderer(res=args.res, device=dev)
    bank = MeshBank(args.db, args.assets, device=dev, limit=args.num_assets)

    model = PoseCorrector().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = None
    logdir = f"{args.logdir}/{args.run_name}"
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(logdir)
        print(f"TensorBoard logging -> {logdir}")
    except Exception as e:  # tensorboard not installed -> run without it, don't crash
        print(f"TensorBoard logging disabled ({e}); rebuild the refine image to enable.")

    print(f"\nTraining {args.steps} steps, batch {args.batch}, res {args.res}, "
          f"yaw bins {YAW_BINS} ({math.degrees(BIN_W):.1f}deg each)")
    run = 0.0
    for step in range(1, args.steps + 1):
        x, pos_t, yaw_t = sample_batch(renderer, bank, args.batch, dev, rng)
        gt_bin, gt_res = yaw_to_bin_res(yaw_t)
        opt.zero_grad()
        pos, logits, res = model(x)
        pos_loss = F.mse_loss(pos, pos_t)
        cls_loss = F.cross_entropy(logits, gt_bin)
        res_pred = res.gather(1, gt_bin[:, None]).squeeze(1)  # residual of the GT bin
        res_loss = F.smooth_l1_loss(res_pred, gt_res)
        loss = pos_loss + W_CLS * cls_loss + W_RES * res_loss
        loss.backward()
        opt.step()
        run += loss.item()
        if writer is not None:
            writer.add_scalar("train/loss_total", loss.item(), step)
            writer.add_scalar("train/loss_pos", pos_loss.item(), step)
            writer.add_scalar("train/loss_yaw_cls", cls_loss.item(), step)
            writer.add_scalar("train/loss_yaw_res", res_loss.item(), step)
            writer.add_scalar("train/yaw_bin_acc", (logits.argmax(1) == gt_bin).float().mean().item(), step)
        if step % 100 == 0:
            print(f"  step {step:5d}  loss={run / 100:.4f}")
            run = 0.0
        if step % 500 == 0:
            ip, iy, fp, fy = evaluate(model, renderer, bank, dev, rng,
                                      n=args.eval_n, iters=args.eval_iters)
            print(f"    [eval] init {ip:.3f}m/{iy:.1f}deg  ->  after {args.eval_iters} iters "
                  f"{fp:.3f}m/{fy:.1f}deg")
            if writer is not None:
                writer.add_scalar("eval/init_pos_m", ip, step)
                writer.add_scalar("eval/init_yaw_deg", iy, step)
                writer.add_scalar("eval/refined_pos_m", fp, step)
                writer.add_scalar("eval/refined_yaw_deg", fy, step)

    torch.save(model.state_dict(), args.out)
    print(f"\nSaved corrector to {args.out}")
    ip, iy, fp, fy = evaluate(model, renderer, bank, dev, rng, n=256, iters=args.eval_iters)
    print(f"FINAL (256 samples): init {ip:.3f}m/{iy:.1f}deg  ->  "
          f"refined {fp:.3f}m/{fy:.1f}deg  (baseline GD was ~0.12m/12deg)")
    if writer is not None:
        writer.add_scalar("eval/final_refined_pos_m", fp, args.steps)
        writer.add_scalar("eval/final_refined_yaw_deg", fy, args.steps)
        writer.close()


if __name__ == "__main__":
    main()
