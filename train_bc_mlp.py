"""Low-dim MLP behavior cloning for DMControl Reacher (qpos_match).

Input  : [qpos(2), qvel(2), goal_qpos(2)] = 6   (live arm state + target joints)
Output : action(2), tanh-bounded to [-1, 1]
Model  : 6 -> 256 -> 256 -> 2 MLP

Data cleaning (the key idea):
  Each CEM episode reaches the goal at some step then drifts away. We KEEP only
  steps 0..t* where t* = argmin over the rollout of angular distance to the goal
  (closest-approach), and DROP episodes that never get within TOL_DEG (success
  filter). This turns raw CEM rollouts into clean approach-only demonstrations,
  which is exactly what the success metric (reached-at-any-step) rewards.

Saves a single checkpoint bundling weights + input normalization stats so
eval_bc.py can reproduce the exact preprocessing.
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ---------------- config ----------------
DATA_DIR   = Path("/data/5pourghe/le_wm_storage/reacher")
PREFIX     = "xl_clean"          # matches collect_all.py output prefix
TOL_DEG    = 10.0                # success filter: drop eps not reaching within this
VAL_FRAC   = 0.15               # fraction of EPISODES held out for validation
HIDDEN     = 256
EPOCHS     = 300
BATCH      = 256
LR         = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE   = 15
SEED       = 42
OUT_PATH   = DATA_DIR / "weights" / "mlp_bc_policy.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


def wrap(d):
    """Shortest signed angular distance, radians in (-pi, pi]."""
    return (d + np.pi) % (2 * np.pi) - np.pi


def load_and_clean():
    """Load low-dim arrays, truncate each episode at closest-approach, filter.

    Returns X (N,6) float32, Y (N,2) float32, plus per-episode bookkeeping so
    the train/val split can be done at the EPISODE level (no leakage).
    """
    qpos      = np.squeeze(np.load(DATA_DIR / f"{PREFIX}_qpos.npy"),      axis=2)  # (T,E,2)
    qvel      = np.squeeze(np.load(DATA_DIR / f"{PREFIX}_qvel.npy"),      axis=2)  # (T,E,2)
    actions   = np.load(DATA_DIR / f"{PREFIX}_actions.npy")                        # (T,E,2)
    goal_qpos = np.squeeze(np.load(DATA_DIR / f"{PREFIX}_goal_qpos.npy"), axis=1)  # (E,2)
    T, E, _ = qpos.shape
    print(f"loaded: steps/ep={T}  episodes={E}")

    ang_err = np.abs(wrap(qpos - goal_qpos[None])).max(axis=2)   # (T,E) both joints align
    cut      = ang_err.argmin(axis=0)                            # (E,) closest-approach step
    best_ang = np.rad2deg(ang_err.min(axis=0))                   # (E,) deg at closest
    keep_ep  = np.nonzero(best_ang < TOL_DEG)[0]
    dropped  = E - len(keep_ep)
    print(f"success filter (<{TOL_DEG:.0f} deg): keep {len(keep_ep)}/{E} episodes "
          f"(dropped {dropped}); best-angle median {np.median(best_ang):.1f} deg")

    X_list, Y_list, ep_id_list = [], [], []
    for e in keep_ep:
        t_end = cut[e] + 1                       # keep steps 0..cut inclusive
        goal_b = np.broadcast_to(goal_qpos[e], (t_end, 2))
        feats  = np.concatenate([qpos[:t_end, e], qvel[:t_end, e], goal_b], axis=1)  # (t_end,6)
        X_list.append(feats.astype(np.float32))
        Y_list.append(actions[:t_end, e].astype(np.float32))                          # (t_end,2)
        ep_id_list.append(np.full(t_end, e, dtype=np.int64))

    X  = np.concatenate(X_list, axis=0)
    Y  = np.concatenate(Y_list, axis=0)
    ep = np.concatenate(ep_id_list, axis=0)
    print(f"training pairs: {len(X)}  (median cut-step "
          f"{int(np.median(cut[keep_ep]))}/{T})")
    return X, Y, ep, keep_ep


class MLP(nn.Module):
    def __init__(self, in_dim=6, hidden=HIDDEN, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)


def main():
    X, Y, ep, keep_ep = load_and_clean()

    # ---- episode-level train/val split (no leakage across the split) ----
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(keep_ep)
    n_val = max(1, int(len(shuffled) * VAL_FRAC))
    val_eps = set(shuffled[:n_val].tolist())
    is_val  = np.array([e in val_eps for e in ep])
    Xtr, Ytr = X[~is_val], Y[~is_val]
    Xva, Yva = X[is_val],  Y[is_val]
    print(f"split: {len(Xtr)} train pairs / {len(Xva)} val pairs "
          f"({len(shuffled)-n_val} / {n_val} episodes)")

    # ---- input normalization (fit on TRAIN only), save for eval ----
    in_mean = Xtr.mean(axis=0)
    in_std  = Xtr.std(axis=0) + 1e-6
    Xtr_n = (Xtr - in_mean) / in_std
    Xva_n = (Xva - in_mean) / in_std

    Xtr_t = torch.tensor(Xtr_n, device=device)
    Ytr_t = torch.tensor(Ytr,   device=device)
    Xva_t = torch.tensor(Xva_n, device=device)
    Yva_t = torch.tensor(Yva,   device=device)

    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    # naive predict-mean-action baseline for sanity (what mean-collapse would score)
    base = ((Ytr - Ytr.mean(0)) ** 2).mean()
    print(f"predict-mean-action baseline val-equivalent MSE ~ {base:.4f}\n")

    n = len(Xtr_t)
    best_val = float("inf")
    best_state = None
    bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            loss = loss_fn(pred, Ytr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        tr_loss = tot / n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t), Yva_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            flag = " *"
        else:
            bad += 1
            flag = ""
        if epoch % 5 == 0 or flag:
            print(f"epoch {epoch:3d}  train {tr_loss:.5f}  val {val_loss:.5f}{flag}")
        if bad >= PATIENCE:
            print(f"early stop at epoch {epoch} (no val improvement for {PATIENCE})")
            break

    # ---- save weights + normalization bundle ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "in_mean": in_mean.astype(np.float32),
        "in_std":  in_std.astype(np.float32),
        "arch": {"in_dim": 6, "hidden": HIDDEN, "out_dim": 2},
        "feature_order": ["qpos(2)", "qvel(2)", "goal_qpos(2)"],
        "tol_deg": TOL_DEG,
        "best_val_mse": best_val,
    }, OUT_PATH)
    print(f"\nbest val MSE {best_val:.5f}  ->  saved {OUT_PATH}")


if __name__ == "__main__":
    main()
