"""Evaluate the low-dim MLP behavior-cloning policy on Reacher (qpos_match).

Mirrors eval_bc.py's harness but bypasses the planner with an MLP that reads
[qpos, qvel, goal_qpos] straight from info_dict — the SAME features and
normalization train_bc_mlp.py used. Run with the reacher eval config, e.g.:

  python eval_bc_mlp.py --config-name reacher
"""
import os
os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import stable_worldmodel as swm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLP_PATH = Path("/data/5pourghe/le_wm_storage/reacher/weights/mlp_bc_policy.pth")


class MLP(nn.Module):
    def __init__(self, in_dim=6, hidden=256, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == e]) + 1 for e in episodes])


@hydra.main(version_base=None, config_path="./config/eval", config_name="reacher")
def run(cfg: DictConfig):
    # ---- world ----
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # ---- dataset (for start states / goals / callables) ----
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name, keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    # ---- valid start points (same logic as collection/eval) ----
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {e: max_start_idx[i] for i, e in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[e] for e in dataset.get_col_data(col_name)])
    valid_indices = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]
    print(len(valid_indices), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = np.sort(valid_indices[
        g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    # ---- load MLP + normalization bundle ----
    ckpt = torch.load(MLP_PATH, map_location=device)
    arch = ckpt.get("arch", {"in_dim": 6, "hidden": 256, "out_dim": 2})
    model = MLP(**arch).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    in_mean = torch.tensor(ckpt["in_mean"], device=device).view(1, -1)
    in_std  = torch.tensor(ckpt["in_std"],  device=device).view(1, -1)
    print(f"loaded MLP {arch}  (train best val MSE {ckpt.get('best_val_mse')})")
    print(f"feature order: {ckpt.get('feature_order')}")

    # We need a Policy object so world.set_policy works; use any WorldModelPolicy
    # only as a shell, then override get_action. Simplest: build a minimal policy
    # shell by monkey-patching a RandomPolicy instance.
    policy = swm.policy.RandomPolicy()

    def _sq(x, ax):
        """Squeeze a size-1 axis if present (info_dict carries a history dim)."""
        x = np.asarray(x)
        return np.squeeze(x, axis=ax) if x.ndim > 2 and x.shape[ax] == 1 else x

    recorded_actions = []

    def patched_get_action(info_dict, **kwargs):
        qpos = _sq(info_dict["qpos"], 1)        # (B,2)
        qvel = _sq(info_dict["qvel"], 1)        # (B,2)
        goal = _sq(info_dict["goal_qpos"], 1)   # (B,2)
        feats = np.concatenate([qpos, qvel, goal], axis=1).astype(np.float32)  # (B,6)
        x = (torch.tensor(feats, device=device) - in_mean) / in_std
        with torch.no_grad():
            action = model(x).cpu().numpy()     # (B,2), already tanh-bounded
        recorded_actions.append(action)
        return action

    policy.get_action = patched_get_action
    world.set_policy(policy)
    print("\n🚀 MLP policy patched in — bypassing planner.")

    results_path = Path("/data/5pourghe/le_wm_storage/reacher/eval_videos")
    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset=dataset,
        episodes_idx=eval_episodes.tolist(),
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    end_time = time.time()
    print("\n==== RESULTS ====")
    print(metrics)
    print(f"eval time: {end_time - start_time:.1f}s  |  steps intercepted: {len(recorded_actions)}")


if __name__ == "__main__":
    run()
