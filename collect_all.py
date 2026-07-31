import os
os.environ["MUJOCO_GL"] = "egl"

import time
import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
import stable_pretraining as spt

print("Starting Memory-Isolated Collection Script...")

# --- Configuration Toggle ---
# Set to False for xl_clean_... | Set to True for large_recovery_...
INJECT_NOISE = False
TOTAL_EPISODES = 2000
CHUNK_SIZE = 20       # Exactly matches the environment instantiation constraint
RECORD_PIXELS = False # low-dim MLP path: skip 14 GB image dump; keep only state

def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])

def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)

@hydra.main(version_base=None, config_path="./config/eval", config_name="reacher")
def run(cfg: DictConfig):
    # Set our chunk evaluation budget explicitly for safety
    cfg.eval.num_eval = CHUNK_SIZE
    
    # Load dataset properties once
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(cfg.eval.dataset_name, keys_to_cache=cfg.dataset.keys_to_cache, cache_dir=dataset_path)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    
    # Preprocessing pipelines
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]: continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action": process[f"goal_{col}"] = process[col]

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    
    g = np.random.default_rng(cfg.seed)
    
    master_actions = []
    master_observations = []
    master_goals = []
    # low-dim state accumulators
    master_qpos, master_qvel, master_obs_prop, master_finger = [], [], [], []
    master_goal_qpos, master_goal_obs, master_goal_finger = [], [], []
    
    # Run loop sequentially across isolated world instances
    for step in range(0, TOTAL_EPISODES, CHUNK_SIZE):
        print(f"\n🚀 Harvesting subset window: episodes {step} to {step + CHUNK_SIZE}...")
        
        # Fresh environment instantiation inside the loop!
        cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
        world = swm.World(**cfg.world, image_shape=(224, 224))
        
        # Fresh solver initialization
        checkpoint_path = Path("/data/5pourghe/le_wm_storage/reacher/lewm_object.ckpt")
        model = torch.load(checkpoint_path, map_location="cuda")
        model.eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        
        solver = hydra.utils.instantiate(cfg.solver, model=model)

        # =============================================================
        # FULL-STRENGTH CEM
        # The previous "safe scaling" capped num_samples to 64 while keeping
        # topk=30 → a 47% elite fraction that removed almost all of CEM's
        # selection pressure, making the expert near-random (most episodes
        # never reached the goal). We now use the config defaults
        # (num_samples=300, topk=30 → 10% elite). Only guard the hard
        # requirement that num_samples > topk so torch.topk can't overflow.
        # =============================================================
        if hasattr(solver, 'topk'):
            assert solver.num_samples > solver.topk, (
                f"num_samples ({solver.num_samples}) must exceed topk ({solver.topk})"
            )
        print(f"CEM: num_samples={getattr(solver, 'num_samples', '?')} "
              f"topk={getattr(solver, 'topk', '?')} "
              f"n_steps={getattr(solver, 'n_steps', '?')}")

        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=swm.PlanConfig(**cfg.plan_config), 
            process=process, transform={"pixels": img_transform(cfg), "goal": img_transform(cfg)}
        )
        
        # Sample index slices for this individual chunk
        random_indices = np.sort(g.choice(valid_indices, size=CHUNK_SIZE, replace=False))
        eval_episodes = dataset.get_row_data(random_indices)[col_name].tolist()
        eval_start_idx = dataset.get_row_data(random_indices)["step_idx"].tolist()
        
        # Interceptor tracking structures
        recorded_actions = []
        recorded_observations = []     # pixels (only if RECORD_PIXELS)
        recorded_goals = []            # goal image, once per chunk (only if RECORD_PIXELS)
        # low-dim state, recorded every step (LIVE rollout state)
        rec_qpos, rec_qvel, rec_obs, rec_finger = [], [], [], []
        # low-dim goal, recorded once per chunk (STATIC per episode)
        goal_lowdim = {}
        original_get_action = policy.get_action

        def _cap(x):
            return np.ascontiguousarray(np.asarray(x))

        def patched_get_action(info_dict, **kwargs):
            # LIVE proprioceptive state of the rollout arm at this step
            rec_qpos.append(_cap(info_dict['qpos']))        # (B, 2)
            rec_qvel.append(_cap(info_dict['qvel']))        # (B, 2)
            rec_obs.append(_cap(info_dict['observation']))  # (B, 6)
            rec_finger.append(_cap(info_dict['finger_pos']))# (B, 2)

            # STATIC goal (joint config the episode must match) — record once
            if not goal_lowdim:
                goal_lowdim['goal_qpos']        = _cap(info_dict['goal_qpos'])
                goal_lowdim['goal_observation'] = _cap(info_dict['goal_observation'])
                goal_lowdim['goal_finger_pos']  = _cap(info_dict['goal_finger_pos'])

            if RECORD_PIXELS:
                recorded_observations.append(info_dict['pixels'].copy())
                if not recorded_goals:
                    g_img = np.asarray(info_dict['goal'])
                    recorded_goals.append(np.ascontiguousarray(g_img.squeeze(1)))

            expert_action = original_get_action(info_dict, **kwargs)
            # Clip to [-1, 1] to match what the environment actually executes
            # (CEM samples from an unbounded Gaussian so raw actions can exceed bounds)
            action_to_save = np.clip(expert_action, -1.0, 1.0)
            if INJECT_NOISE:
                noise = np.random.normal(0.0, 0.15, size=action_to_save.shape)
                action_to_save = np.clip(action_to_save + noise, -1.0, 1.0)
            recorded_actions.append(action_to_save)
            return expert_action  # return original so env clips naturally

        policy.get_action = patched_get_action
        world.set_policy(policy)
        
        results_path = Path("/data/5pourghe/le_wm_storage/reacher/videos")
        results_path.mkdir(parents=True, exist_ok=True)

        # Execute active simulation run.
        # Pass the reacher callables (set_state + set_target_qpos) so the env's
        # initial state AND target joint config are set from the dataset — the
        # working eval_bc.py does this; collection previously passed None, which
        # left the target unset and made CEM optimize toward nothing coherent.
        world.evaluate_from_dataset(
            dataset=dataset, start_steps=eval_start_idx,
            goal_offset_steps=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes,
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video_path=str(results_path)
        )
        
        if recorded_actions:
            master_actions.append(np.array(recorded_actions))          # (T, B, 2)
            master_qpos.append(np.array(rec_qpos))                     # (T, B, 2)
            master_qvel.append(np.array(rec_qvel))                     # (T, B, 2)
            master_obs_prop.append(np.array(rec_obs))                 # (T, B, 6)
            master_finger.append(np.array(rec_finger))                # (T, B, 2)
            master_goal_qpos.append(goal_lowdim['goal_qpos'])         # (B, 2)
            master_goal_obs.append(goal_lowdim['goal_observation'])   # (B, 6)
            master_goal_finger.append(goal_lowdim['goal_finger_pos']) # (B, 2)
            if RECORD_PIXELS:
                master_observations.append(np.array(recorded_observations))
                master_goals.append(recorded_goals[0])

        # Complete memory tear down before launching the next loop step
        del world, solver, model, policy
        torch.cuda.empty_cache()

    # Save finalized matrices
    if master_actions:
        output_dir = Path("/data/5pourghe/le_wm_storage/reacher")
        output_dir.mkdir(parents=True, exist_ok=True)
        pfx = "large_recovery" if INJECT_NOISE else "xl_clean"

        # Per-step arrays: (steps, episodes, dim) — same (T, E) layout as before.
        actions_final = np.concatenate(master_actions,  axis=1)   # (T, E, 2)
        qpos_final    = np.concatenate(master_qpos,      axis=1)   # (T, E, 2)
        qvel_final    = np.concatenate(master_qvel,      axis=1)   # (T, E, 2)
        obsprop_final = np.concatenate(master_obs_prop,  axis=1)   # (T, E, 6)
        finger_final  = np.concatenate(master_finger,    axis=1)   # (T, E, 2)
        # Per-episode goal arrays: (episodes, dim)
        goal_qpos_final   = np.concatenate(master_goal_qpos,   axis=0)  # (E, 2)
        goal_obs_final    = np.concatenate(master_goal_obs,    axis=0)  # (E, 6)
        goal_finger_final = np.concatenate(master_goal_finger, axis=0)  # (E, 2)

        print("\n🎉 Collection complete!")
        print(f"actions {actions_final.shape} | qpos {qpos_final.shape} | "
              f"qvel {qvel_final.shape} | goal_qpos {goal_qpos_final.shape}")

        np.save(output_dir / f"{pfx}_actions.npy",      actions_final)
        np.save(output_dir / f"{pfx}_qpos.npy",         qpos_final)
        np.save(output_dir / f"{pfx}_qvel.npy",         qvel_final)
        np.save(output_dir / f"{pfx}_observation.npy",  obsprop_final)
        np.save(output_dir / f"{pfx}_finger_pos.npy",   finger_final)
        np.save(output_dir / f"{pfx}_goal_qpos.npy",    goal_qpos_final)
        np.save(output_dir / f"{pfx}_goal_observation.npy", goal_obs_final)
        np.save(output_dir / f"{pfx}_goal_finger_pos.npy",  goal_finger_final)

        if RECORD_PIXELS:
            obs_final   = np.concatenate(master_observations, axis=1)
            goals_final = np.concatenate(master_goals, axis=0)
            np.save(output_dir / f"{pfx}_observations.npy", obs_final)
            np.save(output_dir / f"{pfx}_goals.npy",        goals_final)
            print(f"pixels {obs_final.shape} | goal_img {goals_final.shape}")

        print("✅ Files saved to", output_dir)

if __name__ == "__main__":
    run()