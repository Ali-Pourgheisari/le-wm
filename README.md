> **This repository is a fork of [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm).**
> It adds a behavior-cloning distillation of the original repo's CEM planner for the
> DeepMind Control **Reacher** task. The added work is described below; the original
> project's README follows unchanged beneath the divider.

# Lightweight Reacher Control via Behavior Cloning

**Authors:** Ali Pourgheisari, Mehran Rajabi
**Group:** [Knowledge Technology (WTM)](http://www.informatik.uni-hamburg.de/WTM/), Universität Hamburg

## Overview

The upstream repository plans actions for control tasks with a Cross-Entropy Method (CEM)
solver operating over a learned LeWorldModel (JEPA) latent world model. For Reacher, this
is accurate but slow: each planning solve samples 300 candidate action sequences and refines
them over 30 iterations, taking roughly a minute; solving one full episode (two solves,
covering a 5-step planning horizon with a 5-step action block each) costs on the order of
**two minutes**. That's fine for generating data offline, but far too slow for reactive,
real-time control.

This fork treats the CEM planner as an *expert* and distills its behavior into a lightweight
feed-forward policy: a **68k-parameter MLP** that maps the arm's proprioceptive state and
goal directly to motor torques, with no search and no planning loop. The distilled policy
reaches the target in **68% of held-out episodes** at sub-millisecond inference — trading a
modest amount of task performance for roughly five orders of magnitude in speed.

## Pipeline

**1. Source dataset.** `reacher_random` (10,000 episodes × 201 steps) is used only as a bank
of physically realistic states — not imitated directly. For each demonstration we sample a
valid start step and set the goal to the arm's configuration 25 steps later in that same
trajectory, guaranteeing every goal is reachable.

**2. Expert demonstration collection** (`collect_all.py`). For each of 2,000 rollouts, the
environment is reset to a sampled start state and target joint configuration
(`set_state` + `set_target_qpos`), then the CEM/JEPA expert drives it for 50 steps
(`num_samples=300`, `topk=30`, 30 iterations, horizon 5, action block 5). At every step we
record the **low-dimensional state** — joint angles (`qpos`), joint velocities (`qvel`),
fingertip position — and the executed action; the goal joint configuration is recorded once
per episode. No camera pixels are stored: a rendered frame is ≈1.5×10⁵ values, costs more to
train on, and — critically — doesn't even contain the goal, whereas `(qpos, qvel, goal_qpos)`
does.

**3. Data curation.** A raw CEM rollout reaches the target partway through the episode and
then drifts, so cloning the whole trajectory would teach the policy to wander after arriving.
Each episode is truncated at its **closest approach** to the goal (shortest-arc angular
distance across both joints), and episodes that never come within **10°** of the goal on
both joints are dropped as failures. Of 2,000 collected episodes, 1,993 are retained, giving
**58,715** `(state, action)` training pairs.

**4. Policy architecture and training** (`train_bc_mlp.py`). A plain MLP,
`6 → 256 → 256 → 2` (ReLU hidden layers, `tanh`-bounded output, ≈68k parameters), maps
`[qpos, qvel, goal_qpos]` to a 2D torque command. Trained with MSE loss, Adam
(lr `1e-3`, weight decay `1e-5`), batch size 256, up to 300 epochs with early stopping
(patience 15). The train/validation split is at the **episode** level (85%/15%) so no state
from a held-out episode leaks into training. Training takes about a minute on a single GPU.

**5. Evaluation** (`eval_bc_mlp.py`). The trained MLP is patched directly into the policy's
`get_action` hook and rolled out in closed loop on 50 fresh, seeded start/goal pairs. Success
is scored using the same criterion as the expert: the arm counts as successful if it comes
within 10° of the goal (both joints) at **any** step of the rollout.

## Results

| Metric | Value |
|---|---|
| Closed-loop success rate (50 held-out episodes) | **68%** (34/50) |
| Model size | **~68k** parameters (no transformer, no CNN) |
| Training time | **~1 min** on a single GPU |
| Inference cost | **<1 ms** per step |
| Full evaluation (50 episodes incl. video rendering) | **~18 s** |
| Expert (CEM) reach rate, for comparison | **~99.6%**, at ~2 min/episode |

The residual training-loss error is high relative to a predict-the-mean baseline (MSE 0.122
vs. 0.142) — expected, since CEM is a stochastic optimizer and the same state can map to
different sampled actions across episodes. A unimodal MSE regressor recovers the conditional
*mean* action, which for a reaching task still preserves the correct direction of motion —
hence per-step action error is a poor proxy for task success, and the closed-loop rollout
above is the metric that matters.

## Reproducing

```bash
# 1. Collect expert demonstrations (long-running; ~2 min/episode of CEM search)
python collect_all.py

# 2. Train the distilled MLP policy
python train_bc_mlp.py

# 3. Evaluate the distilled policy in closed loop
python eval_bc_mlp.py --config-name reacher
```

Collected data and trained weights are written under `$STABLEWM_HOME` (see the
**Data** section below for how that path is configured) rather than tracked in git.

## Possible next steps

- Scale demonstrations by sampling multiple goals per source episode (2k → 10k+).
- Enrich the goal representation with Cartesian fingertip information.
- Replace the unimodal MSE head with a distributional action head (e.g. diffusion) to
  better capture the expert's stochastic, multimodal action distribution.
- Interactive data aggregation (DAgger) to address covariate shift in the failure cases.

---

# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
