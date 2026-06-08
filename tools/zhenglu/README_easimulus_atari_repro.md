# EASimulus Atari 100k Reproduction Scripts

These scripts prepare and launch only the EASimulus Atari 100k reproduction tasks. They do not modify official training hyperparameters for full training and do not cover EADream, DMC, Craftax, DMC-GB2, or visualization-video evaluation.

## Paths

- Cloud project root: `/data/share/hxd/zhenglu/eawm`
- EASimulus source: `/data/share/hxd/zhenglu/eawm/EASimulus`
- Conda environment: `zhenglu_easimulus`
- Logs: `/data/share/hxd/zhenglu/eawm/logs`
- Cache: `/data/share/hxd/zhenglu/eawm/cache`
- Full-training outputs: `/data/share/hxd/zhenglu/eawm/outputs/easimulus_atari_4gpu_<timestamp>/`

All scripts support overrides:

```bash
PROJECT_ROOT=/data/share/hxd/zhenglu/eawm ENV_NAME=zhenglu_easimulus bash tools/zhenglu/<script>.sh
```

## 1. One-Command Environment Setup

Run this on the cloud setup node:

```bash
bash tools/zhenglu/setup_easimulus_atari_env.sh
```

The setup script:

- enters `$PROJECT_ROOT/EASimulus`
- logs to `$PROJECT_ROOT/logs/setup_easimulus_atari_<timestamp>.log`
- creates or reuses conda env `zhenglu_easimulus` with Python 3.10
- installs PyTorch `2.5.1` CUDA `12.4` from the official PyTorch wheel index
- installs EASimulus requirements from Tsinghua PyPI mirror
- installs `autorom[accept-rom-license]` and runs `AutoROM --accept-license`
- downloads LPIPS/VGG weights through `python get_lpips.py`
- keeps large caches under `$PROJECT_ROOT/cache`
- runs import, CUDA, Atari reset/step, and `src/main.py` dependency smoke checks

Running `AutoROM --accept-license` means the user confirms the required Atari ROM research-use/license acceptance.

If the current user is root on Ubuntu, the script tries to switch apt sources to Tsinghua and install system packages such as `ffmpeg`, `libsm6`, `libgl1`, `libegl1`, `libosmesa6`, `libglew-dev`, `xvfb`, and `mesa-utils`. If the user is not root, it prints the missing-package hint instead of failing immediately.

## 2. Four-GPU Smoke

Run this only inside a job where 4 GPUs are visible:

```bash
bash tools/zhenglu/run_easimulus_atari_4gpu_smoke.sh
```

The setup node having only 1 visible GPU is normal. The 4-GPU smoke script requires `torch.cuda.device_count() >= 4` and exits with an explicit error if only 1 GPU is visible.

GPU mapping:

- GPU 0: `BreakoutNoFrameskip-v4`
- GPU 1: `BoxingNoFrameskip-v4`
- GPU 2: `SeaquestNoFrameskip-v4`
- GPU 3: `RoadRunnerNoFrameskip-v4`

Each subprocess sees one GPU through `CUDA_VISIBLE_DEVICES=<gpu>` and writes to:

```bash
$PROJECT_ROOT/logs/smoke_easimulus_atari_<game>_<timestamp>.log
```

The master log is:

```bash
$PROJECT_ROOT/logs/smoke_easimulus_atari_4gpu_<timestamp>.log
```

Smoke overrides intentionally shorten runtime, including `common.epochs=3`, one train step per component, `evaluation.every=1`, `common.do_checkpoint=False`, and `wandb.mode=disabled`.

## 3. Four-GPU Full Training

Run this only inside a job where 4 GPUs are visible:

```bash
bash tools/zhenglu/run_easimulus_atari_4gpu_train.sh
```

Default seed is `0`; override it with:

```bash
SEED=1 bash tools/zhenglu/run_easimulus_atari_4gpu_train.sh
```

Default W&B mode is `offline`; override it with:

```bash
WANDB_MODE=online bash tools/zhenglu/run_easimulus_atari_4gpu_train.sh
```

Full training uses the official entrypoint:

```bash
python src/main.py tokenizer.image.with_lpips=True benchmark=atari env.train.id=<game> common.seed=<seed> world_model.event_pred=True world_model.ges=True
```

It does not override official core Atari 100k values such as:

- `common.epochs`
- `collection.train.stop_after_epochs`
- `collection.train.config.num_steps`
- `training.*.steps_per_epoch`
- `evaluation.every`
- world model or actor critic structure parameters

Check the official reference in:

```bash
EASimulus/config/benchmark/atari.yaml
```

Per-task logs:

```bash
$PROJECT_ROOT/logs/train_easimulus_atari_<game>_seed<seed>_<timestamp>.log
```

Master log:

```bash
$PROJECT_ROOT/logs/train_easimulus_atari_4gpu_seed<seed>_<timestamp>.log
```

Outputs:

```bash
$PROJECT_ROOT/outputs/easimulus_atari_4gpu_<timestamp>/<game>_seed<seed>/
```

The full-training script keeps checkpoints enabled for recovery. It disables media-heavy outputs by default:

```bash
collection.train.num_episodes_to_save=0
collection.test.num_episodes_to_save=0
evaluation.tokenizer.save_reconstructions=False
```

If the official code rejects `num_episodes_to_save=0`, the scripts retry once with `1`. You can force that fallback from the start:

```bash
MEDIA_EPISODES_TO_SAVE=1 bash tools/zhenglu/run_easimulus_atari_4gpu_train.sh
```

## Troubleshooting

If `NoFrameskip-v4` creation fails, check that Atari ROMs were installed by `AutoROM --accept-license`. If `ALE/Breakout-v5` works but `BreakoutNoFrameskip-v4` does not, the setup script applies only a minimal compatibility registration patch in `EASimulus/src/envs/wrappers/atari.py`:

```python
gymnasium.register_envs(ale_py)
```

It does not replace official `NoFrameskip-v4` env ids with v5 ids.

If `libGL.so.1` or `libSM.so.6` is missing, system packages were not fully installed. Ask an admin or run as root on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsm6 libxext6 libgl1 libegl1 libgl1-mesa-dev libosmesa6 libosmesa6-dev libglew-dev x11-xserver-utils xvfb mesa-utils
```

If Hugging Face is unreachable, it should not block the current Atari-from-scratch stage because these scripts do not download a pretrained checkpoint from Hugging Face.

To inspect a task:

```bash
tail -f /data/share/hxd/zhenglu/eawm/logs/train_easimulus_atari_Breakout_seed0_<timestamp>.log
```

## Official Checkpoint Demo Videos

Put official pretrained Atari weights in the project-root checkpoint directory:

```bash
/data/share/hxd/zhenglu/eawm/ckpt/Breakout.pt
/data/share/hxd/zhenglu/eawm/ckpt/Boxing.pt
/data/share/hxd/zhenglu/eawm/ckpt/RoadRunner.pt
```

The root `.gitignore` ignores checkpoint payloads under `ckpt/`, while keeping `ckpt/.gitkeep`.

After `tools/zhenglu/setup_easimulus_atari_env.sh` has completed, run the demo environment check:

```bash
bash tools/zhenglu/setup_easimulus_atari_demo_env.sh
```

Then run 4-GPU demo recording inside a 4-GPU compute job:

```bash
bash tools/zhenglu/run_easimulus_atari_4gpu_demo.sh
```

The script scans `ckpt/*.pt`, maps each filename stem to the Atari env id, and records one 3-minute video per checkpoint:

```text
ckpt/Breakout.pt    -> BreakoutNoFrameskip-v4
ckpt/Seaquest.pt    -> SeaquestNoFrameskip-v4
ckpt/RoadRunner.pt  -> RoadRunnerNoFrameskip-v4
```

It runs up to four checkpoints concurrently, one process per GPU, then continues with the next batch. To record only selected tasks:

```bash
TASKS=Breakout,Boxing,Seaquest,RoadRunner bash tools/zhenglu/run_easimulus_atari_4gpu_demo.sh
```

Default video settings are 180 seconds at 15 FPS. Override them with:

```bash
VIDEO_SECONDS=180 FPS=15 bash tools/zhenglu/run_easimulus_atari_4gpu_demo.sh
```

Demo logs:

```bash
$PROJECT_ROOT/logs/demo_easimulus_atari_<task>_<timestamp>.log
$PROJECT_ROOT/logs/demo_easimulus_atari_4gpu_<timestamp>.log
```

Demo videos:

```bash
$PROJECT_ROOT/videos/easimulus_atari_demo_<timestamp>/<task>.mp4
```
