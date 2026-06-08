#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from envs import SingleProcessEnv
from game import AgentEnv
from main import build_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a non-interactive EASimulus Atari agent demo.")
    parser.add_argument("--easimulus-dir", required=True, type=Path)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_cfg(easimulus_dir: Path, env_id: str, seed: int):
    config_dir = (easimulus_dir / "config").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None, job_name="atari_demo"):
        cfg = compose(
            config_name="base",
            overrides=[
                "benchmark=atari",
                f"env.train.id={env_id}",
                f"env.test.id={env_id}",
                "common.device=cuda:0",
                f"common.seed={seed}",
                "wandb.mode=disabled",
                "hydra.run.dir=.",
                "hydra.output_subdir=null",
            ],
        )
    lpips_dir = easimulus_dir / "cache" / "rem" / "tokenizer_pretrained_vgg"
    cfg.tokenizer.image.vgg_lpips_ckpt_path = str(lpips_dir.resolve())
    return cfg


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    easimulus_dir = args.easimulus_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This demo recorder expects one visible GPU per process.")

    os.chdir(easimulus_dir)
    set_seed(args.seed)
    cfg = build_cfg(easimulus_dir, args.env_id, args.seed)

    import ale_py
    import gymnasium

    try:
        gymnasium.register_envs(ale_py)
    except Exception:
        pass

    print(f"[demo] env_id={args.env_id}")
    print(f"[demo] checkpoint={checkpoint}")
    print(f"[demo] output={output}")
    print(f"[demo] seconds={args.seconds}, fps={args.fps}, frames={args.seconds * args.fps}")
    print(f"[demo] cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[demo] torch={torch.__version__}, cuda={torch.version.cuda}, device={torch.cuda.get_device_name(0)}")
    print(f"[demo] cfg.env.test.id={cfg.env.test.id}")
    print(f"[demo] cfg.common.device={cfg.common.device}")
    print(f"[demo] lpips={cfg.tokenizer.image.vgg_lpips_ckpt_path}")

    env_partial = instantiate(cfg.env.test)
    env_fn = lambda: env_partial(tokenizer_config=cfg.tokenizer)
    test_env = SingleProcessEnv(env_fn)
    device = torch.device(cfg.common.device)
    agent = build_agent(test_env, cfg, device)
    agent.load(checkpoint, device)
    agent.eval()

    demo_env = AgentEnv(agent, test_env, cfg.env.keymap, do_reconstruction=False)
    demo_env.reset()

    output.parent.mkdir(parents=True, exist_ok=True)
    total_frames = args.seconds * args.fps

    try:
        with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8, macro_block_size=16) as writer:
            for frame_idx in range(total_frames):
                if frame_idx % max(args.fps * 10, 1) == 0:
                    print(f"[demo] frame {frame_idx}/{total_frames}", flush=True)
                image = demo_env.render()
                writer.append_data(np.asarray(image.convert("RGB")))
                _, _, terminated, truncated, _ = demo_env.step()
                if bool(terminated[0]) or bool(truncated[0]):
                    demo_env.reset()
        print(f"[demo] saved {output}")
    finally:
        test_env.close()


if __name__ == "__main__":
    main()
