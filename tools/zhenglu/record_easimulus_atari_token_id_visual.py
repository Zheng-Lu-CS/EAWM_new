#!/usr/bin/env python
from __future__ import annotations

import argparse
import colorsys
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image, ImageDraw, ImageFont

from envs import SingleProcessEnv
from game import AgentEnv
from main import build_agent
from utils import ObsModality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record an EASimulus Atari agent with a VQ token-id map."
    )
    parser.add_argument("--easimulus-dir", required=True, type=Path)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--panel-height", type=int, default=104)
    return parser.parse_args()


def build_cfg(easimulus_dir: Path, env_id: str, seed: int):
    config_dir = (easimulus_dir / "config").resolve()
    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
        job_name="atari_token_id_visual",
    ):
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


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def token_color(token_id: int) -> tuple[int, int, int]:
    hue = ((token_id * 0.61803398875) % 1.0)
    saturation = 0.58 + 0.18 * ((token_id * 37) % 7) / 6
    value = 0.82 + 0.12 * ((token_id * 17) % 5) / 4
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def infer_grid_shape(num_tokens: int, input_hw: tuple[int, int]) -> tuple[int, int]:
    input_h, input_w = input_hw
    grid_h = int(math.sqrt(num_tokens * input_h / input_w))
    grid_h = max(grid_h, 1)
    while num_tokens % grid_h != 0 and grid_h > 1:
        grid_h -= 1
    return grid_h, num_tokens // grid_h


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    text_box = draw.textbbox((0, 0), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    x = box[0] + (box[2] - box[0] - text_w) // 2
    y = box[1] + (box[3] - box[1] - text_h) // 2
    draw.text((x, y), text, font=font, fill=fill)


def render_token_grid(
    tokens: np.ndarray,
    grid_shape: tuple[int, int],
    cell_size: int,
) -> Image.Image:
    grid_h, grid_w = grid_shape
    image = Image.new("RGB", (grid_w * cell_size, grid_h * cell_size), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    font = load_font(max(12, int(cell_size * 0.28)))
    tokens_2d = tokens.reshape(grid_h, grid_w)

    for y in range(grid_h):
        for x in range(grid_w):
            token_id = int(tokens_2d[y, x])
            x0 = x * cell_size
            y0 = y * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            color = token_color(token_id)
            draw.rectangle((x0, y0, x1, y1), fill=color)
            luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
            text_color = (12, 12, 12) if luminance > 145 else (245, 245, 245)
            draw_centered_text(draw, (x0, y0, x1, y1), str(token_id), font, text_color)

    for x in range(grid_w + 1):
        px = x * cell_size
        draw.line((px, 0, px, image.height), fill=(0, 0, 0), width=1)
    for y in range(grid_h + 1):
        py = y * cell_size
        draw.line((0, py, image.width, py), fill=(0, 0, 0), width=1)

    return image


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), resample=Image.NEAREST)


def render_frame(
    task_name: str,
    frame_index: int,
    num_frames: int,
    real_image: Image.Image,
    tokens: np.ndarray,
    grid_shape: tuple[int, int],
    token_change: float | None,
    mean_token_change: float,
    info: dict[str, str],
    cell_size: int,
    panel_height: int,
) -> Image.Image:
    token_grid = render_token_grid(tokens, grid_shape, cell_size)
    real_resized = resize_to_height(real_image.convert("RGB"), token_grid.height)

    content_width = real_resized.width + token_grid.width
    content_height = token_grid.height
    canvas = Image.new(
        "RGB",
        (content_width, content_height + panel_height),
        (238, 240, 236),
    )
    canvas.paste(real_resized, (0, 0))
    canvas.paste(token_grid, (real_resized.width, 0))

    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    small_font = load_font(15)
    panel_y = content_height
    draw.rectangle(
        (0, panel_y, canvas.width, canvas.height),
        fill=(24, 26, 28),
        outline=(24, 26, 28),
    )

    grid_h, grid_w = grid_shape
    change_text = "n/a" if token_change is None else f"{token_change:.4f}"
    timestep = info.get("Timestep", "0")
    action = info.get("Action", "RESET")
    ret = info.get("Return", "0.00")
    lines = [
        f"{task_name} | frame {frame_index + 1:03d}/{num_frames} | env_step {timestep} | action {action} | return {ret}",
        f"token_change_current {change_text} | token_change_mean {mean_token_change:.4f} | token_grid {grid_h}x{grid_w} | video_fps 1",
        "left: observed Atari frame after action repeat | right: VQ-VAE codebook token id map",
    ]
    y = panel_y + 12
    for i, line in enumerate(lines):
        draw.text(
            (14, y),
            line,
            font=font if i == 0 else small_font,
            fill=(245, 245, 245) if i == 0 else (210, 218, 214),
        )
        y += 30

    return canvas


def pad_to_macroblock(image: Image.Image, multiple: int = 16) -> Image.Image:
    width = int(math.ceil(image.width / multiple) * multiple)
    height = int(math.ceil(image.height / multiple) * multiple)
    if width == image.width and height == image.height:
        return image
    padded = Image.new("RGB", (width, height), (24, 26, 28))
    padded.paste(image, (0, 0))
    return padded


@torch.no_grad()
def get_tokens(agent, obs: dict[ObsModality, torch.Tensor]) -> np.ndarray:
    encoded = agent.tokenizer.encode(obs, should_preprocess=True)
    tokens = encoded[ObsModality.image].tokens
    assert tokens.size(0) == 1, tokens.shape
    return tokens[0].detach().cpu().numpy().astype(np.int64)


def main() -> None:
    args = parse_args()
    easimulus_dir = args.easimulus_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    task_name = checkpoint.stem
    num_frames = args.seconds * args.fps

    if args.fps != 1:
        raise ValueError("This visualizer records one observed environment transition per second, so --fps must be 1.")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run one process per visible GPU.")

    os.chdir(easimulus_dir)
    set_seed(args.seed)
    cfg = build_cfg(easimulus_dir, args.env_id, args.seed)

    import ale_py
    import gymnasium

    try:
        gymnasium.register_envs(ale_py)
    except Exception:
        pass

    print(f"[token-vis] task={task_name}")
    print(f"[token-vis] env_id={args.env_id}")
    print(f"[token-vis] checkpoint={checkpoint}")
    print(f"[token-vis] output={output}")
    print(f"[token-vis] seconds={args.seconds}, fps={args.fps}, frames={num_frames}")
    print(f"[token-vis] cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[token-vis] torch={torch.__version__}, cuda={torch.version.cuda}, device={torch.cuda.get_device_name(0)}")

    env_partial = instantiate(cfg.env.test)
    env_fn = lambda: env_partial(tokenizer_config=cfg.tokenizer)
    test_env = SingleProcessEnv(env_fn)
    device = torch.device(cfg.common.device)
    agent = build_agent(test_env, cfg, device)
    agent.load(
        checkpoint,
        device,
        load_tokenizer=True,
        load_world_model=False,
        load_actor_critic=True,
    )
    agent.eval()

    visual_env = AgentEnv(agent, test_env, cfg.env.keymap, do_reconstruction=False)
    visual_env.reset()

    input_resolution = tuple(cfg.tokenizer.image.encoder.config.input_resolution)
    expected_tokens = agent.tokenizer.tokenizers[ObsModality.image.name].tokens_per_obs
    grid_shape = infer_grid_shape(expected_tokens, input_resolution)
    print(f"[token-vis] inferred token grid: {grid_shape[0]}x{grid_shape[1]} ({expected_tokens} tokens)")

    frames: list[Image.Image] = []
    token_arrays: list[np.ndarray] = []
    infos: list[dict[str, str]] = []
    current_info = {"Timestep": "0", "Action": "RESET", "Return": "0.00"}

    try:
        for frame_idx in range(num_frames):
            if frame_idx % 30 == 0:
                print(f"[token-vis] collecting frame {frame_idx}/{num_frames}", flush=True)
            tokens = get_tokens(agent, visual_env.obs)
            if tokens.size != expected_tokens:
                raise RuntimeError(
                    f"Expected {expected_tokens} tokens, got {tokens.size}."
                )
            frames.append(visual_env.render().copy())
            token_arrays.append(tokens.copy())
            infos.append(dict(current_info))

            _, _, terminated, truncated, step_info = visual_env.step()
            current_info = {k: str(v) for k, v in step_info.items()}
            if bool(terminated[0]) or bool(truncated[0]):
                visual_env.reset()
                current_info = {
                    "Timestep": "0",
                    "Action": "RESET",
                    "Return": "0.00",
                }

        change_rates: list[float | None] = [None]
        for prev, curr in zip(token_arrays, token_arrays[1:]):
            change_rates.append(float(np.mean(prev != curr)))
        valid_rates = [x for x in change_rates if x is not None]
        mean_change = float(np.mean(valid_rates)) if valid_rates else 0.0
        print(f"[token-vis] mean adjacent token change rate: {mean_change:.6f}")

        output.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            output,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        ) as writer:
            for frame_idx, (real_image, tokens, rate, info) in enumerate(
                zip(frames, token_arrays, change_rates, infos)
            ):
                if frame_idx % 30 == 0:
                    print(f"[token-vis] writing frame {frame_idx}/{num_frames}", flush=True)
                frame = render_frame(
                    task_name=task_name,
                    frame_index=frame_idx,
                    num_frames=num_frames,
                    real_image=real_image,
                    tokens=tokens,
                    grid_shape=grid_shape,
                    token_change=rate,
                    mean_token_change=mean_change,
                    info=info,
                    cell_size=args.cell_size,
                    panel_height=args.panel_height,
                )
                frame = pad_to_macroblock(frame)
                writer.append_data(np.asarray(frame))
        print(f"[token-vis] saved {output}")
    finally:
        test_env.close()


if __name__ == "__main__":
    main()
