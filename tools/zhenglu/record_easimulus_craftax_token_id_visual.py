#!/usr/bin/env python
from __future__ import annotations

import argparse
import colorsys
import csv
import math
import os
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image, ImageDraw, ImageFont

from envs import SingleProcessEnv
from main import build_agent
from utils import ObsModality
from utils.preprocessing import get_obs_processor


MODALITY_ORDER = [ObsModality.vector, ObsModality.token, ObsModality.token_2d]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a Craftax EASimulus demo with token-id visualization and change-rate statistics."
    )
    parser.add_argument("--easimulus-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-name", default="craftax_token_id_demo.mp4")
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell-size", type=int, default=58)
    parser.add_argument("--panel-height", type=int, default=132)
    parser.add_argument("--pixel-render-size", type=int, default=0)
    return parser.parse_args()


def build_cfg(easimulus_dir: Path, seed: int):
    config_dir = (easimulus_dir / "config").resolve()
    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
        job_name="craftax_token_id_visual",
    ):
        cfg = compose(
            config_name="base",
            overrides=[
                "benchmark=craftax",
                "common.device=cuda:0",
                f"common.seed={seed}",
                "wandb.mode=disabled",
                "hydra.run.dir=.",
                "hydra.output_subdir=null",
            ],
        )
    return cfg


def patch_gymnax_discrete_space_conversion() -> None:
    """Gymnax versions around the Craftax release miss Discrete -> Gym conversion."""
    import gymnasium
    from gymnax.environments import spaces as gymnax_spaces

    original = gymnax_spaces.gymnax_space_to_gym_space
    if getattr(original, "_eawm_craftax_discrete_patch", False):
        return

    def patched(space):
        try:
            return original(space)
        except NotImplementedError:
            if space.__class__.__name__ != "Discrete":
                raise
            n = getattr(space, "n", None)
            if n is None:
                n = getattr(space, "num_categories", None)
            if n is None:
                n = getattr(space, "num_values", None)
            if n is None:
                raise
            if hasattr(n, "item"):
                n = n.item()
            return gymnasium.spaces.Discrete(int(n))

    patched._eawm_craftax_discrete_patch = True
    gymnax_spaces.gymnax_space_to_gym_space = patched


def load_craftax_block_pixel_size() -> int:
    for module_name in ("craftax.craftax.constants", "craftax.craftax.play_craftax"):
        try:
            module = __import__(module_name, fromlist=["BLOCK_PIXEL_SIZE_HUMAN"])
            return int(getattr(module, "BLOCK_PIXEL_SIZE_HUMAN"))
        except (ImportError, AttributeError):
            continue
    return 16


def load_craftax_action_names(num_actions: int) -> list[str]:
    for module_name in ("craftax.craftax.constants", "craftax.craftax.play_craftax"):
        try:
            module = __import__(module_name, fromlist=["Action"])
            action_enum = getattr(module, "Action")
            names = [action.name for action in action_enum]
            if len(names) >= num_actions:
                return names
        except (ImportError, AttributeError, TypeError):
            continue
    return [f"action_{i}" for i in range(num_actions)]


def make_craftax_pixel_renderer(block_pixel_size: int):
    import jax
    from craftax.craftax import renderer

    if hasattr(renderer, "make_craftax_pixel_renderer"):
        render_fn = renderer.make_craftax_pixel_renderer(block_pixel_size)
        return jax.jit(render_fn)

    if hasattr(renderer, "render_craftax_pixels"):
        render_fn = renderer.render_craftax_pixels
        return jax.jit(lambda state: render_fn(state, block_pixel_size=block_pixel_size))

    return None


def find_gymnax_wrapper(env: SingleProcessEnv):
    current = env.env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "env_state"):
            return current
        current = getattr(current, "env", None)
    return None


class CraftaxTokenDemoEnv:
    def __init__(self, agent, env: SingleProcessEnv, pixel_render_size: int) -> None:
        self.agent = agent
        self.env = env
        self.pixel_render_size = pixel_render_size
        self.obs = None
        self._t = 0
        self._return = 0.0
        self.obs_processors = {m: get_obs_processor(m) for m in env.modalities}
        self.action_names = load_craftax_action_names(env.num_actions or 0)
        block_pixel_size = load_craftax_block_pixel_size()
        self._renderer = make_craftax_pixel_renderer(block_pixel_size)
        self._gymnax_wrapper = find_gymnax_wrapper(env)
        if self._renderer is None:
            print("[craftax-token-vis][warn] Craftax pixel renderer not found; using token-map fallback renderer.")
        elif self._gymnax_wrapper is None:
            print("[craftax-token-vis][warn] Gymnax env_state not found; using token-map fallback renderer.")

    @torch.no_grad()
    def _to_tensor(self, obs: dict[str, np.ndarray]):
        assert isinstance(obs, dict)
        expected = {m.name for m in self.env.modalities}
        assert set(obs.keys()) == expected, f"{set(obs.keys())} != {expected}"
        torch_obs = {
            m: self.obs_processors[m].to_torch(obs[m.name], device=self.agent.device)
            for m in self.env.modalities
        }
        return {m: self.obs_processors[m](v) for m, v in torch_obs.items()}

    def reset(self):
        obs, _ = self.env.reset()
        self.obs = self._to_tensor(obs)
        self.agent.actor_critic.reset(1)
        self._t = 0
        self._return = 0.0
        return obs

    def step(self):
        with torch.no_grad():
            act = self.agent.act(self.obs, should_sample=True).cpu().numpy()
        obs, reward, terminated, truncated, _ = self.env.step(act)
        self.obs = self._to_tensor(obs)
        self._t += 1
        self._return += float(reward[0])
        action_idx = int(act[0])
        action_name = self.action_names[action_idx] if action_idx < len(self.action_names) else f"action_{action_idx}"
        info = {
            "Timestep": self._t,
            "Action": action_name,
            "Return": f"{self._return:.2f}",
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        if self._renderer is not None and self._gymnax_wrapper is not None:
            try:
                pixels = self._renderer(self._gymnax_wrapper.env_state)
                arr = np.asarray(pixels)
                if arr.dtype != np.uint8:
                    if arr.size and float(np.nanmax(arr)) <= 1.5:
                        arr = arr * 255.0
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                arr = np.repeat(arr, repeats=self.pixel_render_size, axis=0)
                arr = np.repeat(arr, repeats=self.pixel_render_size, axis=1)
                return Image.fromarray(arr)
            except Exception as exc:
                print(f"[craftax-token-vis][warn] Craftax pixel render failed once; using token-map fallback. error={exc}")
                self._renderer = None

        tokens = self.obs[ObsModality.token_2d][0].detach().cpu().numpy().astype(np.int64)
        image = render_token_grid(tokens, direction_token=None, cell_size=max(18, self.pixel_render_size * 8))
        return image


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


def token_color(token_id: int, channel: int = 0) -> tuple[int, int, int]:
    hue = (((token_id + 997 * channel) * 0.61803398875) % 1.0)
    saturation = 0.55 + 0.2 * ((token_id * 37 + channel * 11) % 7) / 6
    value = 0.78 + 0.14 * ((token_id * 17 + channel * 5) % 5) / 4
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def infer_map_shape(num_cells: int) -> tuple[int, int]:
    if num_cells == 99:
        return 9, 11
    rows = int(math.sqrt(num_cells))
    rows = max(rows, 1)
    while num_cells % rows != 0 and rows > 1:
        rows -= 1
    return rows, num_cells // rows


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
    map_tokens: np.ndarray,
    direction_token: int | None,
    cell_size: int,
) -> Image.Image:
    tokens = np.asarray(map_tokens, dtype=np.int64)
    if tokens.ndim == 1:
        tokens = tokens[:, None]
    num_cells, channels = tokens.shape
    grid_h, grid_w = infer_map_shape(num_cells)
    tokens = tokens.reshape(grid_h, grid_w, channels)

    header_h = max(28, int(cell_size * 0.52))
    image = Image.new("RGB", (grid_w * cell_size, header_h + grid_h * cell_size), (21, 22, 24))
    draw = ImageDraw.Draw(image)
    header_font = load_font(max(12, int(cell_size * 0.22)))
    small_font = load_font(max(8, int(cell_size * 0.16)))

    header = "token_2d map ids: block/item/mob/light"
    if direction_token is not None:
        header += f" | direction {direction_token}"
    draw.text((8, 6), header, font=header_font, fill=(235, 238, 232))

    labels = ["B", "I", "M", "L"]
    for y in range(grid_h):
        for x in range(grid_w):
            x0 = x * cell_size
            y0 = header_h + y * cell_size
            half = cell_size // 2
            boxes = [
                (x0, y0, x0 + half, y0 + half),
                (x0 + half, y0, x0 + cell_size, y0 + half),
                (x0, y0 + half, x0 + half, y0 + cell_size),
                (x0 + half, y0 + half, x0 + cell_size, y0 + cell_size),
            ]
            for channel in range(min(channels, 4)):
                token_id = int(tokens[y, x, channel])
                color = token_color(token_id, channel)
                draw.rectangle(boxes[channel], fill=color)
                luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                text_color = (12, 12, 12) if luminance > 145 else (245, 245, 245)
                draw_centered_text(draw, boxes[channel], f"{labels[channel]}{token_id}", small_font, text_color)
            draw.rectangle((x0, y0, x0 + cell_size, y0 + cell_size), outline=(0, 0, 0), width=2)

    return image


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), resample=Image.NEAREST)


def fmt_rate(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def render_frame(
    frame_index: int,
    num_frames: int,
    real_image: Image.Image,
    map_tokens: np.ndarray,
    direction_token: int | None,
    pixel_current: float | None,
    pixel_mean: float,
    pixel_abs_current: float | None,
    pixel_abs_mean: float,
    token_current: float | None,
    token_mean: float,
    token_map_current: float | None,
    token_vector_current: float | None,
    info: dict[str, str],
    cell_size: int,
    panel_height: int,
    fps: int,
) -> Image.Image:
    token_grid = render_token_grid(map_tokens, direction_token, cell_size)
    real_resized = resize_to_height(real_image.convert("RGB"), token_grid.height)

    content_width = real_resized.width + token_grid.width
    content_height = token_grid.height
    canvas = Image.new("RGB", (content_width, content_height + panel_height), (238, 240, 236))
    canvas.paste(real_resized, (0, 0))
    canvas.paste(token_grid, (real_resized.width, 0))

    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    small_font = load_font(15)
    panel_y = content_height
    draw.rectangle((0, panel_y, canvas.width, canvas.height), fill=(24, 26, 28), outline=(24, 26, 28))

    timestep = info.get("Timestep", "0")
    action = info.get("Action", "RESET")
    ret = info.get("Return", "0.00")
    lines = [
        f"Craftax | frame {frame_index + 1:03d}/{num_frames} | env_step {timestep} | action {action} | return {ret}",
        f"pixel_change_current {fmt_rate(pixel_current)} | pixel_change_mean {pixel_mean:.4f} | pixel_abs_delta {fmt_rate(pixel_abs_current)} / {pixel_abs_mean:.4f}",
        f"token_change_current {fmt_rate(token_current)} | token_change_mean {token_mean:.4f} | map {fmt_rate(token_map_current)} | vector {fmt_rate(token_vector_current)} | video_fps {fps}",
        "left: observed Craftax frame after agent step | right: Simulus observation token ids",
    ]
    y = panel_y + 11
    for i, line in enumerate(lines):
        draw.text((14, y), line, font=font if i == 0 else small_font, fill=(245, 245, 245) if i == 0 else (210, 218, 214))
        y += 29

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
def get_encoded_tokens(agent, obs: dict[ObsModality, torch.Tensor]) -> dict[str, np.ndarray]:
    encoded = agent.tokenizer.encode(obs, should_preprocess=True)
    tokens: dict[str, np.ndarray] = {}
    for modality in MODALITY_ORDER:
        if modality not in encoded:
            continue
        arr = encoded[modality].tokens.detach().cpu().numpy().astype(np.int64)
        if arr.shape[0] == 1:
            arr = arr[0]
        tokens[modality.name] = arr.copy()
    return tokens


def flatten_tokens(tokens: dict[str, np.ndarray]) -> np.ndarray:
    parts = [tokens[m.name].reshape(-1) for m in MODALITY_ORDER if m.name in tokens]
    if not parts:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(parts).astype(np.int64)


def token_change(prev: dict[str, np.ndarray] | None, curr: dict[str, np.ndarray], key: str | None = None) -> float | None:
    if prev is None:
        return None
    if key is None:
        prev_arr = flatten_tokens(prev)
        curr_arr = flatten_tokens(curr)
    else:
        if key not in prev or key not in curr:
            return None
        prev_arr = prev[key].reshape(-1)
        curr_arr = curr[key].reshape(-1)
    if prev_arr.size == 0 or prev_arr.shape != curr_arr.shape:
        return None
    return float(np.mean(prev_arr != curr_arr))


def pixel_change(prev: np.ndarray | None, curr: np.ndarray) -> tuple[float | None, float | None]:
    if prev is None or prev.shape != curr.shape:
        return None, None
    changed = np.any(prev != curr, axis=-1) if curr.ndim == 3 else (prev != curr)
    changed_fraction = float(np.mean(changed))
    abs_delta = float(np.mean(np.abs(curr.astype(np.float32) - prev.astype(np.float32))) / 255.0)
    return changed_fraction, abs_delta


def running_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def finite_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in ["mean", "std", "min", "p50", "p90", "p95", "max"]}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stat_row(name: str, values: list[float]) -> str:
    stats = finite_stats(values)
    return (
        f"| {name} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | "
        f"{stats['p50']:.6f} | {stats['p90']:.6f} | {stats['p95']:.6f} | {stats['max']:.6f} |"
    )


def write_report(
    path: Path,
    video_path: Path,
    csv_path: Path,
    checkpoint: Path,
    cfg,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    rates: dict[str, list[float]],
) -> None:
    top_token = sorted(
        [r for r in rows if isinstance(r["token_change_current"], float)],
        key=lambda r: float(r["token_change_current"]),
        reverse=True,
    )[:10]
    top_pixel = sorted(
        [r for r in rows if isinstance(r["pixel_change_current"], float)],
        key=lambda r: float(r["pixel_change_current"]),
        reverse=True,
    )[:10]

    lines = [
        "# Craftax Token-ID Demo Report",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"Checkpoint: `{checkpoint}`",
        f"Video: `{video_path}`",
        f"Per-frame CSV: `{csv_path}`",
        "",
        "## Run Configuration",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Environment | `{cfg.env.test.id}` |",
        f"| Seed | `{args.seed}` |",
        f"| Duration | `{args.seconds}` seconds |",
        f"| Video FPS | `{args.fps}` |",
        f"| Recorded frames | `{len(rows)}` |",
        f"| Adjacent frame pairs | `{max(len(rows) - 1, 0)}` |",
        "| Agent interaction frame | One policy-driven Craftax environment transition per recorded frame; this Craftax wrapper has no action-repeat wrapper. |",
        "| Token set used for rate | Flattened Simulus observation tokens: `vector`, `token` direction, and `token_2d` map. |",
        "| Right-side visualization | `token_2d` map as 9x11 Craftax cells; each cell shows block/item/mob/light token ids. |",
        "",
        "## Change-Rate Summary",
        "",
        "| Metric | Mean | Std | Min | P50 | P90 | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        stat_row("pixel_changed_fraction", rates["pixel_changed_fraction"]),
        stat_row("pixel_mean_abs_delta_norm", rates["pixel_mean_abs_delta_norm"]),
        stat_row("token_change_all", rates["token_change_all"]),
        stat_row("token_change_vector", rates["token_change_vector"]),
        stat_row("token_change_direction", rates["token_change_direction"]),
        stat_row("token_change_token_2d_map", rates["token_change_token_2d_map"]),
        "",
        "## Highest Token-Change Frames",
        "",
        "| Frame | Env Step | Action | Return | Token Change | Pixel Change |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in top_token:
        lines.append(
            f"| {row['frame_index']} | {row['env_step']} | {row['action']} | {row['return']} | "
            f"{float(row['token_change_current']):.6f} | {float(row['pixel_change_current']):.6f} |"
        )

    lines.extend([
        "",
        "## Highest Pixel-Change Frames",
        "",
        "| Frame | Env Step | Action | Return | Pixel Change | Token Change |",
        "|---:|---:|---|---:|---:|---:|",
    ])
    for row in top_pixel:
        lines.append(
            f"| {row['frame_index']} | {row['env_step']} | {row['action']} | {row['return']} | "
            f"{float(row['pixel_change_current']):.6f} | {float(row['token_change_current']):.6f} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- `pixel_changed_fraction` is the fraction of rendered RGB pixels whose value changed from the previous recorded agent-observation frame.",
        "- `pixel_mean_abs_delta_norm` is the mean absolute RGB delta divided by 255.",
        "- `token_change_all` is the fraction of flattened Simulus observation token ids that changed from the previous recorded agent-observation frame.",
        "- The first frame has no previous adjacent frame and is excluded from all summary statistics.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_pixel_render_size() -> int:
    return max(1, 64 // load_craftax_block_pixel_size())


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive.")

    easimulus_dir = args.easimulus_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    video_path = output_dir / args.video_name
    report_path = output_dir / "report.md"
    csv_path = output_dir / "frame_stats.csv"
    num_frames = args.seconds * args.fps

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This recorder expects one visible GPU.")

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.chdir(easimulus_dir)
    patch_gymnax_discrete_space_conversion()
    set_seed(args.seed)
    cfg = build_cfg(easimulus_dir, args.seed)

    print(f"[craftax-token-vis] checkpoint={checkpoint}")
    print(f"[craftax-token-vis] output_dir={output_dir}")
    print(f"[craftax-token-vis] video={video_path}")
    print(f"[craftax-token-vis] seconds={args.seconds}, fps={args.fps}, frames={num_frames}")
    print(f"[craftax-token-vis] cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[craftax-token-vis] torch={torch.__version__}, cuda={torch.version.cuda}, device={torch.cuda.get_device_name(0)}")
    print("[craftax-token-vis] loading checkpoint fields: tokenizer=False, world_model=False, actor_critic=True")

    env_partial = instantiate(cfg.env.test)
    env_fn = lambda: env_partial(tokenizer_config=cfg.tokenizer)
    test_env = SingleProcessEnv(env_fn)
    device = torch.device(cfg.common.device)
    agent = build_agent(test_env, cfg, device)
    agent.load(checkpoint, device, load_tokenizer=False, load_world_model=False, load_actor_critic=True)
    agent.eval()

    pixel_render_size = args.pixel_render_size if args.pixel_render_size > 0 else default_pixel_render_size()
    visual_env = CraftaxTokenDemoEnv(agent, test_env, pixel_render_size=pixel_render_size)
    visual_env.reset()

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_video = video_path.with_suffix(f".tmp.{os.getpid()}.mp4")
    rows: list[dict[str, object]] = []
    rates: dict[str, list[float]] = {
        "pixel_changed_fraction": [],
        "pixel_mean_abs_delta_norm": [],
        "token_change_all": [],
        "token_change_vector": [],
        "token_change_direction": [],
        "token_change_token_2d_map": [],
    }
    previous_image: np.ndarray | None = None
    previous_tokens: dict[str, np.ndarray] | None = None
    current_info = {"Timestep": "0", "Action": "RESET", "Return": "0.00"}

    try:
        with imageio.get_writer(tmp_video, fps=args.fps, codec="libx264", quality=8, macro_block_size=16) as writer:
            for frame_idx in range(num_frames):
                real_image = visual_env.render().convert("RGB")
                real_arr = np.asarray(real_image, dtype=np.uint8)
                tokens = get_encoded_tokens(agent, visual_env.obs)
                if "token_2d" not in tokens:
                    raise RuntimeError("Craftax token_2d observation is missing; cannot render map token ids.")

                pixel_current, pixel_abs_current = pixel_change(previous_image, real_arr)
                token_current = token_change(previous_tokens, tokens)
                token_vector_current = token_change(previous_tokens, tokens, "vector")
                token_direction_current = token_change(previous_tokens, tokens, "token")
                token_map_current = token_change(previous_tokens, tokens, "token_2d")

                if pixel_current is not None:
                    rates["pixel_changed_fraction"].append(pixel_current)
                if pixel_abs_current is not None:
                    rates["pixel_mean_abs_delta_norm"].append(pixel_abs_current)
                if token_current is not None:
                    rates["token_change_all"].append(token_current)
                if token_vector_current is not None:
                    rates["token_change_vector"].append(token_vector_current)
                if token_direction_current is not None:
                    rates["token_change_direction"].append(token_direction_current)
                if token_map_current is not None:
                    rates["token_change_token_2d_map"].append(token_map_current)

                direction_arr = tokens.get("token")
                direction_token = int(direction_arr.reshape(-1)[0]) if direction_arr is not None and direction_arr.size else None
                frame = render_frame(
                    frame_index=frame_idx,
                    num_frames=num_frames,
                    real_image=real_image,
                    map_tokens=tokens["token_2d"],
                    direction_token=direction_token,
                    pixel_current=pixel_current,
                    pixel_mean=running_mean(rates["pixel_changed_fraction"]),
                    pixel_abs_current=pixel_abs_current,
                    pixel_abs_mean=running_mean(rates["pixel_mean_abs_delta_norm"]),
                    token_current=token_current,
                    token_mean=running_mean(rates["token_change_all"]),
                    token_map_current=token_map_current,
                    token_vector_current=token_vector_current,
                    info=current_info,
                    cell_size=args.cell_size,
                    panel_height=args.panel_height,
                    fps=args.fps,
                )
                writer.append_data(np.asarray(pad_to_macroblock(frame)))

                rows.append({
                    "frame_index": frame_idx + 1,
                    "env_step": current_info.get("Timestep", "0"),
                    "action": current_info.get("Action", "RESET"),
                    "return": current_info.get("Return", "0.00"),
                    "pixel_change_current": "" if pixel_current is None else pixel_current,
                    "pixel_mean_abs_delta_norm_current": "" if pixel_abs_current is None else pixel_abs_current,
                    "token_change_current": "" if token_current is None else token_current,
                    "token_change_vector_current": "" if token_vector_current is None else token_vector_current,
                    "token_change_direction_current": "" if token_direction_current is None else token_direction_current,
                    "token_change_token_2d_map_current": "" if token_map_current is None else token_map_current,
                    "pixel_change_mean_so_far": running_mean(rates["pixel_changed_fraction"]),
                    "token_change_mean_so_far": running_mean(rates["token_change_all"]),
                })

                if frame_idx % max(args.fps * 10, 1) == 0:
                    print(
                        "[craftax-token-vis] "
                        f"frame={frame_idx + 1}/{num_frames} "
                        f"pixel_current={fmt_rate(pixel_current)} "
                        f"pixel_mean={running_mean(rates['pixel_changed_fraction']):.4f} "
                        f"token_current={fmt_rate(token_current)} "
                        f"token_mean={running_mean(rates['token_change_all']):.4f} "
                        f"return={current_info.get('Return', '0.00')}",
                        flush=True,
                    )

                previous_image = real_arr.copy()
                previous_tokens = {k: v.copy() for k, v in tokens.items()}

                _, _, terminated, truncated, step_info = visual_env.step()
                current_info = {k: str(v) for k, v in step_info.items()}
                if bool(terminated[0]) or bool(truncated[0]):
                    visual_env.reset()
                    current_info = {"Timestep": "0", "Action": "RESET", "Return": "0.00"}

        tmp_video.replace(video_path)
        write_csv(csv_path, rows)
        write_report(report_path, video_path, csv_path, checkpoint, cfg, args, rows, rates)
        print(f"[craftax-token-vis] saved video={video_path}")
        print(f"[craftax-token-vis] saved report={report_path}")
        print(f"[craftax-token-vis] saved csv={csv_path}")
        print(f"[craftax-token-vis] final token_change_mean={running_mean(rates['token_change_all']):.6f}")
        print(f"[craftax-token-vis] final pixel_change_mean={running_mean(rates['pixel_changed_fraction']):.6f}")
    finally:
        test_env.close()
        if tmp_video.exists():
            tmp_video.unlink()


if __name__ == "__main__":
    main()
