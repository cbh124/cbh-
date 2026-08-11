"""Train and evaluate a conditional latent diffusion maneuver generator.

The denoiser generates null-space coefficients of a cubic B-spline radius
profile.  The decoder enforces path length and endpoint constraints exactly,
then the shared traditional-project pipeline applies time parameterization,
attitude construction, inverse dynamics, and VLM aerodynamic screening.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import null_space


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
OUTPUTS_DIR = PROJECT_DIR.parent
CONFIG_PATH = PROJECT_DIR / "配置" / "diffusion_config.json"
RESULTS_DIR = PROJECT_DIR / "结果"
DATA_DIR = RESULTS_DIR / "训练数据"
MODEL_DIR = RESULTS_DIR / "模型权重"
LOG_DIR = RESULTS_DIR / "训练记录"
GENERATED_DIR = RESULTS_DIR / "生成轨迹"
COMPARISON_DIR = RESULTS_DIR / "对比评估"
MODEL_PATH = MODEL_DIR / "maneuver_latent_diffusion_best.pt"
DATA_PATH = DATA_DIR / "maneuver_experts_v1.npz"
DIFFUSION_LABEL = "条件潜空间扩散模型"
QUINTIC_LABEL = "五次多项式方法"
BSPLINE_LABEL = "B样条约束优化方法"
ACTION_KEYS = ("loop_360", "immelmann")
ACTION_DIRS = {"loop_360": "360度筋斗", "immelmann": "殷麦曼机动"}

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "#FAFAF8"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(json_ready(value), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_context(config_path: Path = CONFIG_PATH):
    diffusion_config = load_json(config_path)
    traditional_project = (
        config_path.parent / diffusion_config["traditional_project"]
    ).resolve()
    traditional_script = (
        traditional_project / "代码" / "run_dual_traditional_methods.py"
    )
    traditional = load_module("shared_traditional_core", traditional_script)
    traditional.CONFIG_PATH = (
        config_path.parent / diffusion_config["traditional_config"]
    ).resolve()
    traditional_config, aircraft, database = traditional.load_project()
    return diffusion_config, traditional_config, aircraft, database, traditional


def condition_vector(
    action_key: str,
    target_length: float,
    target_height: float,
    settings: dict[str, Any],
) -> np.ndarray:
    nominal_length = float(settings["target_path_length_m"])
    nominal_height = float(settings["target_terminal_local_position_m"][1])
    return np.asarray(
        [
            1.0 if action_key == "loop_360" else 0.0,
            1.0 if action_key == "immelmann" else 0.0,
            (target_length - nominal_length) / 8.0,
            (target_height - nominal_height) / 8.0,
        ],
        dtype=np.float32,
    )


def build_decoder_context(
    action_key: str,
    diffusion_config: dict[str, Any],
    traditional_config: dict[str, Any],
    traditional,
) -> dict[str, Any]:
    settings = traditional_config["maneuvers"][action_key]
    angle_end = math.radians(float(settings["terminal_path_rotation_deg"]))
    decoder = diffusion_config["latent_decoder"]
    dense_count = int(
        decoder[
            "dense_training_samples_full_loop"
            if action_key == "loop_360"
            else "dense_training_samples_immelmann"
        ]
    )
    theta = np.linspace(0.0, angle_end, dense_count)
    node_count = int(decoder["radius_node_count"])
    node_angles, build_spline = traditional.spline_factory(
        action_key,
        angle_end,
        node_count,
    )
    basis = traditional.basis_matrix(build_spline, node_count, theta)
    basis_d1 = traditional.basis_matrix(build_spline, node_count, theta, 1)
    weights = traditional.trapz_weights(theta)
    equality = np.vstack(
        (
            weights @ basis,
            (weights * np.cos(theta)) @ basis,
            (weights * np.sin(theta)) @ basis,
        )
    )
    null_basis = null_space(equality)
    if null_basis.shape != (node_count, node_count - 3):
        raise RuntimeError(f"{action_key}约束零空间维数异常: {null_basis.shape}")
    projection = equality.T @ np.linalg.inv(equality @ equality.T)
    return {
        "action_key": action_key,
        "settings": settings,
        "angle_end": angle_end,
        "theta": theta,
        "node_angles": node_angles,
        "build_spline": build_spline,
        "basis": basis,
        "basis_d1": basis_d1,
        "equality": equality,
        "projection": projection,
        "null_basis": null_basis,
    }


def equality_target(target_length: float, target_height: float) -> np.ndarray:
    return np.asarray([target_length, 0.0, target_height], dtype=float)


def particular_nodes(
    context: dict[str, Any],
    target_length: float,
    target_height: float,
) -> np.ndarray:
    base = np.full(
        context["equality"].shape[1],
        target_length / context["angle_end"],
        dtype=float,
    )
    target = equality_target(target_length, target_height)
    return base + context["projection"] @ (
        target - context["equality"] @ base
    )


def smooth_random_perturbation(
    rng: np.random.Generator,
    context: dict[str, Any],
    amplitude: float,
) -> np.ndarray:
    nodes = context["node_angles"]
    phase = nodes / context["angle_end"]
    raw = np.zeros_like(phase)
    maximum_mode = 4 if context["action_key"] == "loop_360" else 3
    for mode in range(1, maximum_mode + 1):
        scale = 1.0 / mode**1.65
        raw += scale * rng.normal() * np.cos(2.0 * math.pi * mode * phase)
        raw += scale * rng.normal() * np.sin(2.0 * math.pi * mode * phase)
    raw -= np.mean(raw)
    raw_norm = float(np.sqrt(np.mean(raw**2)))
    if raw_norm < 1e-9:
        return smooth_random_perturbation(rng, context, amplitude)
    raw *= amplitude / raw_norm
    equality = context["equality"]
    perturbation = raw - context["projection"] @ (equality @ raw)
    perturbation_norm = float(np.sqrt(np.mean(perturbation**2)))
    if perturbation_norm < 1e-9:
        return smooth_random_perturbation(rng, context, amplitude)
    return perturbation * (amplitude / perturbation_norm)


def quick_profile_metrics(
    nodes: np.ndarray,
    context: dict[str, Any],
    traditional_config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
    traditional,
) -> dict[str, float]:
    theta = context["theta"]
    radius = context["basis"] @ nodes
    speed = traditional.speed_profile(theta, traditional_config)
    speed_dtheta = traditional.speed_derivative_theta(theta, traditional_config)
    theta_rate = speed / radius
    time_values = traditional.cumulative_trapezoid(radius / speed, theta)
    tangent = np.column_stack(
        (np.cos(theta), np.zeros_like(theta), np.sin(theta))
    )
    normal = np.column_stack(
        (-np.sin(theta), np.zeros_like(theta), np.cos(theta))
    )
    tangential_acceleration = speed_dtheta * theta_rate
    normal_acceleration = speed**2 / radius
    acceleration = (
        tangent * tangential_acceleration[:, None]
        + normal * normal_acceleration[:, None]
    )
    jerk = np.gradient(acceleration, time_values, axis=0, edge_order=2)
    jerk_rms = float(np.sqrt(np.mean(np.linalg.norm(jerk, axis=1) ** 2)))
    curvature = 1.0 / radius
    curvature_rate = (
        -(context["basis_d1"] @ nodes) / radius**2
    ) * theta_rate
    curvature_rate_rms = float(np.sqrt(np.mean(curvature_rate**2)))

    gravity = float(aircraft["atmosphere"]["gravity_m_s2"])
    mass = float(aircraft["confirmed_parameters"]["mass_kg"])
    area = float(aircraft["confirmed_parameters"]["geometry_m"]["wing_area"])
    density = float(aircraft["atmosphere"]["density_kg_m3"])
    normal_specific = np.abs(normal_acceleration + gravity * np.cos(theta))
    load = normal_specific / gravity
    dynamic_pressure = 0.5 * density * speed**2
    normal_coefficient = mass * normal_specific / (dynamic_pressure * area)
    alpha_limit = float(traditional_config["evaluation"]["lift_limit_alpha_deg"])
    lift_limit = float(database.sample("CZ", math.radians(alpha_limit), 0.0))
    speed_margin = float(traditional_config["evaluation"]["speed_margin_factor"])
    lift_margin = lift_limit / np.maximum(normal_coefficient, 1e-9) / speed_margin**2
    return {
        "minimum_radius_m": float(np.min(radius)),
        "maximum_radius_m": float(np.max(radius)),
        "maximum_load_g": float(np.max(load)),
        "minimum_lift_margin": float(np.min(lift_margin)),
        "maximum_pitch_rate_degps": float(np.max(np.degrees(theta_rate))),
        "rms_jerk_mps3": jerk_rms,
        "rms_curvature_rate_1pm_s": curvature_rate_rms,
    }


def expert_is_acceptable(
    metrics: dict[str, float],
    diffusion_config: dict[str, Any],
    traditional_config: dict[str, Any],
) -> bool:
    decoder = diffusion_config["latent_decoder"]
    dynamic = traditional_config["evaluation"][
        "preliminary_dynamic_screening_thresholds"
    ]
    return bool(
        metrics["minimum_radius_m"] >= float(decoder["radius_lower_bound_m"])
        and metrics["maximum_radius_m"] <= float(decoder["radius_upper_bound_m"])
        and metrics["maximum_load_g"]
        <= float(dynamic["maximum_absolute_load_factor"])
        and metrics["minimum_lift_margin"]
        >= float(dynamic["minimum_lift_margin_ratio_with_speed_margin"])
        and metrics["maximum_pitch_rate_degps"]
        <= float(dynamic["maximum_pitch_rate_degps"])
        and metrics["rms_jerk_mps3"]
        <= float(decoder["expert_jerk_rms_limit_mps3"])
        and metrics["rms_curvature_rate_1pm_s"]
        <= float(decoder["expert_curvature_rate_rms_limit_1pm_s"])
    )


def build_expert_dataset(
    diffusion_config: dict[str, Any],
    traditional_config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
    traditional,
    contexts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    settings = diffusion_config["diffusion"]
    decoder = diffusion_config["latent_decoder"]
    total_size = int(settings["dataset_size"])
    rng = np.random.default_rng(int(settings["dataset_seed"]))
    raw_latents: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    action_ids: list[int] = []
    target_lengths: list[float] = []
    target_heights: list[float] = []
    expert_metrics: list[list[float]] = []
    attempts = {key: 0 for key in ACTION_KEYS}
    accepted = {key: 0 for key in ACTION_KEYS}
    per_action = total_size // len(ACTION_KEYS)

    for action_id, action_key in enumerate(ACTION_KEYS):
        context = contexts[action_key]
        action_settings = context["settings"]
        nominal_length = float(action_settings["target_path_length_m"])
        nominal_height = float(
            action_settings["target_terminal_local_position_m"][1]
        )
        while accepted[action_key] < per_action:
            attempts[action_key] += 1
            if attempts[action_key] > per_action * 100:
                raise RuntimeError(f"{action_key}专家样本接受率过低")
            length_ratio = rng.uniform(
                1.0 - float(decoder["target_length_variation_ratio"]),
                1.0 + float(decoder["target_length_variation_ratio"]),
            )
            target_length = nominal_length * length_ratio
            target_height = nominal_height
            if action_key == "immelmann":
                target_height += rng.uniform(
                    -float(decoder["immelmann_height_variation_m"]),
                    float(decoder["immelmann_height_variation_m"]),
                )
            center = particular_nodes(context, target_length, target_height)
            amplitude = rng.uniform(
                float(decoder["shape_amplitude_min_m"]),
                float(decoder["shape_amplitude_max_m"]),
            )
            perturbation = smooth_random_perturbation(rng, context, amplitude)
            nodes = center + perturbation
            metrics = quick_profile_metrics(
                nodes,
                context,
                traditional_config,
                aircraft,
                database,
                traditional,
            )
            if not expert_is_acceptable(
                metrics,
                diffusion_config,
                traditional_config,
            ):
                continue
            latent = context["null_basis"].T @ (nodes - center)
            raw_latents.append(latent.astype(np.float32))
            conditions.append(
                condition_vector(
                    action_key,
                    target_length,
                    target_height,
                    action_settings,
                )
            )
            action_ids.append(action_id)
            target_lengths.append(target_length)
            target_heights.append(target_height)
            expert_metrics.append(
                [
                    metrics["rms_jerk_mps3"],
                    metrics["rms_curvature_rate_1pm_s"],
                    metrics["minimum_lift_margin"],
                ]
            )
            accepted[action_key] += 1
            if accepted[action_key] % 500 == 0:
                print(
                    f"[数据] {ACTION_DIRS[action_key]} "
                    f"{accepted[action_key]}/{per_action}"
                )

    raw_latent = np.stack(raw_latents)
    latent_mean = raw_latent.mean(axis=0)
    latent_std = np.maximum(raw_latent.std(axis=0), 1e-4)
    normalized = (raw_latent - latent_mean) / latent_std
    summary = {
        "dataset_size": len(raw_latent),
        "latent_dimension": int(raw_latent.shape[1]),
        "condition_dimension": int(np.stack(conditions).shape[1]),
        "accepted_by_action": accepted,
        "attempts_by_action": attempts,
        "acceptance_rate_by_action": {
            key: accepted[key] / attempts[key] for key in ACTION_KEYS
        },
        "latent_mean": latent_mean.tolist(),
        "latent_std": latent_std.tolist(),
        "expert_rms_jerk_mean_mps3": float(np.mean(np.asarray(expert_metrics)[:, 0])),
        "expert_curvature_rate_mean_1pm_s": float(
            np.mean(np.asarray(expert_metrics)[:, 1])
        ),
        "expert_minimum_lift_margin": float(
            np.min(np.asarray(expert_metrics)[:, 2])
        ),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_PATH,
        latent=normalized.astype(np.float32),
        raw_latent=raw_latent.astype(np.float32),
        condition=np.stack(conditions).astype(np.float32),
        action_id=np.asarray(action_ids, dtype=np.int64),
        target_length=np.asarray(target_lengths, dtype=np.float32),
        target_height=np.asarray(target_heights, dtype=np.float32),
        latent_mean=latent_mean.astype(np.float32),
        latent_std=latent_std.astype(np.float32),
        summary=json.dumps(summary, ensure_ascii=False),
    )
    write_json(DATA_DIR / "dataset_summary.json", summary)
    return summary


class DiffusionSchedule:
    def __init__(self, steps: int, device: torch.device):
        offset = 0.008
        grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
        alpha_bar = torch.cos(
            ((grid + offset) / (1.0 + offset)) * math.pi * 0.5
        ) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        beta = torch.clamp(beta, 1e-5, 0.999)
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.steps = steps

    def add_noise(
        self,
        clean: torch.Tensor,
        step: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        cumulative = self.alpha_bar[step].view(-1, 1)
        return torch.sqrt(cumulative) * clean + torch.sqrt(
            1.0 - cumulative
        ) * noise


def time_embedding(step: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=step.device)
        / max(half - 1, 1)
    )
    angles = step.float()[:, None] * frequency[None]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.normalization = nn.LayerNorm(width)
        self.linear_1 = nn.Linear(width, width * 2)
        self.linear_2 = nn.Linear(width * 2, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.normalization(values)
        residual = F.silu(self.linear_1(residual))
        residual = self.linear_2(residual)
        return values + residual


class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        condition_dimension: int,
        width: int,
        block_count: int,
    ):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.condition_dimension = condition_dimension
        self.width = width
        self.input_projection = nn.Linear(latent_dimension, width)
        self.time_projection = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dimension, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(width) for _ in range(block_count)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, latent_dimension),
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        step: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(noisy_latent)
        hidden += self.time_projection(time_embedding(step, self.width))
        hidden += self.condition_projection(condition)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for current, averaged in zip(model.parameters(), ema_model.parameters()):
            averaged.mul_(decay).add_(current, alpha=1.0 - decay)


@torch.no_grad()
def validation_loss(
    model: ConditionalDenoiser,
    schedule: DiffusionSchedule,
    latent: torch.Tensor,
    condition: torch.Tensor,
) -> float:
    model.eval()
    count = min(512, len(latent))
    clean = latent[:count]
    selected_condition = condition[:count]
    step = torch.randint(0, schedule.steps, (count,), device=latent.device)
    noise = torch.randn_like(clean)
    noisy = schedule.add_noise(clean, step, noise)
    prediction = model(noisy, step, selected_condition)
    return float(F.mse_loss(prediction, noise).item())


def train_model(
    diffusion_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    dataset = np.load(DATA_PATH)
    latent_np = dataset["latent"].astype(np.float32)
    condition_np = dataset["condition"].astype(np.float32)
    latent_mean = dataset["latent_mean"].astype(np.float32)
    latent_std = dataset["latent_std"].astype(np.float32)
    settings = diffusion_config["diffusion"]
    seed = int(settings["training_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(latent_np))
    validation_count = max(
        256,
        int(float(settings["validation_fraction"]) * len(order)),
    )
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    train_latent = torch.from_numpy(latent_np[training_indices]).to(device)
    train_condition = torch.from_numpy(condition_np[training_indices]).to(device)
    validation_latent = torch.from_numpy(latent_np[validation_indices]).to(device)
    validation_condition = torch.from_numpy(condition_np[validation_indices]).to(device)

    model = ConditionalDenoiser(
        latent_np.shape[1],
        condition_np.shape[1],
        int(settings["hidden_width"]),
        int(settings["residual_blocks"]),
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    schedule = DiffusionSchedule(int(settings["diffusion_steps"]), device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=1e-5,
    )
    training_steps = int(settings["train_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        training_steps,
        eta_min=float(settings["learning_rate"]) * 0.05,
    )
    batch_size = int(settings["batch_size"])
    dropout_probability = float(settings["condition_dropout_probability"])
    ema_decay = float(settings["ema_decay"])
    log_interval = int(settings["validation_interval"])
    best_validation = float("inf")
    records: list[dict[str, float | int]] = []
    started = time.perf_counter()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for step_number in range(1, training_steps + 1):
        model.train()
        indices = torch.randint(
            0,
            train_latent.shape[0],
            (batch_size,),
            device=device,
        )
        clean = train_latent[indices]
        condition = train_condition[indices].clone()
        dropped = torch.rand(batch_size, device=device) < dropout_probability
        condition[dropped] = 0.0
        step = torch.randint(
            0,
            schedule.steps,
            (batch_size,),
            device=device,
        )
        noise = torch.randn_like(clean)
        noisy = schedule.add_noise(clean, step, noise)
        prediction = model(noisy, step, condition)
        cumulative = schedule.alpha_bar[step].view(-1, 1)
        predicted_clean = (
            noisy - torch.sqrt(1.0 - cumulative) * prediction
        ) / torch.sqrt(cumulative)
        noise_loss = F.mse_loss(prediction, noise)
        bounds_loss = torch.mean(F.relu(torch.abs(predicted_clean) - 3.2) ** 2)
        loss = noise_loss + 0.0001 * bounds_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        update_ema(model, ema_model, ema_decay)

        if step_number != 1 and step_number % log_interval and step_number != training_steps:
            continue
        current_validation = validation_loss(
            ema_model,
            schedule,
            validation_latent,
            validation_condition,
        )
        record = {
            "step": step_number,
            "train_loss": float(loss.item()),
            "noise_loss": float(noise_loss.item()),
            "bounds_loss": float(bounds_loss.item()),
            "validation_noise_loss": current_validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        records.append(record)
        print(
            f"[训练] {step_number:4d}/{training_steps} "
            f"损失={record['train_loss']:.5f} "
            f"验证={current_validation:.5f}"
        )
        if current_validation < best_validation:
            best_validation = current_validation
            torch.save(
                {
                    "model": ema_model.state_dict(),
                    "latent_dimension": latent_np.shape[1],
                    "condition_dimension": condition_np.shape[1],
                    "hidden_width": int(settings["hidden_width"]),
                    "residual_blocks": int(settings["residual_blocks"]),
                    "diffusion_steps": int(settings["diffusion_steps"]),
                    "latent_mean": latent_mean,
                    "latent_std": latent_std,
                    "training_step": step_number,
                    "best_validation_noise_loss": best_validation,
                },
                MODEL_PATH,
            )

    training_time = time.perf_counter() - started
    log_path = LOG_DIR / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "training_time_s": training_time,
        "best_validation_noise_loss": best_validation,
        "best_training_step": int(torch.load(MODEL_PATH, map_location="cpu")["training_step"]),
        "train_steps": training_steps,
        "batch_size": batch_size,
        "training_samples": int(len(training_indices)),
        "validation_samples": int(len(validation_indices)),
        "device": str(device),
        "checkpoint": str(MODEL_PATH),
        "dataset": str(DATA_PATH),
    }
    write_json(LOG_DIR / "training_summary.json", summary)
    print(
        f"[训练] 完成，用时{training_time:.2f} s，"
        f"最佳验证损失{best_validation:.5f}"
    )
    return summary


def load_trained_model(
    device: torch.device,
) -> tuple[ConditionalDenoiser, DiffusionSchedule, dict[str, Any]]:
    saved = torch.load(MODEL_PATH, map_location=device)
    model = ConditionalDenoiser(
        int(saved["latent_dimension"]),
        int(saved["condition_dimension"]),
        int(saved["hidden_width"]),
        int(saved["residual_blocks"]),
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    schedule = DiffusionSchedule(int(saved["diffusion_steps"]), device)
    return model, schedule, saved


@torch.no_grad()
def sample_latent_candidates(
    model: ConditionalDenoiser,
    schedule: DiffusionSchedule,
    condition: np.ndarray,
    count: int,
    sample_steps: int,
    guidance_scale: float,
    clip_sigma: float,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    torch.manual_seed(seed)
    latent = torch.randn(count, model.latent_dimension, device=device)
    conditional = torch.from_numpy(condition.astype(np.float32)).to(device)
    conditional = conditional[None].expand(count, -1)
    unconditional = torch.zeros_like(conditional)
    steps = np.unique(
        np.linspace(schedule.steps - 1, 0, sample_steps, dtype=int)
    )[::-1]
    trace: list[dict[str, float | int]] = []
    previous_clean = None

    for index, current_step in enumerate(steps):
        step_tensor = torch.full(
            (count,),
            int(current_step),
            dtype=torch.long,
            device=device,
        )
        conditional_noise = model(latent, step_tensor, conditional)
        unconditional_noise = model(latent, step_tensor, unconditional)
        predicted_noise = unconditional_noise + guidance_scale * (
            conditional_noise - unconditional_noise
        )
        cumulative = schedule.alpha_bar[current_step]
        predicted_clean = (
            latent - torch.sqrt(1.0 - cumulative) * predicted_noise
        ) / torch.sqrt(cumulative)
        predicted_clean = torch.clamp(predicted_clean, -clip_sigma, clip_sigma)
        clean_change = 0.0
        if previous_clean is not None:
            clean_change = float(
                torch.linalg.vector_norm(
                    predicted_clean[0] - previous_clean
                ).item()
            )
        trace.append(
            {
                "reverse_index": index,
                "diffusion_step": int(current_step),
                "noisy_latent_norm": float(
                    torch.linalg.vector_norm(latent[0]).item()
                ),
                "predicted_clean_norm": float(
                    torch.linalg.vector_norm(predicted_clean[0]).item()
                ),
                "predicted_clean_change": clean_change,
            }
        )
        previous_clean = predicted_clean[0].clone()
        previous_step = int(steps[index + 1]) if index + 1 < len(steps) else -1
        if previous_step >= 0:
            previous_cumulative = schedule.alpha_bar[previous_step]
            latent = (
                torch.sqrt(previous_cumulative) * predicted_clean
                + torch.sqrt(1.0 - previous_cumulative) * predicted_noise
            )
        else:
            latent = predicted_clean
    return latent.cpu().numpy(), trace


def radius_roughness(nodes: np.ndarray, action_key: str) -> float:
    if action_key == "loop_360":
        extended = np.concatenate((nodes[-1:], nodes, nodes[:1]))
        second = np.diff(extended, n=2)
    else:
        second = np.diff(nodes, n=2)
    return float(np.sqrt(np.mean(second**2)))


def candidate_score(
    metrics: dict[str, Any],
    roughness: float,
    weights: dict[str, Any],
) -> float:
    lift_margin = float(
        metrics["minimum_lift_margin_ratio_with_speed_margin"]
    )
    score = (
        float(weights["rms_jerk"])
        * float(metrics["rms_jerk_mps3"])
        / 8.0
        + float(weights["rms_curvature_rate"])
        * float(metrics["rms_curvature_rate_1pm_s"])
        / 0.006
        + float(weights["maximum_alpha"])
        * float(metrics["maximum_estimated_alpha_required_deg"])
        / 12.0
        + float(weights["maximum_load"])
        * float(metrics["maximum_normal_load_factor"])
        / 2.5
        + float(weights["inverse_lift_margin"])
        / max(lift_margin, 0.1)
        + float(weights["radius_roughness"]) * roughness / 2.0
        + 0.30 * max(1.0 - lift_margin, 0.0) / 0.05
    )
    if not metrics["overall_planning_pass"]:
        failed = sum(
            not value for value in metrics["geometric_checks"].values()
        ) + sum(
            not value
            for value in metrics["dynamic_screening_checks"].values()
        )
        score += 100.0 + 10.0 * failed
    return float(score)


def profile_diversity(profiles: np.ndarray) -> float:
    if len(profiles) < 2:
        return 0.0
    selected = profiles[: min(96, len(profiles))]
    difference = selected[:, None, :] - selected[None, :, :]
    rms = np.sqrt(np.mean(difference**2, axis=2))
    upper = rms[np.triu_indices(len(selected), 1)]
    return float(np.mean(upper))


def evaluate_candidates(
    action_key: str,
    normalized_latents: np.ndarray,
    sampling_time_s: float,
    context: dict[str, Any],
    diffusion_config: dict[str, Any],
    traditional_config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
    traditional,
    checkpoint: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, np.ndarray]:
    settings = context["settings"]
    target_length = float(settings["target_path_length_m"])
    target_height = float(settings["target_terminal_local_position_m"][1])
    center = particular_nodes(context, target_length, target_height)
    mean = np.asarray(checkpoint["latent_mean"], dtype=float)
    std = np.asarray(checkpoint["latent_std"], dtype=float)
    raw_latents = normalized_latents * std[None] + mean[None]
    target = equality_target(target_length, target_height)
    decoder = diffusion_config["latent_decoder"]
    lower = float(decoder["radius_lower_bound_m"])
    upper = float(decoder["radius_upper_bound_m"])
    rows: list[dict[str, Any]] = []
    valid_profiles: list[np.ndarray] = []
    valid_latents: list[np.ndarray] = []
    evaluation_started = time.perf_counter()

    for candidate_index, latent in enumerate(raw_latents):
        nodes = center + context["null_basis"] @ latent
        radius = context["basis"] @ nodes
        residual = context["equality"] @ nodes - target
        roughness = radius_roughness(nodes, action_key)
        row: dict[str, Any] = {
            "candidate_index": candidate_index,
            "minimum_radius_m": float(np.min(radius)),
            "maximum_radius_m": float(np.max(radius)),
            "equality_residual_norm": float(np.linalg.norm(residual)),
            "radius_roughness_m": roughness,
            "decoder_bounds_pass": bool(np.min(radius) >= lower and np.max(radius) <= upper),
        }
        if not row["decoder_bounds_pass"]:
            row.update(
                {
                    "overall_planning_pass": False,
                    "quality_score": 1000.0 + max(lower - np.min(radius), 0.0) + max(np.max(radius) - upper, 0.0),
                }
            )
            rows.append(row)
            continue
        spline = context["build_spline"](nodes)
        frame = traditional.build_frame_from_radius(
            action_key,
            settings,
            traditional_config,
            aircraft,
            database,
            spline,
        )
        metrics = traditional.evaluate_frame(
            DIFFUSION_LABEL,
            action_key,
            frame,
            settings,
            traditional_config,
            aircraft,
            database,
            0.0,
        )
        score = candidate_score(
            metrics,
            roughness,
            diffusion_config["selection_weights"],
        )
        row.update(
            {
                "overall_planning_pass": bool(metrics["overall_planning_pass"]),
                "quality_score": score,
                "rms_jerk_mps3": metrics["rms_jerk_mps3"],
                "rms_curvature_rate_1pm_s": metrics[
                    "rms_curvature_rate_1pm_s"
                ],
                "maximum_load_g": metrics["maximum_normal_load_factor"],
                "maximum_alpha_deg": metrics[
                    "maximum_estimated_alpha_required_deg"
                ],
                "minimum_lift_margin": metrics[
                    "minimum_lift_margin_ratio_with_speed_margin"
                ],
                "maximum_pitch_rate_degps": metrics[
                    "maximum_path_pitch_rate_degps"
                ],
                "maximum_roll_rate_degps": metrics["maximum_roll_rate_degps"],
                "maximum_side_force_coefficient": metrics[
                    "maximum_required_side_force_coefficient"
                ],
            }
        )
        rows.append(row)
        valid_profiles.append(radius)
        valid_latents.append(latent)

    evaluation_time_s = time.perf_counter() - evaluation_started
    candidates = pd.DataFrame(rows)
    passed = candidates[candidates["overall_planning_pass"] == True]
    selection_pool = passed if len(passed) else candidates
    selected_index = int(
        selection_pool.sort_values("quality_score").iloc[0]["candidate_index"]
    )
    selected_latent = raw_latents[selected_index]
    selected_nodes = center + context["null_basis"] @ selected_latent
    selected_spline = context["build_spline"](selected_nodes)
    selected_frame = traditional.build_frame_from_radius(
        action_key,
        settings,
        traditional_config,
        aircraft,
        database,
        selected_spline,
    )
    total_generation_time_s = sampling_time_s + evaluation_time_s
    selected_metrics = traditional.evaluate_frame(
        DIFFUSION_LABEL,
        action_key,
        selected_frame,
        settings,
        traditional_config,
        aircraft,
        database,
        total_generation_time_s,
    )
    profile_array = (
        np.stack(valid_profiles)
        if valid_profiles
        else np.empty((0, len(context["theta"])))
    )
    feasible_scores = passed["quality_score"].to_numpy(dtype=float)
    best_score = float(candidates.loc[candidates["candidate_index"] == selected_index, "quality_score"].iloc[0])
    median_score = float(np.median(feasible_scores)) if len(feasible_scores) else best_score
    diagnostics = {
        "candidate_count": int(len(candidates)),
        "decoder_bounds_pass_rate": float(candidates["decoder_bounds_pass"].mean()),
        "hard_geometry_satisfaction_rate": float(
            (candidates["equality_residual_norm"] <= 1e-8).mean()
        ),
        "planning_feasible_count": int(len(passed)),
        "planning_feasible_rate": float(len(passed) / len(candidates)),
        "radius_profile_diversity_rmse_m": profile_diversity(profile_array),
        "latent_diversity_rms": float(
            np.sqrt(np.mean(np.var(np.stack(valid_latents), axis=0)))
        )
        if len(valid_latents) > 1
        else 0.0,
        "selected_candidate_index": selected_index,
        "selected_quality_score": best_score,
        "median_feasible_quality_score": median_score,
        "best_of_n_selection_gain_ratio": float(
            (median_score - best_score) / max(abs(median_score), 1e-9)
        ),
        "diffusion_sampling_time_s": sampling_time_s,
        "candidate_evaluation_time_s": evaluation_time_s,
        "total_generation_time_s": total_generation_time_s,
        "selected_radius_nodes_m": selected_nodes.tolist(),
        "selected_equality_residual": (
            context["equality"] @ selected_nodes - target
        ).tolist(),
    }
    selected_metrics["diffusion"] = diagnostics
    return selected_frame, selected_metrics, candidates, profile_array


def nominal_condition(
    action_key: str,
    context: dict[str, Any],
) -> np.ndarray:
    settings = context["settings"]
    return condition_vector(
        action_key,
        float(settings["target_path_length_m"]),
        float(settings["target_terminal_local_position_m"][1]),
        settings,
    )


def sample_action(
    action_key: str,
    seed: int,
    candidate_count: int,
    model: ConditionalDenoiser,
    schedule: DiffusionSchedule,
    checkpoint: dict[str, Any],
    context: dict[str, Any],
    diffusion_config: dict[str, Any],
    traditional_config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
    traditional,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    sampling = diffusion_config["sampling"]
    started = time.perf_counter()
    latent, trace = sample_latent_candidates(
        model,
        schedule,
        nominal_condition(action_key, context),
        candidate_count,
        int(sampling["ddim_steps"]),
        float(sampling["guidance_scale"]),
        float(sampling["latent_clip_sigma"]),
        device,
        seed,
    )
    sampling_time_s = time.perf_counter() - started
    frame, metrics, candidates, profiles = evaluate_candidates(
        action_key,
        latent,
        sampling_time_s,
        context,
        diffusion_config,
        traditional_config,
        aircraft,
        database,
        traditional,
        checkpoint,
    )
    return frame, metrics, candidates, profiles, trace


def save_action_result(
    action_key: str,
    frame: pd.DataFrame,
    metrics: dict[str, Any],
    candidates: pd.DataFrame,
    profiles: np.ndarray,
    trace: list[dict[str, Any]],
) -> None:
    output = GENERATED_DIR / ACTION_DIRS[action_key]
    output.mkdir(parents=True, exist_ok=True)
    trajectory = frame.copy()
    trajectory.insert(0, "maneuver", metrics["maneuver_name"])
    trajectory.insert(0, "method", DIFFUSION_LABEL)
    trajectory.to_csv(
        output / "trajectory_attitude.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    candidates.to_csv(
        output / "candidate_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    pd.DataFrame(trace).to_csv(
        output / "denoising_trace.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    np.savez_compressed(output / "candidate_radius_profiles.npz", profiles=profiles)
    write_json(output / "metrics.json", metrics)


def load_traditional_records(
    diffusion_config: dict[str, Any],
) -> dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]]:
    traditional_project = (
        CONFIG_PATH.parent / diffusion_config["traditional_project"]
    ).resolve()
    method_dirs = {
        QUINTIC_LABEL: traditional_project / "结果" / "方法一_五次多项式",
        BSPLINE_LABEL: traditional_project / "结果" / "方法二_B样条约束优化",
    }
    records = {}
    for method, method_dir in method_dirs.items():
        for action_key in ACTION_KEYS:
            action_dir = method_dir / ACTION_DIRS[action_key]
            frame = pd.read_csv(
                action_dir / "trajectory_attitude.csv",
                encoding="utf-8-sig",
            )
            metrics = load_json(action_dir / "metrics.json")
            records[(method, action_key)] = (frame, metrics)
    return records


def metric_summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "方法": metrics["method"],
        "机动动作": metrics["maneuver_name"],
        "生成时间_s": metrics["generation_time_s"],
        "持续时间_s": metrics["duration_s"],
        "轨迹长度_m": metrics["path_length_m"],
        "终端位置误差_m": metrics["terminal_position_error_m"],
        "最小曲率半径_m": metrics["minimum_curvature_radius_m"],
        "最大曲率半径_m": metrics["maximum_curvature_radius_m"],
        "最大过载_g": metrics["maximum_normal_load_factor"],
        "最大迎角需求_deg": metrics["maximum_estimated_alpha_required_deg"],
        "最小含速度裕度": metrics[
            "minimum_lift_margin_ratio_with_speed_margin"
        ],
        "最大俯仰角速度_degps": metrics["maximum_path_pitch_rate_degps"],
        "最大滚转角速度_degps": metrics["maximum_roll_rate_degps"],
        "最大侧向力系数": metrics[
            "maximum_required_side_force_coefficient"
        ],
        "最大jerk_mps3": metrics["maximum_jerk_mps3"],
        "均方根jerk_mps3": metrics["rms_jerk_mps3"],
        "均方根曲率变化率_1pm_s": metrics[
            "rms_curvature_rate_1pm_s"
        ],
        "几何评估": "通过" if metrics["geometric_pass"] else "未通过",
        "动力学筛查": "通过" if metrics["dynamic_screening_pass"] else "未通过",
        "规划级结论": "通过" if metrics["overall_planning_pass"] else "需修正",
    }


def build_all_methods_summary(
    diffusion_config: dict[str, Any],
    diffusion_results: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
) -> pd.DataFrame:
    traditional_project = (
        CONFIG_PATH.parent / diffusion_config["traditional_project"]
    ).resolve()
    traditional_summary = pd.read_csv(
        traditional_project / "结果" / "方法对比" / "comparison_metrics.csv",
        encoding="utf-8-sig",
    )
    diffusion_rows = [
        metric_summary_row(diffusion_results[action_key][1])
        for action_key in ACTION_KEYS
    ]
    return pd.concat(
        [traditional_summary, pd.DataFrame(diffusion_rows)],
        ignore_index=True,
    )


def plot_action_comparison(
    action_key: str,
    diffusion_frame: pd.DataFrame,
    traditional_records: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]],
    path: Path,
) -> None:
    frames = {
        QUINTIC_LABEL: traditional_records[(QUINTIC_LABEL, action_key)][0],
        BSPLINE_LABEL: traditional_records[(BSPLINE_LABEL, action_key)][0],
        DIFFUSION_LABEL: diffusion_frame,
    }
    colors = {
        QUINTIC_LABEL: "#64748B",
        BSPLINE_LABEL: "#0369A1",
        DIFFUSION_LABEL: "#DC2626",
    }
    styles = {QUINTIC_LABEL: "--", BSPLINE_LABEL: "-", DIFFUSION_LABEL: "-"}
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.7), constrained_layout=True)
    for method, frame in frames.items():
        axes[0].plot(
            frame["x"],
            frame["z"],
            color=colors[method],
            linestyle=styles[method],
            linewidth=2.2 if method == DIFFUSION_LABEL else 1.8,
            label=method,
        )
        axes[1].plot(
            frame["path_angle_unwrapped_deg"],
            frame["curvature_radius_m"],
            color=colors[method],
            linestyle=styles[method],
            linewidth=2.0,
            label=method,
        )
        axes[2].plot(
            frame["t"],
            frame["jerk_mps3"],
            color=colors[method],
            linestyle=styles[method],
            linewidth=1.8,
            label=method,
        )
    axes[0].set_title("机动平面轨迹")
    axes[0].set_xlabel("前向位置 x / m")
    axes[0].set_ylabel("高度 z / m")
    axes[0].axis("equal")
    axes[1].axhline(20.0, color="#B91C1C", linestyle=":", linewidth=1.0)
    axes[1].set_title("曲率半径")
    axes[1].set_xlabel("累计航迹转角 / (°)")
    axes[1].set_ylabel("曲率半径 / m")
    axes[2].set_title("加加速度时序")
    axes[2].set_xlabel("时间 / s")
    axes[2].set_ylabel("加加速度 / (m/s³)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=7.8)
    fig.suptitle(
        f"{ACTION_DIRS[action_key]}三种方法几何与平滑性对比",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_dynamics_comparison(
    action_key: str,
    diffusion_frame: pd.DataFrame,
    traditional_records: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]],
    path: Path,
) -> None:
    frames = {
        QUINTIC_LABEL: traditional_records[(QUINTIC_LABEL, action_key)][0],
        BSPLINE_LABEL: traditional_records[(BSPLINE_LABEL, action_key)][0],
        DIFFUSION_LABEL: diffusion_frame,
    }
    colors = {
        QUINTIC_LABEL: "#64748B",
        BSPLINE_LABEL: "#0369A1",
        DIFFUSION_LABEL: "#DC2626",
    }
    styles = {QUINTIC_LABEL: "--", BSPLINE_LABEL: "-", DIFFUSION_LABEL: "-"}
    panels = [
        ("normal_load_factor", "法向过载", "过载 / g", 3.0),
        ("estimated_alpha_required_deg", "估算迎角", "迎角 / (°)", 15.0),
        (
            "lift_margin_ratio_with_speed_margin",
            "计入速度裕度后的升力裕度",
            "裕度比",
            0.95,
        ),
        ("path_pitch_rate_degps", "航迹俯仰角速度", "角速度 / (°/s)", 120.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.5), constrained_layout=True)
    for axis, (column, title, ylabel, threshold) in zip(axes.flat, panels):
        for method, frame in frames.items():
            axis.plot(
                frame["t"],
                frame[column],
                color=colors[method],
                linestyle=styles[method],
                linewidth=1.8,
                label=method,
            )
        axis.axhline(threshold, color="#B91C1C", linestyle=":", linewidth=1.0)
        if column == "lift_margin_ratio_with_speed_margin":
            # Required lift approaches zero near the maneuver boundary, so the
            # ratio can become arbitrarily large. The minimum is the governing
            # criterion; clipping only keeps the informative range readable.
            axis.set_ylim(0.0, 1.25)
            axis.text(
                0.99,
                0.97,
                "大于1.25的裕度仅作截断显示",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.0,
                color="#475569",
            )
        axis.set_title(title)
        axis.set_xlabel("时间 / s")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=7.4)
    fig.suptitle(
        f"{ACTION_DIRS[action_key]}三种方法动力学需求对比",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_diversity(
    action_key: str,
    profiles: np.ndarray,
    selected_frame: pd.DataFrame,
    context: dict[str, Any],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    theta = context["theta"]
    start = np.asarray(context["settings"]["start_position_m"], dtype=float)
    if len(profiles):
        indices = np.linspace(0, len(profiles) - 1, min(36, len(profiles)), dtype=int)
        for index in indices:
            radius = profiles[index]
            x = start[0] + np.concatenate(
                ([0.0], np.cumsum(0.5 * (radius[1:] * np.cos(theta[1:]) + radius[:-1] * np.cos(theta[:-1])) * np.diff(theta)))
            )
            z = start[2] + np.concatenate(
                ([0.0], np.cumsum(0.5 * (radius[1:] * np.sin(theta[1:]) + radius[:-1] * np.sin(theta[:-1])) * np.diff(theta)))
            )
            axes[0].plot(x, z, color="#94A3B8", alpha=0.22, linewidth=0.9)
            axes[1].plot(np.degrees(theta), radius, color="#94A3B8", alpha=0.22, linewidth=0.9)
    axes[0].plot(
        selected_frame["x"],
        selected_frame["z"],
        color="#DC2626",
        linewidth=2.4,
        label="筛选后的输出轨迹",
    )
    axes[1].plot(
        selected_frame["path_angle_unwrapped_deg"],
        selected_frame["curvature_radius_m"],
        color="#DC2626",
        linewidth=2.4,
        label="筛选后的曲率半径",
    )
    axes[0].set_title("解码范围内候选轨迹族")
    axes[0].set_xlabel("前向位置 x / m")
    axes[0].set_ylabel("高度 z / m")
    axes[0].axis("equal")
    axes[1].set_title("候选曲率半径分布")
    axes[1].set_xlabel("累计航迹转角 / (°)")
    axes[1].set_ylabel("曲率半径 / m")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8.5)
    fig.suptitle(
        f"{ACTION_DIRS[action_key]}扩散候选多样性与Best-of-N筛选",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(path: Path) -> None:
    history = pd.read_csv(LOG_DIR / "training_log.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5), constrained_layout=True)
    axes[0].plot(history["step"], history["train_loss"], color="#0369A1", label="训练损失")
    axes[0].plot(
        history["step"],
        history["validation_noise_loss"],
        color="#DC2626",
        label="验证噪声损失",
    )
    axes[0].set_title("扩散噪声预测损失")
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("均方误差")
    axes[0].legend(loc="best")
    axes[1].semilogy(history["step"], history["learning_rate"], color="#0F766E")
    axes[1].set_title("余弦退火学习率")
    axes[1].set_xlabel("训练步数")
    axes[1].set_ylabel("学习率")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("条件潜空间扩散模型训练过程", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_denoising_traces(
    traces: dict[str, list[dict[str, Any]]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5), constrained_layout=True)
    colors = {"loop_360": "#0369A1", "immelmann": "#DC2626"}
    for action_key, trace in traces.items():
        frame = pd.DataFrame(trace)
        axes[0].plot(
            frame["reverse_index"],
            frame["predicted_clean_norm"],
            color=colors[action_key],
            linewidth=2.0,
            label=ACTION_DIRS[action_key],
        )
        axes[1].plot(
            frame["reverse_index"],
            frame["predicted_clean_change"],
            color=colors[action_key],
            linewidth=2.0,
            label=ACTION_DIRS[action_key],
        )
    axes[0].set_title("预测无噪潜变量范数")
    axes[0].set_ylabel("潜变量范数")
    axes[1].set_title("相邻去噪步预测变化")
    axes[1].set_ylabel("变化范数")
    for axis in axes:
        axis.set_xlabel("反向去噪迭代序号")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    fig.suptitle("DDIM反向去噪过程", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_key_metrics(summary: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("最大过载_g", "最大过载 / g"),
        ("最大迎角需求_deg", "最大迎角 / (°)"),
        ("最小含速度裕度", "最小升力裕度"),
        ("均方根jerk_mps3", "均方根加加速度 / (m/s³)"),
        ("均方根曲率变化率_1pm_s", "曲率变化率RMS"),
        ("生成时间_s", "生成时间 / s"),
    ]
    methods = (QUINTIC_LABEL, BSPLINE_LABEL, DIFFUSION_LABEL)
    colors = {
        QUINTIC_LABEL: "#64748B",
        BSPLINE_LABEL: "#0369A1",
        DIFFUSION_LABEL: "#DC2626",
    }
    maneuvers = ["360度筋斗", "殷麦曼机动（半筋斗接半滚转）"]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.8), constrained_layout=True)
    x = np.arange(2)
    width = 0.25
    for axis, (column, title) in zip(axes.flat, metrics):
        for method_index, method in enumerate(methods):
            values = (
                summary[summary["方法"] == method]
                .set_index("机动动作")
                .loc[maneuvers, column]
                .astype(float)
                .to_numpy()
            )
            bars = axis.bar(
                x + (method_index - 1) * width,
                values,
                width,
                color=colors[method],
                label=method,
            )
            axis.bar_label(bars, fmt="%.3g", padding=2, fontsize=7.2)
        axis.set_title(title)
        axis.set_xticks(x, ["筋斗", "殷麦曼"])
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.025))
    fig.suptitle("扩散模型与两种传统方法关键指标对比", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def run_robustness_evaluation(
    model: ConditionalDenoiser,
    schedule: DiffusionSchedule,
    checkpoint: dict[str, Any],
    contexts: dict[str, dict[str, Any]],
    diffusion_config: dict[str, Any],
    traditional_config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
    traditional,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    sampling = diffusion_config["sampling"]
    count = int(sampling["robustness_candidates_per_seed"])
    for action_key in ACTION_KEYS:
        for seed in sampling["robustness_seeds"]:
            actual_seed = int(seed) + (10000 if action_key == "immelmann" else 0)
            _, metrics, _, _, _ = sample_action(
                action_key,
                actual_seed,
                count,
                model,
                schedule,
                checkpoint,
                contexts[action_key],
                diffusion_config,
                traditional_config,
                aircraft,
                database,
                traditional,
                device,
            )
            diagnostics = metrics["diffusion"]
            rows.append(
                {
                    "maneuver_key": action_key,
                    "maneuver": metrics["maneuver_name"],
                    "seed": int(seed),
                    "seed_success": bool(metrics["overall_planning_pass"]),
                    "planning_feasible_rate": diagnostics[
                        "planning_feasible_rate"
                    ],
                    "hard_geometry_satisfaction_rate": diagnostics[
                        "hard_geometry_satisfaction_rate"
                    ],
                    "radius_profile_diversity_rmse_m": diagnostics[
                        "radius_profile_diversity_rmse_m"
                    ],
                    "best_of_n_selection_gain_ratio": diagnostics[
                        "best_of_n_selection_gain_ratio"
                    ],
                    "selected_quality_score": diagnostics[
                        "selected_quality_score"
                    ],
                    "generation_time_s": metrics["generation_time_s"],
                    "rms_jerk_mps3": metrics["rms_jerk_mps3"],
                    "minimum_lift_margin": metrics[
                        "minimum_lift_margin_ratio_with_speed_margin"
                    ],
                    "maximum_load_g": metrics["maximum_normal_load_factor"],
                }
            )
            print(
                f"[鲁棒性] {ACTION_DIRS[action_key]} seed={seed} "
                f"可行率={diagnostics['planning_feasible_rate']:.1%}"
            )
    robustness = pd.DataFrame(rows)
    robustness.to_csv(
        COMPARISON_DIR / "seed_robustness.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    aggregate = []
    for action_key in ACTION_KEYS:
        subset = robustness[robustness["maneuver_key"] == action_key]
        aggregate.append(
            {
                "maneuver_key": action_key,
                "maneuver": ACTION_DIRS[action_key],
                "seed_success_rate": float(subset["seed_success"].mean()),
                "mean_planning_feasible_rate": float(
                    subset["planning_feasible_rate"].mean()
                ),
                "minimum_planning_feasible_rate": float(
                    subset["planning_feasible_rate"].min()
                ),
                "mean_radius_diversity_rmse_m": float(
                    subset["radius_profile_diversity_rmse_m"].mean()
                ),
                "mean_best_of_n_gain_ratio": float(
                    subset["best_of_n_selection_gain_ratio"].mean()
                ),
                "selected_score_coefficient_of_variation": float(
                    subset["selected_quality_score"].std(ddof=0)
                    / max(subset["selected_quality_score"].mean(), 1e-9)
                ),
                "mean_generation_time_s": float(
                    subset["generation_time_s"].mean()
                ),
            }
        )
    pd.DataFrame(aggregate).to_csv(
        COMPARISON_DIR / "seed_robustness_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    return robustness


def plot_robustness(robustness: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), constrained_layout=True)
    colors = {"loop_360": "#0369A1", "immelmann": "#DC2626"}
    for action_key in ACTION_KEYS:
        subset = robustness[robustness["maneuver_key"] == action_key]
        label = ACTION_DIRS[action_key]
        axes[0].plot(
            subset["seed"],
            100.0 * subset["planning_feasible_rate"],
            marker="o",
            color=colors[action_key],
            label=label,
        )
        axes[1].plot(
            subset["seed"],
            subset["radius_profile_diversity_rmse_m"],
            marker="o",
            color=colors[action_key],
            label=label,
        )
        axes[2].plot(
            subset["seed"],
            subset["selected_quality_score"],
            marker="o",
            color=colors[action_key],
            label=label,
        )
    axes[0].set_title("规划级可行候选率")
    axes[0].set_ylabel("可行率 / %")
    axes[1].set_title("候选曲率多样性")
    axes[1].set_ylabel("两两RMS差异 / m")
    axes[2].set_title("Best-of-N输出质量分数")
    axes[2].set_ylabel("综合分数（越低越好）")
    for axis in axes:
        axis.set_xlabel("随机种子")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8.5)
    fig.suptitle("扩散采样随机种子鲁棒性", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def write_comparison_conclusion(
    summary: pd.DataFrame,
    diffusion_results: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
    robustness: pd.DataFrame | None,
) -> None:
    indexed = summary.set_index(["方法", "机动动作"])
    lines = [
        "# 扩散模型测试结论",
        "",
        "条件潜空间扩散模型已使用与传统方法相同的飞机参数、动作边界、速度规律和规划级动力学筛查口径，完成360度筋斗与殷麦曼机动生成。",
        "扩散模型的主要增量能力不是单次确定性求解，而是一次生成多条满足硬边界的候选轨迹，并通过物理约束进行Best-of-N筛选。",
        "",
    ]
    for action_key in ACTION_KEYS:
        name = diffusion_results[action_key][1]["maneuver_name"]
        diffusion = indexed.loc[(DIFFUSION_LABEL, name)]
        bspline = indexed.loc[(BSPLINE_LABEL, name)]
        jerk_change = 100.0 * (
            float(diffusion["均方根jerk_mps3"])
            / float(bspline["均方根jerk_mps3"])
            - 1.0
        )
        curvature_change = 100.0 * (
            float(diffusion["均方根曲率变化率_1pm_s"])
            / float(bspline["均方根曲率变化率_1pm_s"])
            - 1.0
        )
        diagnostics = diffusion_results[action_key][1]["diffusion"]
        lines.extend(
            [
                f"## {name}",
                "",
                (
                    f"主采样批次共生成{diagnostics['candidate_count']}条候选，规划级可行率为"
                    f"{diagnostics['planning_feasible_rate']:.1%}，曲率半径候选多样性为"
                    f"{diagnostics['radius_profile_diversity_rmse_m']:.3f} m。"
                    f"相对B样条基线，最终扩散轨迹的均方根jerk变化{jerk_change:+.2f}%，"
                    f"均方根曲率变化率变化{curvature_change:+.2f}%。"
                ),
                "",
            ]
        )
    if robustness is not None:
        for action_key in ACTION_KEYS:
            subset = robustness[robustness["maneuver_key"] == action_key]
            lines.append(
                f"{ACTION_DIRS[action_key]}在{len(subset)}个随机种子下的输出成功率为"
                f"{subset['seed_success'].mean():.1%}，平均候选可行率为"
                f"{subset['planning_feasible_rate'].mean():.1%}。"
            )
    lines.extend(
        [
            "",
            "扩散模型能够表达同一任务下的多种可行曲率分配，并利用批量采样提高获得高质量轨迹的概率；代价是训练成本、推理时间和结果随机性均高于确定性传统方法。",
            "当前动力学结论仍属于参考轨迹逆动力学与VLM附着流气动表支持下的规划级筛查，不能替代六自由度闭环跟踪、执行机构约束和实飞安全验证。",
            "",
        ]
    )
    (COMPARISON_DIR / "comparison_conclusion.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--train-steps", type=int)
    args = parser.parse_args()

    (
        diffusion_config,
        traditional_config,
        aircraft,
        database,
        traditional,
    ) = load_context()
    if args.dataset_size:
        diffusion_config["diffusion"]["dataset_size"] = args.dataset_size
    if args.train_steps:
        diffusion_config["diffusion"]["train_steps"] = args.train_steps
    contexts = {
        action_key: build_decoder_context(
            action_key,
            diffusion_config,
            traditional_config,
            traditional,
        )
        for action_key in ACTION_KEYS
    }
    for directory in (DATA_DIR, MODEL_DIR, LOG_DIR, GENERATED_DIR, COMPARISON_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(10, torch.get_num_threads())))
    print(f"[环境] 使用设备: {device}")

    if args.rebuild_data or not DATA_PATH.exists():
        data_summary = build_expert_dataset(
            diffusion_config,
            traditional_config,
            aircraft,
            database,
            traditional,
            contexts,
        )
    else:
        cached = np.load(DATA_PATH)
        data_summary = json.loads(str(cached["summary"].item()))
        print(f"[数据] 复用训练集: {DATA_PATH}")

    if args.retrain or not MODEL_PATH.exists():
        training_summary = train_model(diffusion_config, device)
    else:
        training_summary = load_json(LOG_DIR / "training_summary.json")
        print(f"[训练] 复用模型: {MODEL_PATH}")

    model, schedule, checkpoint = load_trained_model(device)
    main_seed = int(diffusion_config["sampling"]["main_seed"])
    candidate_count = int(diffusion_config["sampling"]["candidate_count"])
    diffusion_results: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    profiles_by_action: dict[str, np.ndarray] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    for action_key in ACTION_KEYS:
        actual_seed = main_seed + (10000 if action_key == "immelmann" else 0)
        frame, metrics, candidates, profiles, trace = sample_action(
            action_key,
            actual_seed,
            candidate_count,
            model,
            schedule,
            checkpoint,
            contexts[action_key],
            diffusion_config,
            traditional_config,
            aircraft,
            database,
            traditional,
            device,
        )
        if not metrics["overall_planning_pass"]:
            raise RuntimeError(f"{ACTION_DIRS[action_key]}未获得规划级可行扩散候选")
        save_action_result(action_key, frame, metrics, candidates, profiles, trace)
        diffusion_results[action_key] = (frame, metrics)
        profiles_by_action[action_key] = profiles
        traces[action_key] = trace
        print(
            f"[生成] {ACTION_DIRS[action_key]} 可行率="
            f"{metrics['diffusion']['planning_feasible_rate']:.1%}，"
            f"选中候选={metrics['diffusion']['selected_candidate_index']}"
        )

    traditional_records = load_traditional_records(diffusion_config)
    summary = build_all_methods_summary(diffusion_config, diffusion_results)
    summary.to_csv(
        COMPARISON_DIR / "all_methods_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    unique_rows = []
    for action_key in ACTION_KEYS:
        metrics = diffusion_results[action_key][1]
        diagnostics = metrics["diffusion"]
        unique_rows.append(
            {
                "机动动作": metrics["maneuver_name"],
                "候选数量": diagnostics["candidate_count"],
                "硬边界满足率": diagnostics["hard_geometry_satisfaction_rate"],
                "规划级可行率": diagnostics["planning_feasible_rate"],
                "曲率候选多样性_RMS_m": diagnostics[
                    "radius_profile_diversity_rmse_m"
                ],
                "潜变量多样性_RMS": diagnostics["latent_diversity_rms"],
                "Best_of_N筛选收益": diagnostics[
                    "best_of_n_selection_gain_ratio"
                ],
                "扩散采样时间_s": diagnostics["diffusion_sampling_time_s"],
                "候选评价时间_s": diagnostics["candidate_evaluation_time_s"],
            }
        )
    pd.DataFrame(unique_rows).to_csv(
        COMPARISON_DIR / "diffusion_unique_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    plot_training_history(COMPARISON_DIR / "training_history.png")
    plot_denoising_traces(traces, COMPARISON_DIR / "denoising_process.png")
    for action_key in ACTION_KEYS:
        plot_action_comparison(
            action_key,
            diffusion_results[action_key][0],
            traditional_records,
            COMPARISON_DIR / f"{action_key}_geometry_comparison.png",
        )
        plot_dynamics_comparison(
            action_key,
            diffusion_results[action_key][0],
            traditional_records,
            COMPARISON_DIR / f"{action_key}_dynamics_comparison.png",
        )
        plot_candidate_diversity(
            action_key,
            profiles_by_action[action_key],
            diffusion_results[action_key][0],
            contexts[action_key],
            COMPARISON_DIR / f"{action_key}_candidate_diversity.png",
        )
    plot_key_metrics(summary, COMPARISON_DIR / "key_metrics_comparison.png")

    robustness = None
    if not args.skip_robustness:
        robustness = run_robustness_evaluation(
            model,
            schedule,
            checkpoint,
            contexts,
            diffusion_config,
            traditional_config,
            aircraft,
            database,
            traditional,
            device,
        )
        plot_robustness(robustness, COMPARISON_DIR / "seed_robustness.png")
    write_comparison_conclusion(summary, diffusion_results, robustness)
    run_summary = {
        "configuration": str(CONFIG_PATH),
        "device": str(device),
        "data_summary": data_summary,
        "training_summary": training_summary,
        "checkpoint": {
            "training_step": int(checkpoint["training_step"]),
            "best_validation_noise_loss": float(
                checkpoint["best_validation_noise_loss"]
            ),
        },
        "actions": {
            action_key: diffusion_results[action_key][1]
            for action_key in ACTION_KEYS
        },
    }
    write_json(COMPARISON_DIR / "run_summary.json", run_summary)
    print("\n" + summary.to_string(index=False))
    print(f"\n结果目录: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
