"""Compare quintic-polynomial trajectories with B-spline optimization.

The script regenerates both traditional methods using the same aircraft data,
speed law, output interval, maneuver boundaries, and screening thresholds.
Results are planning-level references and are not flight-release evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
OUTPUTS_DIR = PROJECT_DIR.parent
CONFIG_PATH = PROJECT_DIR / "配置" / "comparison_config.json"
RESULTS_DIR = PROJECT_DIR / "结果"
QUINTIC_LABEL = "五次多项式方法"
BSPLINE_LABEL = "B样条约束优化方法"
METHOD_DIRS = {
    QUINTIC_LABEL: RESULTS_DIR / "方法一_五次多项式",
    BSPLINE_LABEL: RESULTS_DIR / "方法二_B样条约束优化",
}
MANEUVER_DIRS = {
    "loop_360": "360度筋斗",
    "immelmann": "殷麦曼机动",
}

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "#FAFAF8"
plt.rcParams["axes.edgecolor"] = "#475569"
plt.rcParams["axes.titleweight"] = "bold"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(value), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_project():
    config = load_json(CONFIG_PATH)
    aircraft_path = (
        CONFIG_PATH.parent / config["aircraft_parameters_file"]
    ).resolve()
    aircraft = load_json(aircraft_path)
    aero_dir = aircraft_path.parent
    if str(aero_dir) not in sys.path:
        sys.path.insert(0, str(aero_dir))
    from aerodynamic_database import AerodynamicDatabase

    database = AerodynamicDatabase(aircraft["aerodynamic_database"]["mat_file"])
    return config, aircraft, database


def cumulative_trapezoid(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(coordinate)
    return np.concatenate(([0.0], np.cumsum(increments)))


def quintic_smoothstep(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def speed_profile(theta: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    bottom = float(config["speed_profile"]["bottom_speed_mps"])
    top = float(config["speed_profile"]["top_speed_mps"])
    return top + 0.5 * (bottom - top) * (1.0 + np.cos(theta))


def speed_derivative_theta(
    theta: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    bottom = float(config["speed_profile"]["bottom_speed_mps"])
    top = float(config["speed_profile"]["top_speed_mps"])
    return -0.5 * (bottom - top) * np.sin(theta)


def roll_schedule(
    maneuver_key: str,
    theta: np.ndarray,
    settings: dict[str, Any],
) -> np.ndarray:
    if maneuver_key != "immelmann":
        return np.zeros_like(theta)
    start = math.radians(float(settings["roll_start_path_angle_deg"]))
    end = math.radians(float(settings["roll_end_path_angle_deg"]))
    phase = (theta - start) / (end - start)
    return math.radians(float(settings["terminal_roll_deg"])) * quintic_smoothstep(
        phase
    )


def spline_factory(
    maneuver_key: str,
    angle_end: float,
    node_count: int,
) -> tuple[np.ndarray, Callable[[np.ndarray], Any]]:
    if maneuver_key == "loop_360":
        nodes = np.linspace(0.0, angle_end, node_count + 1)

        def build(values: np.ndarray):
            periodic_values = np.concatenate((values, values[:1]))
            return make_interp_spline(
                nodes,
                periodic_values,
                k=3,
                bc_type="periodic",
            )

        return nodes[:-1], build

    nodes = np.linspace(0.0, angle_end, node_count)

    def build(values: np.ndarray):
        return make_interp_spline(
            nodes,
            values,
            k=3,
            bc_type=([(1, 0.0)], [(1, 0.0)]),
        )

    return nodes, build


def basis_matrix(
    build_spline: Callable[[np.ndarray], Any],
    variable_count: int,
    theta: np.ndarray,
    derivative_order: int = 0,
) -> np.ndarray:
    matrix = np.empty((len(theta), variable_count), dtype=float)
    for index in range(variable_count):
        unit = np.zeros(variable_count)
        unit[index] = 1.0
        spline = build_spline(unit)
        if derivative_order:
            spline = spline.derivative(derivative_order)
        matrix[:, index] = spline(theta)
    return matrix


def trapz_weights(coordinate: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(coordinate)
    intervals = np.diff(coordinate)
    weights[:-1] += 0.5 * intervals
    weights[1:] += 0.5 * intervals
    return weights


def reference_radius(theta: np.ndarray) -> np.ndarray:
    return 32.0 + 8.0 * np.cos(3.0 * theta)


def quintic_coefficients(
    position_start: np.ndarray,
    position_end: np.ndarray,
    derivative_start: np.ndarray,
    derivative_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray:
    """Return vector-valued quintic coefficients on a unit segment."""

    coefficients = np.zeros((6, len(position_start)), dtype=float)
    coefficients[0] = position_start
    coefficients[1] = derivative_start
    coefficients[2] = 0.5 * second_start
    matrix = np.array(
        [
            [1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0],
            [6.0, 12.0, 20.0],
        ]
    )
    right_hand = np.vstack(
        (
            position_end - coefficients[0] - coefficients[1] - coefficients[2],
            derivative_end - coefficients[1] - 2.0 * coefficients[2],
            second_end - 2.0 * coefficients[2],
        )
    )
    coefficients[3:6] = np.linalg.solve(matrix, right_hand)
    return coefficients


def reference_position(theta: np.ndarray) -> np.ndarray:
    """Design waypoints used as boundary data for the quintic segments."""

    theta = np.asarray(theta, dtype=float)
    x = 32.0 * np.sin(theta) + np.sin(4.0 * theta) + 2.0 * np.sin(
        2.0 * theta
    )
    z = (
        32.0 * (1.0 - np.cos(theta))
        - 1.0
        - np.cos(4.0 * theta)
        + 2.0 * np.cos(2.0 * theta)
    )
    return np.column_stack((x, z))


def reference_derivatives(
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    radius = reference_radius(theta)
    radius_derivative = -24.0 * np.sin(3.0 * theta)
    tangent = np.column_stack((np.cos(theta), np.sin(theta)))
    normal = np.column_stack((-np.sin(theta), np.cos(theta)))
    first = radius[:, None] * tangent
    second = radius_derivative[:, None] * tangent + radius[:, None] * normal
    return first, second


def build_quintic_geometry(
    maneuver_key: str,
    settings: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build a C2 piecewise-quintic path for loop or Immelmann geometry."""

    angle_end = math.radians(float(settings["terminal_path_rotation_deg"]))
    segment_count = 6 if maneuver_key == "loop_360" else 3
    node_angles = np.linspace(0.0, angle_end, segment_count + 1)
    node_positions = reference_position(node_angles)
    node_first, node_second = reference_derivatives(node_angles)
    coefficients: list[np.ndarray] = []
    for index in range(segment_count):
        span = node_angles[index + 1] - node_angles[index]
        coefficients.append(
            quintic_coefficients(
                node_positions[index],
                node_positions[index + 1],
                span * node_first[index],
                span * node_first[index + 1],
                span**2 * node_second[index],
                span**2 * node_second[index + 1],
            )
        )

    dense_count = int(
        config["sampling"][
            "dense_angle_samples_full_loop"
            if maneuver_key == "loop_360"
            else "dense_angle_samples_immelmann"
        ]
    )
    theta = np.linspace(0.0, angle_end, dense_count)
    position = np.empty((dense_count, 2), dtype=float)
    first = np.empty_like(position)
    second = np.empty_like(position)
    segment_index = np.minimum(
        np.searchsorted(node_angles, theta, side="right") - 1,
        segment_count - 1,
    )
    segment_index = np.maximum(segment_index, 0)
    for index in range(segment_count):
        mask = segment_index == index
        span = node_angles[index + 1] - node_angles[index]
        u = (theta[mask] - node_angles[index]) / span
        powers = np.column_stack([u**order for order in range(6)])
        first_powers = np.column_stack(
            [
                np.zeros_like(u),
                np.ones_like(u),
                2.0 * u,
                3.0 * u**2,
                4.0 * u**3,
                5.0 * u**4,
            ]
        )
        second_powers = np.column_stack(
            [
                np.zeros_like(u),
                np.zeros_like(u),
                2.0 * np.ones_like(u),
                6.0 * u,
                12.0 * u**2,
                20.0 * u**3,
            ]
        )
        position[mask] = powers @ coefficients[index]
        first[mask] = (first_powers @ coefficients[index]) / span
        second[mask] = (second_powers @ coefficients[index]) / span**2

    diagnostics = {
        "segment_count": segment_count,
        "polynomial_degree": 5,
        "node_angles_deg": np.degrees(node_angles).tolist(),
        "node_positions_local_m": node_positions.tolist(),
        "continuity": "position_first_derivative_second_derivative_C2",
    }
    return {
        "theta": theta,
        "position": position,
        "first": first,
        "second": second,
    }, diagnostics


def optimize_radius_profile(
    maneuver_key: str,
    settings: dict[str, Any],
    config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    angle_end = math.radians(float(settings["terminal_path_rotation_deg"]))
    dense_count = int(
        config["sampling"][
            "dense_angle_samples_full_loop"
            if maneuver_key == "loop_360"
            else "dense_angle_samples_immelmann"
        ]
    )
    theta = np.linspace(0.0, angle_end, dense_count)
    node_count = int(settings["radius_node_count"])
    node_angles, build_spline = spline_factory(
        maneuver_key,
        angle_end,
        node_count,
    )
    variable_count = len(node_angles)
    basis = basis_matrix(build_spline, variable_count, theta)
    basis_d1 = basis_matrix(build_spline, variable_count, theta, 1)
    weights = trapz_weights(theta)

    speed = speed_profile(theta, config)
    speed_dtheta = speed_derivative_theta(theta, config)
    gravity = float(aircraft["atmosphere"]["gravity_m_s2"])
    mass = float(aircraft["confirmed_parameters"]["mass_kg"])
    area = float(aircraft["confirmed_parameters"]["geometry_m"]["wing_area"])
    density = float(aircraft["atmosphere"]["density_kg_m3"])
    dynamic_pressure = 0.5 * density * speed**2

    target_length = float(settings["target_path_length_m"])
    target_x, target_z = [
        float(value) for value in settings["target_terminal_local_position_m"]
    ]
    equality_matrix = np.vstack(
        (
            weights @ basis,
            (weights * np.cos(theta)) @ basis,
            (weights * np.sin(theta)) @ basis,
        )
    )
    equality_target = np.array([target_length, target_x, target_z])

    initial = reference_radius(node_angles)
    gram = equality_matrix @ equality_matrix.T
    initial = initial - equality_matrix.T @ np.linalg.solve(
        gram,
        equality_matrix @ initial - equality_target,
    )

    optimization = config["bspline_optimization"]
    lower = float(optimization["radius_lower_bound_m"])
    upper = float(optimization["radius_upper_bound_m"])
    objective_weights = optimization["objective_weights"]

    def profile_quantities(values: np.ndarray) -> dict[str, np.ndarray | float]:
        radius = basis @ values
        time = cumulative_trapezoid(radius / speed, theta)
        theta_rate = speed / radius
        tangential_acceleration = speed_dtheta * theta_rate
        normal_acceleration = speed**2 / radius
        tangent = np.column_stack(
            (np.cos(theta), np.zeros_like(theta), np.sin(theta))
        )
        normal = np.column_stack(
            (-np.sin(theta), np.zeros_like(theta), np.cos(theta))
        )
        acceleration = (
            tangent * tangential_acceleration[:, None]
            + normal * normal_acceleration[:, None]
        )
        jerk = np.gradient(acceleration, time, axis=0, edge_order=2)
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        normal_specific = np.abs(normal_acceleration + gravity * np.cos(theta))
        load = normal_specific / gravity
        normal_coefficient = (
            mass * normal_specific / np.maximum(dynamic_pressure * area, 1e-9)
        )
        curvature = 1.0 / radius
        curvature_rate = (
            -(basis_d1 @ values) / radius**2
        ) * theta_rate
        return {
            "radius": radius,
            "time": time,
            "jerk": jerk_magnitude,
            "load": load,
            "normal_coefficient": normal_coefficient,
            "curvature_rate": curvature_rate,
            "pitch_rate_degps": np.degrees(theta_rate),
        }

    initial_quantities = profile_quantities(initial)
    reference_scales = {
        "rms_jerk": max(
            float(np.sqrt(np.mean(np.asarray(initial_quantities["jerk"]) ** 2))),
            1e-6,
        ),
        "peak_normal_coefficient": max(
            float(np.max(initial_quantities["normal_coefficient"])),
            1e-6,
        ),
        "curvature_rate": max(
            float(
                np.sqrt(
                    np.mean(
                        np.asarray(initial_quantities["curvature_rate"]) ** 2
                    )
                )
            ),
            1e-6,
        ),
        "reference_deviation": 8.0,
    }

    def objective(values: np.ndarray) -> float:
        quantities = profile_quantities(values)
        rms_jerk = float(
            np.sqrt(np.mean(np.asarray(quantities["jerk"]) ** 2))
        )
        peak_coefficient = float(np.max(quantities["normal_coefficient"]))
        rms_curvature_rate = float(
            np.sqrt(
                np.mean(np.asarray(quantities["curvature_rate"]) ** 2)
            )
        )
        deviation = float(
            np.sqrt(np.mean((basis @ values - reference_radius(theta)) ** 2))
        )
        return (
            float(objective_weights["rms_jerk"])
            * (rms_jerk / reference_scales["rms_jerk"]) ** 2
            + float(objective_weights["peak_normal_coefficient"])
            * (
                peak_coefficient
                / reference_scales["peak_normal_coefficient"]
            )
            ** 2
            + float(objective_weights["curvature_rate"])
            * (
                rms_curvature_rate / reference_scales["curvature_rate"]
            )
            ** 2
            + float(objective_weights["reference_deviation"])
            * (
                deviation / reference_scales["reference_deviation"]
            )
            ** 2
        )

    alpha_limit = float(config["evaluation"]["lift_limit_alpha_deg"])
    lift_limit = float(database.sample("CZ", math.radians(alpha_limit), 0.0))
    margin_factor = float(config["evaluation"]["speed_margin_factor"])
    min_margin = float(
        config["evaluation"]["preliminary_dynamic_screening_thresholds"][
            "minimum_lift_margin_ratio_with_speed_margin"
        ]
    )
    maximum_normal_coefficient = lift_limit / (margin_factor**2 * min_margin)

    def inequality(values: np.ndarray) -> np.ndarray:
        quantities = profile_quantities(values)
        return np.array(
            [
                float(np.min(quantities["radius"])) - 20.0,
                3.0 - float(np.max(quantities["load"])),
                maximum_normal_coefficient
                - float(np.max(quantities["normal_coefficient"])),
                120.0 - float(np.max(quantities["pitch_rate_degps"])),
            ]
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda values: equality_matrix @ values - equality_target,
        },
        {"type": "ineq", "fun": inequality},
    ]
    started = time.perf_counter()
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(lower, upper)] * variable_count,
        constraints=constraints,
        options={
            "maxiter": int(optimization["maximum_iterations"]),
            "ftol": float(optimization["function_tolerance"]),
            "disp": False,
        },
    )
    elapsed = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(
            f"{settings['display_name']} B样条优化失败: {result.message}"
        )

    final_spline = build_spline(result.x)
    diagnostics = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective_initial": float(objective(initial)),
        "objective_final": float(result.fun),
        "generation_time_s": elapsed,
        "node_angles_deg": np.degrees(node_angles).tolist(),
        "initial_radius_nodes_m": initial.tolist(),
        "optimized_radius_nodes_m": result.x.tolist(),
        "equality_residuals": (
            equality_matrix @ result.x - equality_target
        ).tolist(),
        "inequality_margins": inequality(result.x).tolist(),
    }
    return final_spline, diagnostics


def attitude_from_path(
    tangent: np.ndarray,
    path_normal: np.ndarray,
    roll_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(roll_rad)[:, None]
    sine = np.sin(roll_rad)[:, None]
    dot = np.sum(tangent * path_normal, axis=1)[:, None]
    body_up = (
        path_normal * cosine
        + np.cross(tangent, path_normal) * sine
        + tangent * dot * (1.0 - cosine)
    )
    body_up /= np.linalg.norm(body_up, axis=1, keepdims=True)
    body_right = np.cross(body_up, tangent)
    body_right /= np.linalg.norm(body_right, axis=1, keepdims=True)
    body_up = np.cross(tangent, body_right)
    rotation = np.stack((tangent, body_right, body_up), axis=2)

    from scipy.spatial.transform import Rotation

    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    quaternion = np.column_stack(
        (quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2])
    )
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0.0:
            quaternion[index] *= -1.0
    return rotation, quaternion


def body_rates(rotation: np.ndarray, time_values: np.ndarray) -> np.ndarray:
    derivative = np.gradient(rotation, time_values, axis=0, edge_order=2)
    rates = np.empty((len(time_values), 3), dtype=float)
    for index in range(len(time_values)):
        skew = rotation[index].T @ derivative[index]
        skew = 0.5 * (skew - skew.T)
        rates[index] = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return rates


def assemble_frame(
    maneuver_key: str,
    settings: dict[str, Any],
    config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
    time_values: np.ndarray,
    theta: np.ndarray,
    local_position: np.ndarray,
    tangent: np.ndarray,
    path_normal: np.ndarray,
    speed: np.ndarray,
    curvature: np.ndarray,
    parameter_rate: np.ndarray,
) -> pd.DataFrame:
    start = np.asarray(settings["start_position_m"], dtype=float)
    position = np.column_stack(
        (
            start[0] + local_position[:, 0],
            np.full(len(local_position), start[1]),
            start[2] + local_position[:, 1],
        )
    )
    expected_local = np.asarray(
        settings["target_terminal_local_position_m"],
        dtype=float,
    )
    position[-1] = start + np.array(
        [expected_local[0], 0.0, expected_local[1]]
    )

    velocity = tangent * speed[:, None]
    tangential_acceleration = (
        speed_derivative_theta(theta, config) * parameter_rate
    )
    normal_acceleration = speed**2 * curvature
    acceleration = (
        tangent * tangential_acceleration[:, None]
        + path_normal * normal_acceleration[:, None]
    )
    jerk = np.gradient(acceleration, time_values, axis=0, edge_order=2)

    roll_rad = roll_schedule(maneuver_key, theta, settings)
    rotation, quaternion = attitude_from_path(tangent, path_normal, roll_rad)
    rates = body_rates(rotation, time_values)

    gravity = float(aircraft["atmosphere"]["gravity_m_s2"])
    gravity_vector = np.array([0.0, 0.0, -gravity])
    specific_force = acceleration - gravity_vector[None]
    tangential_specific = np.sum(specific_force * tangent, axis=1)
    normal_specific = specific_force - tangential_specific[:, None] * tangent
    normal_specific_magnitude = np.linalg.norm(normal_specific, axis=1)
    signed_path_load = (
        np.sum(normal_specific * path_normal, axis=1) / gravity
    )
    normal_load = normal_specific_magnitude / gravity

    body_up = rotation[:, :, 2]
    body_lift_specific = np.sum(normal_specific * body_up, axis=1)
    side_specific = normal_specific - body_lift_specific[:, None] * body_up
    side_specific_magnitude = np.linalg.norm(side_specific, axis=1)

    mass = float(aircraft["confirmed_parameters"]["mass_kg"])
    area = float(aircraft["confirmed_parameters"]["geometry_m"]["wing_area"])
    density = float(aircraft["atmosphere"]["density_kg_m3"])
    dynamic_pressure = 0.5 * density * speed**2
    normal_coefficient = (
        mass * normal_specific_magnitude / (dynamic_pressure * area)
    )
    body_lift_coefficient = (
        mass * body_lift_specific / (dynamic_pressure * area)
    )
    side_force_coefficient = (
        mass * side_specific_magnitude / (dynamic_pressure * area)
    )

    alpha_limit = float(config["evaluation"]["lift_limit_alpha_deg"])
    alpha_grid = np.linspace(
        math.radians(-alpha_limit),
        math.radians(alpha_limit),
        1201,
    )
    lift_grid = np.asarray(
        database.sample("CZ", alpha_grid, np.zeros_like(alpha_grid)),
        dtype=float,
    )
    order = np.argsort(lift_grid)
    lift_sorted = lift_grid[order]
    alpha_sorted = np.degrees(alpha_grid[order])
    alpha_required = np.full_like(body_lift_coefficient, np.nan)
    valid = (
        (body_lift_coefficient >= lift_sorted[0])
        & (body_lift_coefficient <= lift_sorted[-1])
    )
    alpha_required[valid] = np.interp(
        body_lift_coefficient[valid],
        lift_sorted,
        alpha_sorted,
    )
    lift_limit = float(database.sample("CZ", math.radians(alpha_limit), 0.0))
    raw_lift_margin = lift_limit / np.maximum(normal_coefficient, 1e-9)
    speed_margin_factor = float(config["evaluation"]["speed_margin_factor"])
    lift_margin_with_speed_margin = (
        raw_lift_margin / speed_margin_factor**2
    )

    tangential_force = mass * tangential_specific
    mechanical_power = tangential_force * speed
    curvature_rate = np.gradient(curvature, time_values, edge_order=2)
    radius = 1.0 / np.maximum(np.abs(curvature), 1e-9)
    path_angle = np.unwrap(np.arctan2(tangent[:, 2], tangent[:, 0]))
    path_angle -= path_angle[0]

    return pd.DataFrame(
        {
            "t": time_values,
            "x": position[:, 0],
            "y": position[:, 1],
            "z": position[:, 2],
            "vx": velocity[:, 0],
            "vy": velocity[:, 1],
            "vz": velocity[:, 2],
            "ax": acceleration[:, 0],
            "ay": acceleration[:, 1],
            "az": acceleration[:, 2],
            "speed_mps": speed,
            "path_angle_unwrapped_deg": np.degrees(path_angle),
            "roll_command_deg": np.degrees(roll_rad),
            "qw": quaternion[:, 0],
            "qx": quaternion[:, 1],
            "qy": quaternion[:, 2],
            "qz": quaternion[:, 3],
            "p_degps": np.degrees(rates[:, 0]),
            "q_degps": np.degrees(rates[:, 1]),
            "r_degps": np.degrees(rates[:, 2]),
            "path_pitch_rate_degps": np.degrees(speed * curvature),
            "curvature_radius_m": radius,
            "curvature_1pm": curvature,
            "curvature_rate_1pm_s": curvature_rate,
            "acceleration_mps2": np.linalg.norm(acceleration, axis=1),
            "jerk_mps3": np.linalg.norm(jerk, axis=1),
            "signed_path_load_factor": signed_path_load,
            "normal_load_factor": normal_load,
            "required_normal_force_coefficient": normal_coefficient,
            "required_body_lift_coefficient": body_lift_coefficient,
            "required_side_force_coefficient": side_force_coefficient,
            "estimated_alpha_required_deg": alpha_required,
            "raw_lift_margin_ratio": raw_lift_margin,
            "lift_margin_ratio_with_speed_margin": (
                lift_margin_with_speed_margin
            ),
            "required_net_tangential_force_n": tangential_force,
            "required_mechanical_power_no_drag_w": mechanical_power,
        }
    )


def uniform_time_samples(
    dense_time: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    time_step = float(config["sampling"]["time_step_s"])
    time_values = np.arange(0.0, dense_time[-1], time_step)
    if len(time_values) == 0 or not math.isclose(
        time_values[-1],
        dense_time[-1],
    ):
        time_values = np.append(time_values, dense_time[-1])
    return time_values


def build_frame_from_quintic(
    maneuver_key: str,
    settings: dict[str, Any],
    config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
    geometry: dict[str, np.ndarray],
) -> pd.DataFrame:
    dense_theta = geometry["theta"]
    dense_position = geometry["position"]
    dense_first = geometry["first"]
    dense_second = geometry["second"]
    dense_ds = np.linalg.norm(dense_first, axis=1)
    dense_curvature = (
        dense_first[:, 0] * dense_second[:, 1]
        - dense_first[:, 1] * dense_second[:, 0]
    ) / np.maximum(dense_ds**3, 1e-12)
    dense_speed = speed_profile(dense_theta, config)
    dense_time = cumulative_trapezoid(
        dense_ds / dense_speed,
        dense_theta,
    )
    time_values = uniform_time_samples(dense_time, config)
    theta = np.interp(time_values, dense_time, dense_theta)
    position = np.column_stack(
        (
            np.interp(theta, dense_theta, dense_position[:, 0]),
            np.interp(theta, dense_theta, dense_position[:, 1]),
        )
    )
    first = np.column_stack(
        (
            np.interp(theta, dense_theta, dense_first[:, 0]),
            np.interp(theta, dense_theta, dense_first[:, 1]),
        )
    )
    ds_dtheta = np.linalg.norm(first, axis=1)
    tangent = np.column_stack(
        (
            first[:, 0] / ds_dtheta,
            np.zeros_like(theta),
            first[:, 1] / ds_dtheta,
        )
    )
    path_normal = np.column_stack(
        (-tangent[:, 2], np.zeros_like(theta), tangent[:, 0])
    )
    curvature = np.interp(theta, dense_theta, dense_curvature)
    speed = speed_profile(theta, config)
    parameter_rate = speed / ds_dtheta
    return assemble_frame(
        maneuver_key,
        settings,
        config,
        aircraft,
        database,
        time_values,
        theta,
        position,
        tangent,
        path_normal,
        speed,
        curvature,
        parameter_rate,
    )


def build_frame_from_radius(
    maneuver_key: str,
    settings: dict[str, Any],
    config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
    radius_spline,
) -> pd.DataFrame:
    angle_end = math.radians(float(settings["terminal_path_rotation_deg"]))
    dense_count = int(
        config["sampling"][
            "dense_angle_samples_full_loop"
            if maneuver_key == "loop_360"
            else "dense_angle_samples_immelmann"
        ]
    )
    dense_theta = np.linspace(0.0, angle_end, dense_count)
    dense_radius = np.asarray(radius_spline(dense_theta), dtype=float)
    dense_speed = speed_profile(dense_theta, config)
    dense_time = cumulative_trapezoid(
        dense_radius / dense_speed,
        dense_theta,
    )
    dense_x = cumulative_trapezoid(
        dense_radius * np.cos(dense_theta),
        dense_theta,
    )
    dense_z = cumulative_trapezoid(
        dense_radius * np.sin(dense_theta),
        dense_theta,
    )
    time_values = uniform_time_samples(dense_time, config)
    theta = np.interp(time_values, dense_time, dense_theta)
    radius = np.asarray(radius_spline(theta), dtype=float)
    speed = speed_profile(theta, config)
    position = np.column_stack(
        (
            np.interp(theta, dense_theta, dense_x),
            np.interp(theta, dense_theta, dense_z),
        )
    )
    tangent = np.column_stack(
        (np.cos(theta), np.zeros_like(theta), np.sin(theta))
    )
    path_normal = np.column_stack(
        (-np.sin(theta), np.zeros_like(theta), np.cos(theta))
    )
    return assemble_frame(
        maneuver_key,
        settings,
        config,
        aircraft,
        database,
        time_values,
        theta,
        position,
        tangent,
        path_normal,
        speed,
        1.0 / radius,
        speed / radius,
    )


def evaluate_frame(
    method_label: str,
    maneuver_key: str,
    frame: pd.DataFrame,
    settings: dict[str, Any],
    config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
    generation_time_s: float,
    method_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    geometric_limits = config["evaluation"]["geometric_thresholds"]
    dynamic_limits = config["evaluation"][
        "preliminary_dynamic_screening_thresholds"
    ]
    start = np.asarray(settings["start_position_m"], dtype=float)
    target_local = np.asarray(
        settings["target_terminal_local_position_m"],
        dtype=float,
    )
    expected_end = start + np.array(
        [target_local[0], 0.0, target_local[1]]
    )
    end = frame.loc[frame.index[-1], ["x", "y", "z"]].to_numpy(dtype=float)
    terminal_position_error = float(np.linalg.norm(end - expected_end))
    plane_deviation = float(np.max(np.abs(frame["y"] - start[1])))
    path_angle_error = abs(
        float(frame["path_angle_unwrapped_deg"].iloc[-1])
        - float(settings["terminal_path_rotation_deg"])
    )
    roll_error = abs(
        float(frame["roll_command_deg"].iloc[-1])
        - float(settings["terminal_roll_deg"])
    )
    alpha = frame["estimated_alpha_required_deg"].to_numpy(dtype=float)
    finite_alpha = alpha[np.isfinite(alpha)]
    max_alpha = (
        float(np.max(np.abs(finite_alpha))) if len(finite_alpha) else math.inf
    )

    geometric_checks = {
        "terminal_position": terminal_position_error
        <= float(geometric_limits["terminal_position_error_m"]),
        "maneuver_plane": plane_deviation
        <= float(geometric_limits["maneuver_plane_deviation_m"]),
        "terminal_path_angle": path_angle_error
        <= float(geometric_limits["terminal_path_angle_error_deg"]),
        "terminal_roll": roll_error
        <= float(geometric_limits["terminal_roll_error_deg"]),
        "minimum_curvature_radius": float(
            frame["curvature_radius_m"].min()
        )
        >= float(geometric_limits["minimum_curvature_radius_m"]),
    }
    dynamic_checks = {
        "speed": (
            float(frame["speed_mps"].min())
            >= float(dynamic_limits["minimum_speed_mps"])
            and float(frame["speed_mps"].max())
            <= float(dynamic_limits["maximum_speed_mps"])
        ),
        "load_factor": float(frame["normal_load_factor"].max())
        <= float(dynamic_limits["maximum_absolute_load_factor"]),
        "required_alpha": bool(
            np.all(np.isfinite(alpha))
            and max_alpha <= float(dynamic_limits["maximum_required_alpha_deg"])
        ),
        "raw_lift_margin": float(frame["raw_lift_margin_ratio"].min())
        >= float(dynamic_limits["minimum_raw_lift_margin_ratio"]),
        "lift_margin_with_speed_margin": float(
            frame["lift_margin_ratio_with_speed_margin"].min()
        )
        >= float(dynamic_limits["minimum_lift_margin_ratio_with_speed_margin"]),
        "pitch_rate": float(frame["path_pitch_rate_degps"].abs().max())
        <= float(dynamic_limits["maximum_pitch_rate_degps"]),
        "roll_rate": float(frame["p_degps"].abs().max())
        <= float(dynamic_limits["maximum_roll_rate_degps"]),
        "side_force": float(
            frame["required_side_force_coefficient"].abs().max()
        )
        <= float(dynamic_limits["maximum_side_force_coefficient"]),
    }
    lift_limit = float(
        database.sample(
            "CZ",
            math.radians(float(config["evaluation"]["lift_limit_alpha_deg"])),
            0.0,
        )
    )
    metrics = {
        "schema_version": "2.0",
        "method": method_label,
        "maneuver_key": maneuver_key,
        "maneuver_name": settings["display_name"],
        "analysis_status": "planning_level_not_for_flight_release",
        "generation_time_s": generation_time_s,
        "duration_s": float(frame["t"].iloc[-1]),
        "samples": int(len(frame)),
        "path_length_m": float(
            np.trapz(
                frame["speed_mps"].to_numpy(),
                frame["t"].to_numpy(),
            )
        ),
        "terminal_position_error_m": terminal_position_error,
        "maneuver_plane_max_deviation_m": plane_deviation,
        "terminal_path_angle_error_deg": path_angle_error,
        "terminal_roll_error_deg": roll_error,
        "path_width_m": float(frame["x"].max() - frame["x"].min()),
        "path_height_m": float(frame["z"].max() - frame["z"].min()),
        "altitude_gain_m": float(frame["z"].iloc[-1] - frame["z"].iloc[0]),
        "minimum_curvature_radius_m": float(
            frame["curvature_radius_m"].min()
        ),
        "maximum_curvature_radius_m": float(
            frame["curvature_radius_m"].max()
        ),
        "minimum_speed_mps": float(frame["speed_mps"].min()),
        "maximum_speed_mps": float(frame["speed_mps"].max()),
        "mean_speed_mps": float(frame["speed_mps"].mean()),
        "maximum_acceleration_mps2": float(frame["acceleration_mps2"].max()),
        "maximum_jerk_mps3": float(frame["jerk_mps3"].max()),
        "rms_jerk_mps3": float(
            np.sqrt(np.mean(frame["jerk_mps3"].to_numpy() ** 2))
        ),
        "rms_curvature_rate_1pm_s": float(
            np.sqrt(
                np.mean(frame["curvature_rate_1pm_s"].to_numpy() ** 2)
            )
        ),
        "maximum_normal_load_factor": float(
            frame["normal_load_factor"].max()
        ),
        "rms_normal_load_factor": float(
            np.sqrt(np.mean(frame["normal_load_factor"].to_numpy() ** 2))
        ),
        "maximum_required_normal_force_coefficient": float(
            frame["required_normal_force_coefficient"].max()
        ),
        "maximum_required_body_lift_coefficient": float(
            frame["required_body_lift_coefficient"].max()
        ),
        "maximum_required_side_force_coefficient": float(
            frame["required_side_force_coefficient"].max()
        ),
        "maximum_estimated_alpha_required_deg": max_alpha,
        "positive_lift_limit_coefficient_at_alpha_limit": lift_limit,
        "minimum_raw_lift_margin_ratio": float(
            frame["raw_lift_margin_ratio"].min()
        ),
        "minimum_lift_margin_ratio_with_speed_margin": float(
            frame["lift_margin_ratio_with_speed_margin"].min()
        ),
        "maximum_path_pitch_rate_degps": float(
            frame["path_pitch_rate_degps"].abs().max()
        ),
        "maximum_roll_rate_degps": float(frame["p_degps"].abs().max()),
        "maximum_positive_net_tangential_force_n": float(
            np.maximum(frame["required_net_tangential_force_n"], 0.0).max()
        ),
        "maximum_positive_mechanical_power_no_drag_w": float(
            np.maximum(
                frame["required_mechanical_power_no_drag_w"],
                0.0,
            ).max()
        ),
        "geometric_checks": geometric_checks,
        "dynamic_screening_checks": dynamic_checks,
        "geometric_pass": bool(all(geometric_checks.values())),
        "dynamic_screening_pass": bool(all(dynamic_checks.values())),
        "overall_planning_pass": bool(
            all(geometric_checks.values()) and all(dynamic_checks.values())
        ),
        "aircraft": {
            "mass_kg": aircraft["confirmed_parameters"]["mass_kg"],
            "wing_area_m2": aircraft["confirmed_parameters"]["geometry_m"][
                "wing_area"
            ],
            "wing_span_m": aircraft["confirmed_parameters"]["geometry_m"][
                "wing_span"
            ],
            "mean_aerodynamic_chord_m": aircraft["confirmed_parameters"][
                "geometry_m"
            ]["mean_aerodynamic_chord"],
            "inertia_kg_m2": aircraft["confirmed_parameters"][
                "inertia_kg_m2"
            ],
        },
    }
    if method_diagnostics is not None:
        diagnostics_key = (
            "optimization"
            if method_label == BSPLINE_LABEL
            else "quintic_polynomial"
        )
        metrics[diagnostics_key] = method_diagnostics
    return metrics


def generate_quintic(
    maneuver_key: str,
    config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = config["maneuvers"][maneuver_key]
    started = time.perf_counter()
    geometry, diagnostics = build_quintic_geometry(
        maneuver_key,
        settings,
        config,
    )
    frame = build_frame_from_quintic(
        maneuver_key,
        settings,
        config,
        aircraft,
        database,
        geometry,
    )
    elapsed = time.perf_counter() - started
    diagnostics["generation_time_s"] = elapsed
    metrics = evaluate_frame(
        QUINTIC_LABEL,
        maneuver_key,
        frame,
        settings,
        config,
        aircraft,
        database,
        elapsed,
        diagnostics,
    )
    return frame, metrics


def generate_bspline(
    maneuver_key: str,
    config: dict[str, Any],
    aircraft: dict[str, Any],
    database,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = config["maneuvers"][maneuver_key]
    spline, diagnostics = optimize_radius_profile(
        maneuver_key,
        settings,
        config,
        aircraft,
        database,
    )
    frame = build_frame_from_radius(
        maneuver_key,
        settings,
        config,
        aircraft,
        database,
        spline,
    )
    metrics = evaluate_frame(
        BSPLINE_LABEL,
        maneuver_key,
        frame,
        settings,
        config,
        aircraft,
        database,
        float(diagnostics["generation_time_s"]),
        diagnostics,
    )
    return frame, metrics


def save_method_result(
    method_label: str,
    maneuver_key: str,
    frame: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    output = METHOD_DIRS[method_label] / MANEUVER_DIRS[maneuver_key]
    output.mkdir(parents=True, exist_ok=True)
    frame_to_save = frame.copy()
    frame_to_save.insert(0, "maneuver", metrics["maneuver_name"])
    frame_to_save.insert(0, "method", method_label)
    frame_to_save.to_csv(
        output / "trajectory_attitude.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    write_json(output / "metrics.json", metrics)


def build_summary(
    records: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]],
) -> pd.DataFrame:
    rows = []
    for (method_label, maneuver_key), (_, item) in records.items():
        rows.append(
            {
                "方法": method_label,
                "机动动作": item["maneuver_name"],
                "生成时间_s": item["generation_time_s"],
                "持续时间_s": item["duration_s"],
                "轨迹长度_m": item["path_length_m"],
                "终端位置误差_m": item["terminal_position_error_m"],
                "最小曲率半径_m": item["minimum_curvature_radius_m"],
                "最大曲率半径_m": item["maximum_curvature_radius_m"],
                "最大过载_g": item["maximum_normal_load_factor"],
                "最大迎角需求_deg": item[
                    "maximum_estimated_alpha_required_deg"
                ],
                "最小含速度裕度": item[
                    "minimum_lift_margin_ratio_with_speed_margin"
                ],
                "最大俯仰角速度_degps": item[
                    "maximum_path_pitch_rate_degps"
                ],
                "最大滚转角速度_degps": item["maximum_roll_rate_degps"],
                "最大侧向力系数": item[
                    "maximum_required_side_force_coefficient"
                ],
                "最大jerk_mps3": item["maximum_jerk_mps3"],
                "均方根jerk_mps3": item["rms_jerk_mps3"],
                "均方根曲率变化率_1pm_s": item[
                    "rms_curvature_rate_1pm_s"
                ],
                "几何评估": "通过" if item["geometric_pass"] else "未通过",
                "动力学筛查": (
                    "通过" if item["dynamic_screening_pass"] else "未通过"
                ),
                "规划级结论": (
                    "通过" if item["overall_planning_pass"] else "需修正"
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_bspline_principle(
    records: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    for axis, maneuver_key in zip(axes, ("loop_360", "immelmann")):
        frame, metrics = records[(BSPLINE_LABEL, maneuver_key)]
        optimization = metrics["optimization"]
        axis.plot(
            frame["path_angle_unwrapped_deg"],
            frame["curvature_radius_m"],
            color="#0369A1",
            linewidth=2.3,
            label="优化后B样条曲线",
        )
        axis.scatter(
            optimization["node_angles_deg"],
            optimization["optimized_radius_nodes_m"],
            color="#DC2626",
            s=42,
            zorder=4,
            label="优化节点值",
        )
        axis.plot(
            optimization["node_angles_deg"],
            optimization["initial_radius_nodes_m"],
            color="#64748B",
            linestyle="--",
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="五次多项式轨迹初值",
        )
        axis.set_title(metrics["maneuver_name"])
        axis.set_xlabel("累计航迹转角 / (°)")
        axis.set_ylabel("曲率半径 / m")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8.5)
    fig.suptitle(
        "B样条曲率参数化及控制节点优化结果",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_geometry_comparison(
    maneuver_key: str,
    records: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]],
    path: Path,
) -> None:
    quintic, quintic_metrics = records[(QUINTIC_LABEL, maneuver_key)]
    bspline, bspline_metrics = records[(BSPLINE_LABEL, maneuver_key)]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)

    axes[0].plot(
        quintic["x"],
        quintic["z"],
        color="#64748B",
        linestyle="--",
        linewidth=2.0,
        label=QUINTIC_LABEL,
    )
    axes[0].plot(
        bspline["x"],
        bspline["z"],
        color="#0369A1",
        linewidth=2.2,
        label=BSPLINE_LABEL,
    )
    axes[0].scatter(
        [quintic["x"].iloc[0]],
        [quintic["z"].iloc[0]],
        color="#16A34A",
        s=48,
        label="起点",
    )
    axes[0].scatter(
        [quintic["x"].iloc[-1]],
        [quintic["z"].iloc[-1]],
        color="#DC2626",
        marker="X",
        s=54,
        label="终点",
    )
    axes[0].set_title("机动平面轨迹")
    axes[0].set_xlabel("前向位置 x / m")
    axes[0].set_ylabel("高度 z / m")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    for frame, label, color, style in (
        (quintic, QUINTIC_LABEL, "#64748B", "--"),
        (bspline, BSPLINE_LABEL, "#0369A1", "-"),
    ):
        axes[1].plot(
            frame["path_angle_unwrapped_deg"],
            frame["curvature_radius_m"],
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=label,
        )
        axes[2].plot(
            frame["t"],
            frame["jerk_mps3"],
            color=color,
            linestyle=style,
            linewidth=1.8,
            label=label,
        )
    axes[1].axhline(20.0, color="#DC2626", linestyle=":", linewidth=1.0)
    axes[1].set_title("曲率半径变化")
    axes[1].set_xlabel("累计航迹转角 / (°)")
    axes[1].set_ylabel("曲率半径 / m")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    axes[2].set_title("轨迹jerk时序")
    axes[2].set_xlabel("时间 / s")
    axes[2].set_ylabel("jerk / (m/s³)")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best", fontsize=8)

    fig.suptitle(
        f"{quintic_metrics['maneuver_name']}几何与平滑性对比",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dynamics_comparison(
    maneuver_key: str,
    records: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]],
    path: Path,
) -> None:
    quintic, quintic_metrics = records[(QUINTIC_LABEL, maneuver_key)]
    bspline, _ = records[(BSPLINE_LABEL, maneuver_key)]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.5), constrained_layout=True)
    panels = [
        ("normal_load_factor", "法向过载", "过载系数 / g", 3.0),
        (
            "estimated_alpha_required_deg",
            "估算迎角需求",
            "迎角 / (°)",
            15.0,
        ),
        (
            "lift_margin_ratio_with_speed_margin",
            "计入速度裕度后的升力裕度（阈值附近）",
            "裕度比",
            0.95,
        ),
        (
            "path_pitch_rate_degps",
            "航迹俯仰角速度",
            "角速度 / (°/s)",
            120.0,
        ),
    ]
    for axis, (column, title, ylabel, threshold) in zip(axes.flat, panels):
        axis.plot(
            quintic["t"],
            quintic[column],
            color="#64748B",
            linestyle="--",
            linewidth=1.8,
            label=QUINTIC_LABEL,
        )
        axis.plot(
            bspline["t"],
            bspline[column],
            color="#0369A1",
            linewidth=2.0,
            label=BSPLINE_LABEL,
        )
        axis.axhline(
            threshold,
            color="#DC2626",
            linestyle=":",
            linewidth=1.0,
            label="筛查阈值",
        )
        axis.set_title(title)
        axis.set_xlabel("时间 / s")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)
        if column == "lift_margin_ratio_with_speed_margin":
            axis.set_ylim(0.85, 1.60)
    fig.suptitle(
        f"{quintic_metrics['maneuver_name']}动力学需求对比",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.7), constrained_layout=True)
    metrics = [
        ("最大过载_g", "最大过载 / g"),
        ("最大迎角需求_deg", "最大迎角需求 / (°)"),
        ("最小含速度裕度", "最小升力裕度比"),
        ("均方根jerk_mps3", "均方根jerk / (m/s³)"),
        ("均方根曲率变化率_1pm_s", "均方根曲率变化率 / (1/(m·s))"),
        ("生成时间_s", "生成时间 / s"),
    ]
    colors_by_method = {
        QUINTIC_LABEL: "#64748B",
        BSPLINE_LABEL: "#0369A1",
    }
    x = np.arange(2)
    width = 0.34
    maneuver_order = ["360度筋斗", "殷麦曼机动（半筋斗接半滚转）"]
    for axis, (column, title) in zip(axes.flat, metrics):
        for index, method in enumerate((QUINTIC_LABEL, BSPLINE_LABEL)):
            subset = (
                summary[summary["方法"] == method]
                .set_index("机动动作")
                .loc[maneuver_order]
            )
            values = subset[column].astype(float).to_numpy()
            bars = axis.bar(
                x + (index - 0.5) * width,
                values,
                width,
                color=colors_by_method[method],
                label=method,
            )
            axis.bar_label(bars, fmt="%.3g", padding=2, fontsize=8)
        axis.set_title(title)
        axis.set_xticks(x, ["筋斗", "殷麦曼"])
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.035),
        frameon=True,
    )
    fig.suptitle(
        "两种传统轨姿生成方法关键指标对比",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_conclusion(summary: pd.DataFrame, path: Path) -> None:
    indexed = summary.set_index(["方法", "机动动作"])
    lines = [
        "# 双传统方法测试结论",
        "",
        "五次多项式方法和B样条约束优化方法均完成360度筋斗与殷麦曼机动的规划级生成。",
        "两种方法使用同一飞机参数、动作边界、速度规律、采样步长和筛查阈值。",
        "",
    ]
    for maneuver in ("360度筋斗", "殷麦曼机动（半筋斗接半滚转）"):
        quintic = indexed.loc[(QUINTIC_LABEL, maneuver)]
        bspline = indexed.loc[(BSPLINE_LABEL, maneuver)]
        jerk_change = 100.0 * (
            float(bspline["均方根jerk_mps3"])
            / float(quintic["均方根jerk_mps3"])
            - 1.0
        )
        margin_change = 100.0 * (
            float(bspline["最小含速度裕度"])
            / float(quintic["最小含速度裕度"])
            - 1.0
        )
        lines.extend(
            [
                f"## {maneuver}",
                "",
                (
                    f"B样条方法相对五次多项式方法的均方根jerk变化"
                    f"{jerk_change:+.2f}%，最小含速度裕度变化"
                    f"{margin_change:+.2f}%。两种方法的几何评估和动力学筛查"
                    f"分别为{quintic['规划级结论']}和{bspline['规划级结论']}。"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "B样条方法的计算时间高于五次多项式方法，但可以在生成阶段直接处理闭合、终端位置、路径长度、曲率半径、过载和升力裕度约束。",
            "五次多项式方法适合作为快速、确定且可解释的基准；B样条方法适合作为需要局部调整和多约束折中的传统优化基线。",
            "",
            "本结论属于参考轨迹逆动力学与VLM气动表支持下的规划级评估。尚缺推力曲线、舵面限位、舵机速率、真实失速和结构过载数据，不能据此直接发布实飞动作。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    global CONFIG_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="双传统方法配置文件",
    )
    args = parser.parse_args()
    CONFIG_PATH = args.config.resolve()

    config, aircraft, database = load_project()
    records: dict[
        tuple[str, str],
        tuple[pd.DataFrame, dict[str, Any]],
    ] = {}

    for maneuver_key in ("loop_360", "immelmann"):
        quintic = generate_quintic(
            maneuver_key,
            config,
            aircraft,
            database,
        )
        bspline = generate_bspline(
            maneuver_key,
            config,
            aircraft,
            database,
        )
        records[(QUINTIC_LABEL, maneuver_key)] = quintic
        records[(BSPLINE_LABEL, maneuver_key)] = bspline
        save_method_result(QUINTIC_LABEL, maneuver_key, *quintic)
        save_method_result(BSPLINE_LABEL, maneuver_key, *bspline)

    comparison_dir = RESULTS_DIR / "方法对比"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(records)
    summary.to_csv(
        comparison_dir / "comparison_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    plot_bspline_principle(
        records,
        comparison_dir / "bspline_principle.png",
    )
    for maneuver_key in ("loop_360", "immelmann"):
        short = "loop" if maneuver_key == "loop_360" else "immelmann"
        plot_geometry_comparison(
            maneuver_key,
            records,
            comparison_dir / f"{short}_geometry_comparison.png",
        )
        plot_dynamics_comparison(
            maneuver_key,
            records,
            comparison_dir / f"{short}_dynamics_comparison.png",
        )
    plot_summary(summary, comparison_dir / "key_metrics_comparison.png")
    write_conclusion(summary, comparison_dir / "comparison_conclusion.md")
    write_json(
        comparison_dir / "run_summary.json",
        {
            "config_file": str(CONFIG_PATH),
            "aircraft_file": str(
                (
                    CONFIG_PATH.parent
                    / config["aircraft_parameters_file"]
                ).resolve()
            ),
            "records": {
                f"{method}|{maneuver}": metrics
                for (method, maneuver), (_, metrics) in records.items()
            },
        },
    )
    print(summary.to_string(index=False))
    print(f"\n结果目录: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
