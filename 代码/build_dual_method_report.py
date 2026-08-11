"""Build the rewritten traditional-method test report.

The report follows the retained Task-1 report page system while replacing the
content with a three-part structure: principles, geometric comparison, and
aircraft-dynamics compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
OUTPUTS_DIR = PROJECT_DIR.parent
WORKSPACE = OUTPUTS_DIR.parent
RESULTS_DIR = PROJECT_DIR / "结果"
COMPARISON_DIR = RESULTS_DIR / "方法对比"
REPORT_DIR = PROJECT_DIR / "测试报告"
ASSET_DIR = REPORT_DIR / "插图"
FORMULA_DIR = ASSET_DIR / "公式"
OUTPUT_DOCX = (
    REPORT_DIR
    / "任务二测试报告_传统方法部分_筋斗与殷麦曼_五次多项式与B样条_V0.3.docx"
)
CONFIG_PATH = PROJECT_DIR / "配置" / "comparison_config.json"
SUMMARY_PATH = COMPARISON_DIR / "comparison_metrics.csv"
RUN_SUMMARY_PATH = COMPARISON_DIR / "run_summary.json"

HELPER_PATH = OUTPUTS_DIR / "任务二测试报告" / "build_task2_report.py"
REFERENCE_DOCX = Path(r"C:\Users\admin\Desktop\固定翼项目评审\任务一测试报告.docx")
REFERENCE_SHA256 = "9DBAC008E633AA9BCE54651AF9A92E0551D0AB45EB7E579029F6C64FF7B9FC30"

QUINTIC_LABEL = "五次多项式方法"
BSPLINE_LABEL = "B样条约束优化方法"
LOOP_NAME = "360度筋斗"
IMMELMANN_NAME = "殷麦曼机动（半筋斗接半滚转）"

plt = None
HELPER = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def metric(
    summary: pd.DataFrame,
    method: str,
    maneuver: str,
    column: str,
) -> float | str:
    row = summary[
        (summary["方法"] == method) & (summary["机动动作"] == maneuver)
    ]
    if len(row) != 1:
        raise KeyError(f"找不到唯一结果: {method}, {maneuver}, {column}")
    return row.iloc[0][column]


def pct_change(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def make_formula(
    name: str,
    latex: str,
    number: str,
    font_size: float = 19,
) -> Path:
    if plt is None:
        raise RuntimeError("公式图片只能在素材生成模式下创建。")
    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    path = FORMULA_DIR / f"{name}.png"
    figure = plt.figure(figsize=(10.8, 0.76))
    figure.patch.set_facecolor("white")
    figure.text(
        0.48,
        0.52,
        f"${latex}$",
        ha="center",
        va="center",
        fontsize=font_size,
        color="#111827",
    )
    figure.text(
        0.965,
        0.52,
        f"({number})",
        ha="right",
        va="center",
        fontsize=12,
        color="#111827",
    )
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return path


def make_formula_images() -> dict[str, Path]:
    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "quintic_curve": make_formula(
            "式6-1_五次多项式轨迹方程",
            r"\mathbf{p}_i(u)=\sum_{k=0}^{5}\mathbf{a}_{i,k}u^k,"
            r"\qquad u\in[0,1],\quad i=0,\ldots,N-1",
            "6-1",
            17,
        ),
        "quintic_boundary": make_formula(
            "式6-2_五次多项式边界约束",
            r"\mathbf{p}_i^{(r)}(1)=\mathbf{p}_{i+1}^{(r)}(0),\ r=0,1,2;"
            r"\qquad[\mathbf{p},\mathbf{p}',\mathbf{p}'']_{0,1}"
            r"=[\mathbf{P},\mathbf{V},\mathbf{A}]_{0,1}",
            "6-2",
            15,
        ),
        "bspline": make_formula(
            "式6-3_B样条曲率参数化",
            r"R_{\mathrm{b}}(\theta)=\sum_{i=0}^{n}N_{i,3}(\theta)P_i,"
            r"\qquad R_{\min}\leq R_{\mathrm{b}}(\theta)\leq R_{\max}",
            "6-3",
            17,
        ),
        "objective": make_formula(
            "式6-4_优化目标函数",
            r"J=w_j\bar j_{\mathrm{rms}}^2+w_n C_{N,\max}^2"
            r"+w_\kappa\overline{\dot\kappa}_{\mathrm{rms}}^2"
            r"+w_r\overline{(R_{\mathrm{b}}-R_{\mathrm{q}})^2}",
            "6-4",
            16,
        ),
        "constraints": make_formula(
            "式6-5_几何与动力学约束",
            r"\mathbf{c}_{\mathrm{eq}}=[L-L_d,\ x_f-x_d,\ z_f-z_d]^T"
            r"=\mathbf{0},\quad R\geq20\,\mathrm{m},\ n\leq3,"
            r"\ M_L\geq0.95",
            "6-5",
            15,
        ),
        "time_law": make_formula(
            "式6-6_速度与时间参数化",
            r"V(\theta)=V_t+\frac{V_b-V_t}{2}(1+\cos\theta),\qquad"
            r"t(\theta)=\int_0^\theta"
            r"\frac{\|\mathbf{p}'(\xi)\|}{V(\xi)}\,d\xi",
            "6-6",
            17,
        ),
        "inverse_dynamics": make_formula(
            "式6-7_参考轨迹逆动力学",
            r"\mathbf{a}=\dot V\,\mathbf{t}+\frac{V^2}{R}\mathbf{n},\qquad"
            r"\mathbf{F}_{\mathrm{req}}=m(\mathbf{a}-\mathbf{g})",
            "6-7",
            17,
        ),
        "aero_metrics": make_formula(
            "式6-8_过载与气动系数",
            r"n=\frac{\|\mathbf{F}_{\perp}\|}{mg},\qquad"
            r"C_{N,\mathrm{req}}=\frac{\|\mathbf{F}_{\perp}\|}"
            r"{\frac{1}{2}\rho V^2S},\qquad"
            r"M_L=\frac{C_{Z,\lim}}{1.2^2C_{N,\mathrm{req}}}",
            "6-8",
            15,
        ),
    }


def read_frames() -> dict[tuple[str, str], pd.DataFrame]:
    mapping = {
        QUINTIC_LABEL: "方法一_五次多项式",
        BSPLINE_LABEL: "方法二_B样条约束优化",
    }
    maneuvers = {
        LOOP_NAME: "360度筋斗",
        IMMELMANN_NAME: "殷麦曼机动",
    }
    frames = {}
    for method, method_dir in mapping.items():
        for maneuver, maneuver_dir in maneuvers.items():
            path = (
                RESULTS_DIR
                / method_dir
                / maneuver_dir
                / "trajectory_attitude.csv"
            )
            frames[(method, maneuver)] = pd.read_csv(
                path,
                encoding="utf-8-sig",
            )
    return frames


def make_method_framework() -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / "图6-1_双传统方法测试框架.png"
    fig, axis = plt.subplots(figsize=(12.8, 5.0))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    boxes = [
        (0.04, 0.37, 0.16, 0.26, "统一输入\n动作边界与飞机参数", "#E0F2FE", "#0369A1"),
        (0.28, 0.63, 0.20, 0.22, "方法一\n分段五次多项式", "#E8EEF5", "#475569"),
        (0.28, 0.15, 0.20, 0.22, "方法二\nB样条约束优化", "#CCFBF1", "#0F766E"),
        (0.57, 0.37, 0.16, 0.26, "统一轨姿输出\n0.02 s时间步长", "#FEF3C7", "#B45309"),
        (0.81, 0.37, 0.15, 0.26, "统一评价\n几何与动力学", "#FEE2E2", "#B91C1C"),
    ]
    for x, y, width, height, text, fill, edge in boxes:
        rectangle = plt.Rectangle(
            (x, y),
            width,
            height,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.8,
        )
        axis.add_patch(rectangle)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=11.5,
            color="#0F172A",
        )
    arrows = [
        ((0.20, 0.53), (0.28, 0.74)),
        ((0.20, 0.47), (0.28, 0.26)),
        ((0.48, 0.74), (0.57, 0.56)),
        ((0.48, 0.26), (0.57, 0.44)),
        ((0.73, 0.50), (0.81, 0.50)),
    ]
    for start, end in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#475569"),
        )
    axis.set_title(
        "两种传统轨姿生成方法的统一测试流程",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_maneuver_definition(
    frames: dict[tuple[str, str], pd.DataFrame],
) -> Path:
    path = ASSET_DIR / "图6-2_筋斗与殷麦曼动作定义.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    for axis, maneuver, title in (
        (axes[0], LOOP_NAME, "360度筋斗：闭合纵向回路"),
        (axes[1], IMMELMANN_NAME, "殷麦曼机动：反向高位退出"),
    ):
        quintic = frames[(QUINTIC_LABEL, maneuver)]
        bspline = frames[(BSPLINE_LABEL, maneuver)]
        axis.plot(
            quintic["x"],
            quintic["z"],
            color="#64748B",
            linestyle="--",
            linewidth=2.0,
            label=QUINTIC_LABEL,
        )
        axis.plot(
            bspline["x"],
            bspline["z"],
            color="#0369A1",
            linewidth=2.2,
            label=BSPLINE_LABEL,
        )
        indices = np.linspace(0, len(bspline) - 1, 8, dtype=int)
        for index in indices:
            velocity = bspline.loc[index, ["vx", "vz"]].to_numpy(dtype=float)
            velocity /= max(np.linalg.norm(velocity), 1e-9)
            axis.arrow(
                bspline["x"].iloc[index],
                bspline["z"].iloc[index],
                2.6 * velocity[0],
                2.6 * velocity[1],
                width=0.10,
                head_width=0.8,
                color="#0F766E",
                length_includes_head=True,
                alpha=0.80,
            )
        axis.scatter(
            [bspline["x"].iloc[0]],
            [bspline["z"].iloc[0]],
            color="#16A34A",
            s=50,
            label="起点",
            zorder=4,
        )
        axis.scatter(
            [bspline["x"].iloc[-1]],
            [bspline["z"].iloc[-1]],
            color="#DC2626",
            marker="X",
            s=55,
            label="终点",
            zorder=4,
        )
        axis.set_title(title)
        axis.set_xlabel("前向位置 x / m")
        axis.set_ylabel("高度 z / m")
        axis.axis("equal")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    fig.suptitle(
        "两种高机动动作及统一边界条件",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_quintic_principle(
    frames: dict[tuple[str, str], pd.DataFrame],
) -> Path:
    path = ASSET_DIR / "图6-3_五次多项式原理.png"
    loop = frames[(QUINTIC_LABEL, LOOP_NAME)]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), constrained_layout=True)
    colored = axes[0].scatter(
        loop["x"],
        loop["z"],
        c=loop["curvature_radius_m"],
        cmap="plasma",
        s=12,
    )
    axes[0].plot(loop["x"], loop["z"], color="#475569", linewidth=0.8)
    node_angles = np.arange(0.0, 361.0, 60.0)
    node_indices = [
        int((loop["path_angle_unwrapped_deg"] - angle).abs().idxmin())
        for angle in node_angles
    ]
    axes[0].scatter(
        loop.loc[node_indices, "x"],
        loop.loc[node_indices, "z"],
        color="#DC2626",
        edgecolor="white",
        linewidth=0.8,
        s=48,
        zorder=5,
        label="分段边界点",
    )
    axes[0].set_title("五次多项式轨迹上的局部曲率半径")
    axes[0].set_xlabel("前向位置 x / m")
    axes[0].set_ylabel("高度 z / m")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8.5)
    bar = fig.colorbar(colored, ax=axes[0], fraction=0.047, pad=0.03)
    bar.set_label("曲率半径 / m")

    axes[1].plot(
        loop["path_angle_unwrapped_deg"],
        loop["curvature_radius_m"],
        color="#475569",
        linewidth=2.2,
        label="五次多项式轨迹",
    )
    for angle in node_angles:
        axes[1].axvline(
            angle,
            color="#CBD5E1",
            linestyle=":",
            linewidth=0.9,
        )
    axes[1].set_title("各五次多项式段的曲率连续性")
    axes[1].set_xlabel("累计航迹转角 / (°)")
    axes[1].set_ylabel("曲率半径 / m")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.suptitle(
        "分段五次多项式轨迹构造",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_immelmann_attitude(
    frames: dict[tuple[str, str], pd.DataFrame],
) -> Path:
    path = ASSET_DIR / "图6-6_殷麦曼姿态时序.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    for method, color, style in (
        (QUINTIC_LABEL, "#64748B", "--"),
        (BSPLINE_LABEL, "#0369A1", "-"),
    ):
        frame = frames[(method, IMMELMANN_NAME)]
        axes[0].plot(
            frame["t"],
            frame["path_angle_unwrapped_deg"],
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=method,
        )
        axes[1].plot(
            frame["t"],
            frame["roll_command_deg"],
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=method,
        )
    axes[0].set_title("殷麦曼机动航迹转角")
    axes[0].set_xlabel("时间 / s")
    axes[0].set_ylabel("累计航迹转角 / (°)")
    axes[1].set_title("顶部半滚转指令")
    axes[1].set_xlabel("时间 / s")
    axes[1].set_ylabel("滚转角 / (°)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8.5)
    fig.suptitle(
        "殷麦曼机动的轨迹与姿态边界",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_optimization_diagnostics(run_summary: dict) -> Path:
    path = ASSET_DIR / "图6-9_B样条优化收敛与计算代价.png"
    loop = run_summary["records"][f"{BSPLINE_LABEL}|loop_360"]["optimization"]
    immelmann = run_summary["records"][f"{BSPLINE_LABEL}|immelmann"]["optimization"]
    labels = ["360度筋斗", "殷麦曼机动"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)

    x = np.arange(2)
    initial = [loop["objective_initial"], immelmann["objective_initial"]]
    final = [loop["objective_final"], immelmann["objective_final"]]
    width = 0.34
    axes[0].bar(
        x - width / 2,
        initial,
        width,
        color="#94A3B8",
        label="初始目标值",
    )
    axes[0].bar(
        x + width / 2,
        final,
        width,
        color="#0369A1",
        label="最终目标值",
    )
    axes[0].set_xticks(x, ["筋斗", "殷麦曼机动"])
    axes[0].set_title("归一化目标函数")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    bars = axes[1].bar(
        labels,
        [loop["iterations"], immelmann["iterations"]],
        color=["#0F766E", "#2DD4BF"],
    )
    axes[1].bar_label(bars, fmt="%.0f", padding=3)
    axes[1].set_title("优化迭代次数")
    axes[1].set_ylabel("迭代次数")
    axes[1].tick_params(axis="x", rotation=8)
    axes[1].grid(axis="y", alpha=0.25)

    bars = axes[2].bar(
        labels,
        [loop["generation_time_s"], immelmann["generation_time_s"]],
        color=["#B45309", "#F59E0B"],
    )
    axes[2].bar_label(bars, fmt="%.3f", padding=3)
    axes[2].set_title("单次优化计算时间")
    axes[2].set_ylabel("时间 / s")
    axes[2].tick_params(axis="x", rotation=8)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "B样条约束优化的数值求解表现",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_force_decomposition() -> Path:
    path = ASSET_DIR / "图6-11_参考轨迹受力分解.png"
    fig, axis = plt.subplots(figsize=(10.8, 5.3))
    axis.set_xlim(-1.1, 1.2)
    axis.set_ylim(-0.75, 1.0)
    axis.set_aspect("equal")
    axis.axis("off")

    center = np.array([0.0, 0.0])
    tangent = np.array([0.78, 0.42])
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    axis.plot(
        [-0.95, 0.95],
        [-0.51, 0.51],
        color="#94A3B8",
        linestyle="--",
        linewidth=1.5,
    )
    axis.scatter([0.0], [0.0], s=250, color="#0369A1", zorder=5)
    axis.text(0.0, -0.10, "无人机质心", ha="center", va="top", fontsize=11)

    vectors = [
        (tangent * 0.92, "切向加速度 $\\dot V\\,\\mathbf{t}$", "#B45309"),
        (normal * 0.84, "法向加速度 $V^2/R\\,\\mathbf{n}$", "#0F766E"),
        (np.array([0.0, -0.68]), "重力 $m\\mathbf{g}$", "#DC2626"),
        (normal * 0.58 + tangent * 0.23, "所需气动力/推力合量", "#7C3AED"),
    ]
    for vector, label, color in vectors:
        axis.annotate(
            "",
            xy=vector,
            xytext=center,
            arrowprops=dict(arrowstyle="-|>", lw=2.3, color=color),
        )
        offset = 0.06 * vector / max(np.linalg.norm(vector), 1e-9)
        label_position = vector + offset
        axis.text(
            label_position[0],
            label_position[1],
            label,
            color=color,
            fontsize=10.5,
            ha="left" if vector[0] >= 0 else "right",
            va="bottom" if vector[1] >= 0 else "top",
        )
    axis.set_title(
        "参考轨迹逆动力学中的切向—法向受力分解",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_loop_phase_analysis(
    frames: dict[tuple[str, str], pd.DataFrame],
) -> Path:
    path = ASSET_DIR / "图6-13_筋斗关键阶段动力学.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    colors = {
        QUINTIC_LABEL: "#64748B",
        BSPLINE_LABEL: "#0369A1",
    }
    styles = {QUINTIC_LABEL: "--", BSPLINE_LABEL: "-"}
    for method in (QUINTIC_LABEL, BSPLINE_LABEL):
        frame = frames[(method, LOOP_NAME)]
        angle = frame["path_angle_unwrapped_deg"]
        axes[0].plot(
            angle,
            frame["normal_load_factor"],
            color=colors[method],
            linestyle=styles[method],
            linewidth=2.0,
            label=method,
        )
        axes[1].plot(
            angle,
            frame["estimated_alpha_required_deg"],
            color=colors[method],
            linestyle=styles[method],
            linewidth=2.0,
            label=method,
        )
    for axis in axes:
        for angle, label in ((0, "底部"), (90, "上升侧"), (180, "顶部"), (270, "下降侧"), (360, "出口")):
            axis.axvline(angle, color="#CBD5E1", linewidth=0.8)
            axis.text(
                angle,
                0.98,
                label,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#475569",
            )
        axis.set_xlabel("累计航迹转角 / (°)")
        axis.grid(alpha=0.22)
        axis.legend(loc="best", fontsize=8)
    axes[0].set_title("法向过载在动作各阶段的分布")
    axes[0].set_ylabel("过载系数 / g")
    axes[1].set_title("估算迎角需求在动作各阶段的分布")
    axes[1].set_ylabel("迎角 / (°)")
    fig.suptitle(
        "360度筋斗关键阶段动力学需求",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_roll_coupling(
    frames: dict[tuple[str, str], pd.DataFrame],
) -> Path:
    path = ASSET_DIR / "图6-15_殷麦曼滚转耦合.png"
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), constrained_layout=True)
    for method, color, style in (
        (QUINTIC_LABEL, "#64748B", "--"),
        (BSPLINE_LABEL, "#0369A1", "-"),
    ):
        frame = frames[(method, IMMELMANN_NAME)]
        angle = frame["path_angle_unwrapped_deg"]
        axes[0].plot(
            angle,
            frame["roll_command_deg"],
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=method,
        )
        axes[1].plot(
            angle,
            frame["p_degps"],
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=method,
        )
        axes[2].plot(
            angle,
            frame["required_side_force_coefficient"],
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=method,
        )
    for axis in axes:
        axis.axvspan(120.0, 180.0, color="#FEF3C7", alpha=0.45)
        axis.set_xlabel("累计航迹转角 / (°)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=7.8)
    axes[0].set_title("滚转角指令")
    axes[0].set_ylabel("滚转角 / (°)")
    axes[1].set_title("机体系滚转角速度")
    axes[1].set_ylabel("角速度 / (°/s)")
    axes[2].set_title("侧向力系数需求")
    axes[2].set_ylabel("系数")
    fig.suptitle(
        "殷麦曼机动顶部120°～180°区段的滚转耦合",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_control_interface() -> Path:
    path = ASSET_DIR / "图6-16_轨姿到控制验证接口.png"
    fig, axis = plt.subplots(figsize=(12.8, 4.2))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    labels = [
        "参考轨姿\n位置、速度、航迹角、滚转角",
        "轨迹跟踪器\n位置/速度误差",
        "姿态控制器\n角度/角速度误差",
        "执行机构模型\n舵面与推进系统",
        "六自由度飞机\n闭环响应",
        "评价输出\n跟踪误差与约束违反",
    ]
    fills = ["#E0F2FE", "#CCFBF1", "#FEF3C7", "#FDE68A", "#EDE9FE", "#FEE2E2"]
    edges = ["#0369A1", "#0F766E", "#B45309", "#A16207", "#7C3AED", "#B91C1C"]
    centers = np.linspace(0.085, 0.915, len(labels))
    for index, (center, label) in enumerate(zip(centers, labels)):
        rectangle = plt.Rectangle(
            (center - 0.067, 0.31),
            0.134,
            0.38,
            facecolor=fills[index],
            edgecolor=edges[index],
            linewidth=1.6,
        )
        axis.add_patch(rectangle)
        axis.text(
            center,
            0.50,
            label,
            ha="center",
            va="center",
            fontsize=9.3,
            color="#0F172A",
        )
        if index < len(labels) - 1:
            axis.annotate(
                "",
                xy=(centers[index + 1] - 0.073, 0.50),
                xytext=(center + 0.073, 0.50),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#475569"),
            )
    axis.set_title(
        "生成轨姿进入后续闭环跟踪验证的接口关系",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(
    frames: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, Path]:
    run_summary = load_json(RUN_SUMMARY_PATH)
    figures = {
        "framework": make_method_framework(),
        "maneuver_definition": make_maneuver_definition(frames),
        "quintic_principle": make_quintic_principle(frames),
        "bspline_principle": COMPARISON_DIR / "bspline_principle.png",
        "immelmann_attitude": make_immelmann_attitude(frames),
        "loop_geometry": COMPARISON_DIR / "loop_geometry_comparison.png",
        "immelmann_geometry": COMPARISON_DIR / "immelmann_geometry_comparison.png",
        "key_metrics": COMPARISON_DIR / "key_metrics_comparison.png",
        "optimization_diagnostics": make_optimization_diagnostics(run_summary),
        "force_decomposition": make_force_decomposition(),
        "loop_dynamics": COMPARISON_DIR / "loop_dynamics_comparison.png",
        "loop_phase": make_loop_phase_analysis(frames),
        "immelmann_dynamics": COMPARISON_DIR / "immelmann_dynamics_comparison.png",
        "roll_coupling": make_roll_coupling(frames),
        "control_interface": make_control_interface(),
    }
    for key, path in figures.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少报告插图 {key}: {path}")
    return figures


def figure_paths() -> dict[str, Path]:
    figures = {
        "framework": ASSET_DIR / "图6-1_双传统方法测试框架.png",
        "maneuver_definition": ASSET_DIR / "图6-2_筋斗与殷麦曼动作定义.png",
        "quintic_principle": ASSET_DIR / "图6-3_五次多项式原理.png",
        "bspline_principle": COMPARISON_DIR / "bspline_principle.png",
        "immelmann_attitude": ASSET_DIR / "图6-6_殷麦曼姿态时序.png",
        "loop_geometry": COMPARISON_DIR / "loop_geometry_comparison.png",
        "immelmann_geometry": COMPARISON_DIR / "immelmann_geometry_comparison.png",
        "key_metrics": COMPARISON_DIR / "key_metrics_comparison.png",
        "optimization_diagnostics": ASSET_DIR / "图6-9_B样条优化收敛与计算代价.png",
        "force_decomposition": ASSET_DIR / "图6-11_参考轨迹受力分解.png",
        "loop_dynamics": COMPARISON_DIR / "loop_dynamics_comparison.png",
        "loop_phase": ASSET_DIR / "图6-13_筋斗关键阶段动力学.png",
        "immelmann_dynamics": COMPARISON_DIR / "immelmann_dynamics_comparison.png",
        "roll_coupling": ASSET_DIR / "图6-15_殷麦曼滚转耦合.png",
        "control_interface": ASSET_DIR / "图6-16_轨姿到控制验证接口.png",
    }
    for key, path in figures.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少报告插图 {key}: {path}")
    return figures


def build_pages(
    config: dict,
    summary: pd.DataFrame,
    run_summary: dict,
) -> list[list[tuple]]:
    loop_a = {
        column: metric(summary, QUINTIC_LABEL, LOOP_NAME, column)
        for column in summary.columns
    }
    loop_b = {
        column: metric(summary, BSPLINE_LABEL, LOOP_NAME, column)
        for column in summary.columns
    }
    immelmann_a = {
        column: metric(summary, QUINTIC_LABEL, IMMELMANN_NAME, column)
        for column in summary.columns
    }
    immelmann_b = {
        column: metric(summary, BSPLINE_LABEL, IMMELMANN_NAME, column)
        for column in summary.columns
    }

    loop_jerk_reduction = -pct_change(
        float(loop_b["均方根jerk_mps3"]),
        float(loop_a["均方根jerk_mps3"]),
    )
    immelmann_jerk_reduction = -pct_change(
        float(immelmann_b["均方根jerk_mps3"]),
        float(immelmann_a["均方根jerk_mps3"]),
    )
    loop_curvature_reduction = -pct_change(
        float(loop_b["均方根曲率变化率_1pm_s"]),
        float(loop_a["均方根曲率变化率_1pm_s"]),
    )
    immelmann_curvature_reduction = -pct_change(
        float(immelmann_b["均方根曲率变化率_1pm_s"]),
        float(immelmann_a["均方根曲率变化率_1pm_s"]),
    )
    loop_margin_gain = pct_change(
        float(loop_b["最小含速度裕度"]),
        float(loop_a["最小含速度裕度"]),
    )
    immelmann_margin_gain = pct_change(
        float(immelmann_b["最小含速度裕度"]),
        float(immelmann_a["最小含速度裕度"]),
    )

    aircraft = run_summary["records"][
        f"{QUINTIC_LABEL}|loop_360"
    ]["aircraft"]
    loop_opt = run_summary["records"][
        f"{BSPLINE_LABEL}|loop_360"
    ]["optimization"]
    immelmann_opt = run_summary["records"][
        f"{BSPLINE_LABEL}|immelmann"
    ]["optimization"]

    test_rows = [
        ["动作", "360度筋斗", "360°闭合纵向回路", "终点与起点重合"],
        [
            "动作",
            "殷麦曼机动",
            "180°纵向转向并升高64 m",
            "顶部保留180°半滚转",
        ],
        ["飞机", "质量与机翼面积", "3.0 kg；0.24 m²", "用户确认参数"],
        ["几何", "平均尺度", "平均曲率半径32 m", "两种方法同尺度"],
        ["速度", "统一速度规律", "17.5～20.0 m/s", "底部快、顶部慢"],
        ["采样", "输出时间步长", "0.02 s", "统一CSV时间序列"],
        ["方法", "五次多项式方法", "分段五次多项式", "确定性基准"],
        ["方法", "B样条方法", "三次B样条+SLSQP", "约束优化基准"],
    ]
    threshold_rows = [
        ["几何", "终端位置误差", "≤0.05 m", "动作边界一致性"],
        ["几何", "终端角度误差", "≤0.5°", "航迹转角与滚转角"],
        ["几何", "最小曲率半径", "≥20 m", "规划级转弯能力"],
        ["包线", "速度范围", "16.5～20.5 m/s", "当前初步飞行包线"],
        ["载荷", "最大法向过载", "≤3.0 g", "结构限制待实测确认"],
        ["气动", "估算迎角需求", "≤15°", "VLM附着流范围筛查"],
        ["气动", "计入速度裕度的升力裕度", "≥0.95", "1.2倍速度裕度"],
        ["角速率", "最大俯仰/滚转角速度", "≤120/240°/s", "工程假设阈值"],
        ["耦合", "最大侧向力系数", "≤0.25", "顶部滚转耦合筛查"],
    ]
    result_rows = [
        [
            "360度筋斗",
            "五次多项式",
            (
                f"闭合误差{float(loop_a['终端位置误差_m']):.3f} m；"
                f"最小半径{float(loop_a['最小曲率半径_m']):.2f} m"
            ),
            (
                f"{float(loop_a['最大过载_g']):.3f} g；"
                f"迎角{float(loop_a['最大迎角需求_deg']):.2f}°；"
                f"裕度{float(loop_a['最小含速度裕度']):.3f}"
            ),
            str(loop_a["规划级结论"]),
        ],
        [
            "360度筋斗",
            "B样条",
            (
                f"闭合误差{float(loop_b['终端位置误差_m']):.3f} m；"
                f"最小半径{float(loop_b['最小曲率半径_m']):.2f} m"
            ),
            (
                f"{float(loop_b['最大过载_g']):.3f} g；"
                f"迎角{float(loop_b['最大迎角需求_deg']):.2f}°；"
                f"裕度{float(loop_b['最小含速度裕度']):.3f}"
            ),
            str(loop_b["规划级结论"]),
        ],
        [
            "殷麦曼机动",
            "五次多项式",
            (
                f"终端误差{float(immelmann_a['终端位置误差_m']):.3f} m；"
                f"最小半径{float(immelmann_a['最小曲率半径_m']):.2f} m"
            ),
            (
                f"{float(immelmann_a['最大过载_g']):.3f} g；"
                f"滚转率{float(immelmann_a['最大滚转角速度_degps']):.1f}°/s；"
                f"裕度{float(immelmann_a['最小含速度裕度']):.3f}"
            ),
            str(immelmann_a["规划级结论"]),
        ],
        [
            "殷麦曼机动",
            "B样条",
            (
                f"终端误差{float(immelmann_b['终端位置误差_m']):.3f} m；"
                f"最小半径{float(immelmann_b['最小曲率半径_m']):.2f} m"
            ),
            (
                f"{float(immelmann_b['最大过载_g']):.3f} g；"
                f"滚转率{float(immelmann_b['最大滚转角速度_degps']):.1f}°/s；"
                f"裕度{float(immelmann_b['最小含速度裕度']):.3f}"
            ),
            str(immelmann_b["规划级结论"]),
        ],
    ]

    pages: list[list[tuple]] = [
        [
            ("h1", "任务二测试报告：传统轨姿生成方法"),
            ("h2", "360度筋斗与殷麦曼机动双方法测试"),
            (
                "p",
                "本部分面向微小型固定翼无人机高机动轨迹—姿态生成任务，"
                "选取360度筋斗与殷麦曼机动作为测试动作，对两类传统方法进行统一实现和对比。"
                "第一类方法采用分段五次多项式轨迹直接构造动作；第二类方法采用三次B样条描述"
                "曲率半径，并通过非线性约束优化确定控制节点。两种方法均输出时间、位置、速度、"
                "加速度、航迹转角、滚转指令及动力学需求量，供后续跟踪控制设计使用。",
            ),
            (
                "status",
                "报告结构",
                "全文只按三条主线展开：6.1原理概述，6.2生成轨迹的几何对比与指标评估，"
                "6.3实际飞机动力学的表现与评估。所有测试参数均在对应方法和测试条件中说明。",
            ),
            ("figure", "framework", "图6-1 双传统方法统一测试框架", 5.65),
            (
                "p",
                "当前评估等级为规划级。已使用质量、机翼几何、转动惯量和VLM气动系数表，"
                "但尚未取得经确认的推力—油门—空速曲线、舵面限位、舵机速率、实测失速边界"
                "和结构允许过载。因此，报告结论用于判断参考轨迹是否值得进入闭环仿真，"
                "不等同于实飞安全放行。",
            ),
        ],
        [
            ("h2", "6.1 原理概述"),
            ("h3", "6.1.1 测试对象与动作定义"),
            (
                "p",
                "筋斗和殷麦曼机动均在由初始前向轴x与竖直向上轴z构成的平面内生成。"
                "360度筋斗要求无人机从水平飞行状态进入纵向闭合回路，累计航迹转角达到360°，"
                "最终位置和速度方向回到起始边界。殷麦曼机动由前半个筋斗与顶部半滚转组成："
                "质心轨迹在纵向平面内完成180°转向并升高64 m，速度方向与初始方向相反；"
                "同时在航迹转角120°～180°阶段完成0°到180°平滑滚转。测试时把这两部分"
                "作为同一个轨迹—姿态动作评价，而不是把殷麦曼简化成单独的半筋斗轨迹。",
            ),
            ("figure", "maneuver_definition", "图6-2 筋斗与殷麦曼动作定义", 5.7),
            (
                "p",
                "选择这两种动作具有递进关系：殷麦曼机动能够检查大俯仰、高度快速变化、"
                "反向高位退出及顶部滚转耦合；完整筋斗在此基础上进一步检查轨迹闭合、"
                "整周姿态连续和上下半程载荷变化。两者共享同一纵向几何尺度和速度规律，"
                "便于分析动作长度加倍后对方法性能的影响。",
            ),
        ],
        [
            ("h3", "6.1.2 坐标、状态与轨姿输出定义"),
            (
                "p",
                "本测试采用局部右手直角坐标系。x轴沿动作开始时的飞行方向，z轴竖直向上，"
                "y轴垂直于筋斗平面。由于两种动作的几何主运动发生在x-z平面，理想轨迹的"
                "y坐标保持不变；但姿态并不等同于平面位置，殷麦曼机动顶部滚转时机翼方向会绕"
                "速度切向旋转，因此仍需独立给出滚转角和机体系角速度。",
            ),
            (
                "p",
                "轨迹位置由r(t)=[x(t),y(t),z(t)]给出，速度由位置对时间的一阶导数得到，"
                "加速度和jerk分别对应二阶、三阶时间导数。航迹转角θ用于描述速度方向在"
                "竖直平面内的累计旋转：θ=0°表示水平前飞，θ=90°表示竖直上升，"
                "θ=180°表示水平反向飞行，θ=270°表示竖直下降，θ=360°重新回到初始方向。"
                "这种不折返的累计角定义能够避免普通俯仰角在±180°处跳变，便于评价完整筋斗。",
            ),
            (
                "p",
                "姿态输出采用“航迹方向+滚转指令”的任务级表达。机体纵轴与参考速度方向对齐，"
                "机翼法向首先与轨迹主法向一致，再按滚转指令绕速度方向旋转。"
                "完整筋斗滚转角保持0°；殷麦曼机动在顶部完成180°滚转后，机翼姿态与反向水平"
                "飞行状态相匹配。该表达足以为后续位置—姿态分层控制器提供参考，"
                "同时避免在测试报告中引入尚未确定的完整姿态表示方案。",
            ),
            (
                "status",
                "输出含义",
                "几何轨迹回答“飞机质心沿什么路径运动”；姿态时序回答“飞机沿该路径时机翼"
                "如何定向”。两者必须按同一时间轴输出，才能用于实际跟踪控制。",
            ),
        ],
        [
            ("h3", "6.1.3 方法一：分段五次多项式轨迹"),
            (
                "p",
                "方法一采用分段五次多项式直接描述机动平面内的位置。对第i段定义归一化参数"
                "u∈[0,1]，前向位置xi(u)和高度zi(u)均写成五次多项式，因此每个坐标各有"
                "6个待定系数。360度筋斗按60°航迹转角划分为6段，殷麦曼的半筋斗阶段"
                "划分为3段。这样的分段数量既能布置底部、侧部、顶部等关键边界，又不会"
                "因航点过密而引入不必要的局部振荡。",
            ),
            ("formula", "quintic_curve", "分段五次多项式轨迹方程"),
            (
                "p",
                "每段多项式的6个系数由段首和段末的位置、一阶导数、二阶导数唯一确定。"
                "相邻两段共用同一边界位置、切向导数和二阶导数，因而整条轨迹达到C²连续。"
                "位置连续保证轨迹不间断，一阶导数连续避免速度方向突变，二阶导数连续则"
                "抑制曲率和法向加速度在分段点发生跳变。边界点按照32 m特征半径布置，"
                "并对侧部曲率进行适度调节，使轨迹满足闭合或高位反向退出条件。",
            ),
            ("formula", "quintic_boundary", "五次多项式端点与段间连续条件"),
            (
                "p",
                "多项式求得后，程序直接由p'(θ)计算切向方向和弧长微元，由p'(θ)、p''(θ)"
                "计算曲率κ=(x'z''-z'x'')/||p'||³。当前五次多项式筋斗和殷麦曼的"
                "最小曲率半径均为约23.87 m，最大值约40.12 m，均高于20 m几何筛查线。"
                "该方法无需迭代优化，重复运行结果确定，适合作为B样条方法和后续扩散模型的"
                "稳定传统基线。",
            ),
            ("figure", "quintic_principle", "图6-3 分段五次多项式轨迹原理", 5.65),
        ],
        [
            ("h3", "6.1.4 方法二：B样条曲率参数化"),
            (
                "p",
                "B样条方法仍以累计航迹转角作为轨迹推进变量，但不再预先指定固定谐波形状，"
                "而是把曲率半径表示为三次B样条基函数的线性组合。Pi为待优化节点值，"
                "Ni,3(θ)为三次B样条基函数。每个节点主要影响附近区段，因此可以在不显著"
                "扰动整条轨迹的情况下局部改变曲率分配；三次基函数同时保证半径曲线具有"
                "足够的连续导数，可避免逐段圆弧在连接处出现曲率跳变。",
            ),
            ("formula", "bspline", "B样条曲率半径参数化"),
            (
                "p",
                "完整筋斗使用12个周期节点，并在0°与360°处施加周期连续条件，"
                "保证闭合点两侧曲率及其变化趋势一致。殷麦曼机动使用9个开式节点，"
                "端点的一阶导数设置为零，以降低动作进入和退出时的曲率变化。"
                "节点初值来自五次多项式轨迹，经线性投影后先满足路径长度与终端位置条件，"
                "再交由非线性优化器调整。",
            ),
            ("figure", "bspline_principle", "图6-4 B样条曲率参数化与节点优化结果", 5.7),
            (
                "p",
                f"优化后，筋斗曲率半径范围收敛到"
                f"{float(loop_b['最小曲率半径_m']):.2f}～"
                f"{float(loop_b['最大曲率半径_m']):.2f} m；殷麦曼机动收敛到"
                f"{float(immelmann_b['最小曲率半径_m']):.2f}～"
                f"{float(immelmann_b['最大曲率半径_m']):.2f} m。"
                "与五次多项式方法约23.87～40.12 m的局部变化相比，优化后的半径分布更加集中，"
                "说明优化器倾向于用较平缓的曲率变化满足同一动作边界。",
            ),
        ],
        [
            ("h3", "6.1.5 B样条约束优化模型"),
            (
                "p",
                "B样条节点不是通过人工观察曲线后逐点调整，而是由明确的目标函数和约束求得。"
                "目标函数包含四项：轨迹jerk均方根、所需法向力系数峰值、曲率变化率均方根，"
                "以及相对五次多项式初值的形状偏离。前三项分别对应控制输入变化、气动负担和几何"
                "平滑性；最后一项用于抑制无必要的形状漂移，使数值方法仍保留清晰的筋斗特征。",
            ),
            ("formula", "objective", "B样条节点优化目标函数"),
            (
                "p",
                "优化过程中同时施加等式约束与不等式约束。等式约束保证路径长度、"
                "终端前向位置和终端高度严格满足动作定义；完整筋斗由此回到起点，"
                "殷麦曼机动由此在前向位置回到零的同时升高64 m。不等式约束限制最小曲率半径、"
                "法向过载、最大气动力系数以及航迹俯仰角速度。升力裕度下限被直接写入"
                "优化约束，而不只是生成后的被动检查。",
            ),
            ("formula", "constraints", "B样条优化中的主要等式与不等式约束"),
            (
                "p",
                f"数值求解采用SLSQP算法。筋斗在{int(loop_opt['iterations'])}次迭代、"
                f"{int(loop_opt['function_evaluations'])}次目标函数计算后收敛，"
                f"求解时间约{float(loop_b['生成时间_s']):.3f} s；殷麦曼机动在"
                f"{int(immelmann_opt['iterations'])}次迭代、"
                f"{int(immelmann_opt['function_evaluations'])}次目标函数计算后收敛，"
                f"求解时间约{float(immelmann_b['生成时间_s']):.3f} s。"
                "所有等式约束残差均保持在数值容差范围内，说明优化结果不是先生成曲线后"
                "强行移动终点，而是在求解过程中满足动作边界。",
            ),
        ],
        [
            ("h3", "6.1.6 时间参数化与姿态边界"),
            (
                "p",
                "为了使两种方法的差异集中在空间轨迹生成机制，本测试使用相同速度规律。"
                "无人机在动作底部取20.0 m/s，在顶部取17.5 m/s，速度随累计航迹转角"
                "按余弦规律连续变化。五次多项式轨迹按dt/dθ=||p'(θ)||/V积分，"
                "B样条曲率轨迹按dt/dθ=R(θ)/V积分。随后两种结果均以0.02 s间隔"
                "重采样，形成控制算法可直接读取的时间序列。",
            ),
            ("formula", "time_law", "统一速度规律与时间参数化"),
            (
                "p",
                "完整筋斗不额外施加滚转指令，机翼方向保持垂直于机动平面。"
                "殷麦曼机动在顶部低过载区间执行180°半滚转，滚转角采用五次平滑函数，"
                "使滚转开始和结束时的角速度、角加速度连续回到零。"
                "两种传统方法使用相同滚转时序，因此殷麦曼机动的差异主要来自纵向曲率分配，"
                "不会被不同姿态指令混入比较。",
            ),
            ("figure", "immelmann_attitude", "图6-5 殷麦曼机动航迹转角与顶部半滚转时序", 5.65),
            (
                "p",
                "报告只使用航迹转角、滚转角和机体系角速度描述轨姿输出，"
                "不展开尚未确定的完整姿态参数化理论。程序内部只需保持姿态序列连续"
                "并能够计算角速度，该实现细节不作为本次传统方法测试的理论重点。",
            ),
        ],
        [
            ("h3", "6.1.7 统一输出与评价链"),
            (
                "p",
                "每条轨迹按照同一字段保存，包括时间t、位置x/y/z、速度vx/vy/vz、"
                "加速度、速度大小、累计航迹转角、滚转指令、机体系角速度、曲率半径、"
                "jerk、法向过载、气动力系数需求、估算迎角、升力裕度以及切向力需求。"
                "统一格式使几何评价与动力学评价针对同一条时间化轨迹完成，避免出现"
                "几何图使用一组数据、动力学计算又使用另一组数据的情况。",
            ),
            (
                "p",
                "评价链分为两个层级。第一层只检查动作形状和边界，包括终端位置、"
                "终端航迹转角、终端滚转角、动作平面偏差、路径长度和最小曲率半径。"
                "第二层将轨迹速度、加速度和姿态方向代入参考轨迹逆动力学，计算维持该"
                "运动所需的法向力、切向力、侧向力和角速度，再与当前气动数据和工程阈值比较。"
            ),
            ("formula", "inverse_dynamics", "时间化轨迹的逆动力学关系"),
            (
                "status",
                "方法比较原则",
                "同一动作、同一飞机、同一速度、同一采样与同一阈值。五次多项式轨迹作为确定性基准，"
                "B样条轨迹只能用五次多项式结果初始化，不能直接复制五次多项式结果后宣称优化有效。",
            ),
            (
                "p",
                "从工程定位看，五次多项式方法回答“能否快速构造一个边界正确的动作”，"
                "B样条方法回答“在边界和动力学限制不变时，能否重新分配曲率以改善综合指标”。"
                "这种分工使两类方法的优缺点可以由测试结果直接呈现，而不是预设B样条必须在"
                "所有指标上优于五次多项式方法。",
            ),
        ],
        [
            ("h3", "6.1.8 两种传统方法的理论差异与适用场景"),
            (
                "p",
                "五次多项式属于边界插值型构造方法。设计者先确定分段边界位置、切向和"
                "二阶导数，再由线性方程直接求得每段系数。"
                "其主要误差来源不是数值优化失败，而是边界条件和分段数量能否表达期望动作。"
                "只要参数有效，算法每次运行都给出相同结果，因此特别适合用于基准测试、"
                "在线快速重规划以及后续算法异常时的确定性回退。",
            ),
            (
                "p",
                "B样条方法属于参数化数值优化方法。动作边界不再完全依赖某个固定闭式函数，"
                "而由等式约束保证；平滑性和动力学裕度通过目标函数与不等式约束共同调节。"
                "它可以把局部曲率从载荷敏感区转移到裕度较大的区段，但求解结果依赖初值、"
                "权重和优化器收敛。为降低这种依赖，本测试使用五次多项式轨迹作为可行初值，"
                "并保存迭代次数、函数计算次数和约束残差作为可重复性证据。",
            ),
            (
                "p",
                "从后续扩散模型研究角度看，这两种传统方法提供了互补的数据来源。"
                "五次多项式方法能够快速生成大量结构一致的动作样本，适合覆盖速度、半径和高度等"
                "基本参数；B样条方法能够对这些样本进一步做约束修正和质量筛选，"
                "适合作为训练集中的高质量参考。扩散模型后续是否具有优势，应与两类传统"
                "基线分别比较，而不能只选择较弱基线。",
            ),
            (
                "status",
                "方法定位",
                "五次多项式方法强调快速、确定和可解释；B样条方法强调约束处理、局部可调和综合质量。"
                "本报告比较的是工程取舍，不把某一种方法预设为绝对最优。",
            ),
        ],
        [
            ("h2", "6.2 生成轨迹的几何对比与指标评估"),
            ("h3", "6.2.1 测试方法与判定指标"),
            (
                "p",
                "几何测试以动作定义是否被准确实现为第一目标，以轨迹是否连续、"
                "是否具备可接受的弯曲尺度和时间平滑性为第二目标。每种方法分别生成筋斗和"
                "殷麦曼机动，共形成四组轨姿时间序列。测试程序从CSV重新计算终端误差、路径长度、"
                "动作宽度与高度、曲率半径、jerk以及曲率变化率，并与配置阈值逐项比较。",
            ),
            (
                "table",
                ["类别", "测试项", "统一设置", "说明"],
                test_rows,
                [0.8, 1.55, 1.85, 2.05],
                8.2,
            ),
            (
                "p",
                "其中，终端位置误差和终端角度误差用于检验动作边界；最小曲率半径用于避免"
                "局部弯曲过急；jerk用于衡量加速度变化强度；曲率变化率用于衡量几何转弯需求"
                "变化的快慢。最大jerk容易受单个采样点影响，因此报告同时采用均方根jerk"
                "描述整段动作的总体平滑程度。",
            ),
            (
                "p",
                "对于完整筋斗，闭合误差必须在位置层面单独检查，不能只凭轨迹图目测。"
                "对于殷麦曼机动，终点与起点本就不重合，因此检查目标是前向位置回到零、"
                "高度增加64 m、航迹方向反转及顶部滚转完成。两种动作都要求y方向偏差接近零，"
                "以确认轨迹保持在指定竖直机动平面内。",
            ),
        ],
        [
            ("h3", "6.2.2 360度筋斗几何对比"),
            (
                "p",
                f"两种方法均生成约{float(loop_a['轨迹长度_m']):.3f} m的闭合回路。"
                f"五次多项式方法持续{float(loop_a['持续时间_s']):.3f} s，B样条方法持续"
                f"{float(loop_b['持续时间_s']):.3f} s；由于路径长度和速度规律保持一致，"
                "两者持续时间几乎相同。终端位置误差均为"
                f"{float(loop_b['终端位置误差_m']):.3f} m，累计航迹转角达到360°，"
                "说明B样条优化没有以牺牲动作闭合为代价改善平滑性。",
            ),
            ("figure", "loop_geometry", "图6-6 360度筋斗几何与平滑性对比", 5.75),
            (
                "p",
                f"五次多项式方法的曲率半径在{float(loop_a['最小曲率半径_m']):.2f}～"
                f"{float(loop_a['最大曲率半径_m']):.2f} m之间变化；B样条方法"
                f"将其收缩到{float(loop_b['最小曲率半径_m']):.2f}～"
                f"{float(loop_b['最大曲率半径_m']):.2f} m。"
                "两条轨迹的总体宽度和高度接近，但B样条轨迹更接近均匀回路，"
                "局部曲率的峰谷明显减少。",
            ),
            (
                "p",
                f"五次多项式方法的均方根jerk为{float(loop_a['均方根jerk_mps3']):.3f} m/s³，"
                f"B样条方法降至{float(loop_b['均方根jerk_mps3']):.3f} m/s³，"
                f"下降{loop_jerk_reduction:.2f}%；均方根曲率变化率由"
                f"{float(loop_a['均方根曲率变化率_1pm_s']):.6f}降至"
                f"{float(loop_b['均方根曲率变化率_1pm_s']):.6f} 1/(m·s)，"
                f"下降{loop_curvature_reduction:.2f}%。因此，B样条方法在保持闭合边界和"
                "路径尺度的同时，显著降低了整周动作中的几何变化强度。",
            ),
        ],
        [
            ("h3", "6.2.3 360度筋斗平滑性与控制意义"),
            (
                "p",
                "仅从闭合轨迹轮廓看，两种筋斗都满足任务要求，但控制器真正需要跟踪的是"
                "随时间变化的位置、速度和加速度。五次多项式各段虽然达到C²连续，但分段边界"
                "给定的二阶导数会在一周内形成若干曲率峰谷，使法向加速度V²/R及其变化率起伏；"
                "这种起伏不会造成位置或切向不连续，"
                "却会使升降舵指令和俯仰角速度需求更频繁地调整。B样条优化降低半径变化幅度后，"
                "加速度方向和大小的变化更均匀。",
            ),
            (
                "p",
                f"最大jerk从{float(loop_a['最大jerk_mps3']):.3f}降至"
                f"{float(loop_b['最大jerk_mps3']):.3f} m/s³，下降"
                f"{-pct_change(float(loop_b['最大jerk_mps3']), float(loop_a['最大jerk_mps3'])):.2f}%；"
                f"均方根jerk下降{loop_jerk_reduction:.2f}%。峰值指标反映最不利瞬间，"
                "均方根指标反映全动作平均强度，两者同步下降说明B样条没有通过牺牲少数区段"
                "来换取平均指标改善。",
            ),
            (
                "p",
                f"曲率变化率均方根下降{loop_curvature_reduction:.2f}%，"
                "其改善幅度大于jerk。这是因为jerk除曲率变化外还受统一速度规律、重力方向"
                "投影和时间重采样影响，不能随曲率变化率按同一比例下降。"
                "该差异也说明几何指标和动力学指标不能互相替代：曲率更平滑通常有利于控制，"
                "但最终仍需用加速度、jerk和气动力需求进行验证。",
            ),
            (
                "status",
                "控制意义",
                "B样条筋斗降低了参考加速度的变化强度，预期可减小控制器的快速舵面修正需求。"
                "该预期仍需在加入舵机速率和六自由度模型后，用实际跟踪误差验证。",
            ),
        ],
        [
            ("h3", "6.2.4 殷麦曼机动几何对比"),
            (
                "p",
                f"殷麦曼机动路径长度为{float(immelmann_a['轨迹长度_m']):.3f} m，"
                f"持续时间约{float(immelmann_a['持续时间_s']):.3f} s。"
                "两种方法均从100 m初始高度进入动作，并在164 m高度反向退出；"
                "前向终端位置回到起点投影，累计航迹转角为180°，顶部滚转角为180°。"
                "这些结果说明两种方法都完成了殷麦曼机动几何和姿态边界。",
            ),
            ("figure", "immelmann_geometry", "图6-7 殷麦曼机动几何与平滑性对比", 5.75),
            (
                "p",
                f"五次多项式殷麦曼机动曲率半径为"
                f"{float(immelmann_a['最小曲率半径_m']):.2f}～"
                f"{float(immelmann_a['最大曲率半径_m']):.2f} m，B样条结果为"
                f"{float(immelmann_b['最小曲率半径_m']):.2f}～"
                f"{float(immelmann_b['最大曲率半径_m']):.2f} m。"
                "从轨迹图看，两者均保持连续的拱形上升，但五次多项式轨迹在侧部和顶部附近"
                "存在更明显的曲率起伏；B样条优化后半径变化平缓，动作几何更接近单调、"
                "连续的半回路。",
            ),
            (
                "p",
                f"殷麦曼机动均方根jerk由{float(immelmann_a['均方根jerk_mps3']):.3f}降至"
                f"{float(immelmann_b['均方根jerk_mps3']):.3f} m/s³，下降"
                f"{immelmann_jerk_reduction:.2f}%；均方根曲率变化率下降"
                f"{immelmann_curvature_reduction:.2f}%。该结果与完整筋斗趋势一致，说明"
                "B样条方法的平滑性改善并非只发生在周期闭合轨迹上，对开放式殷麦曼机动同样有效。",
            ),
        ],
        [
            ("h3", "6.2.5 殷麦曼机动平滑性与终端质量"),
            (
                "p",
                "殷麦曼机动是开放轨迹，优化器不仅要改善中间区段，还必须同时满足终端位置、"
                "高度和反向航迹方向。因此，若节点自由度不足，优化可能出现终端正确但中部"
                "曲率突变，或中部平滑但终端误差超限。本测试使用9个节点和端点零一阶导数条件，"
                "在保持终端约束的同时，使进入和退出区段的曲率变化保持平缓。",
            ),
            (
                "p",
                f"五次多项式与B样条殷麦曼机动的路径长度差小于"
                f"{abs(float(immelmann_b['轨迹长度_m']) - float(immelmann_a['轨迹长度_m'])):.6f} m，"
                f"持续时间差小于"
                f"{abs(float(immelmann_b['持续时间_s']) - float(immelmann_a['持续时间_s'])):.6f} s。"
                "因此，两者在飞行距离和时间预算上等价。B样条方法的jerk下降不能解释为"
                "简单延长动作，而是来自曲率节点重新分配。",
            ),
            (
                "p",
                f"殷麦曼机动最大jerk下降"
                f"{-pct_change(float(immelmann_b['最大jerk_mps3']), float(immelmann_a['最大jerk_mps3'])):.2f}%，"
                f"均方根jerk下降{immelmann_jerk_reduction:.2f}%，均方根曲率变化率下降"
                f"{immelmann_curvature_reduction:.2f}%。同时，顶部半滚转时序保持不变，"
                "说明这些改善来源于纵向轨迹本身，而非降低滚转要求。",
            ),
            (
                "status",
                "终端质量",
                "B样条殷麦曼机动在位置、高度、航迹方向和滚转姿态边界均正确的前提下改善平滑性，"
                "可作为完整殷麦曼动作的轨迹—姿态参考。",
            ),
        ],
        [
            ("h3", "6.2.6 优化收敛、计算代价与重复性"),
            (
                "p",
                "数值优化方法除了比较轨迹质量，还必须报告是否稳定收敛以及需要多少计算量。"
                "本测试保存每次求解的初始目标值、最终目标值、迭代次数、目标函数计算次数、"
                "等式约束残差和不等式约束裕量。两种动作均由五次多项式可行轨迹初始化，"
                "优化结束标志为SLSQP满足收敛条件，而不是达到最大迭代次数后强制接受结果。",
            ),
            (
                "figure",
                "optimization_diagnostics",
                "图6-8 B样条约束优化的收敛与计算代价",
                5.7,
            ),
            (
                "p",
                f"筋斗求解时间约{float(loop_b['生成时间_s']):.3f} s，殷麦曼机动约"
                f"{float(immelmann_b['生成时间_s']):.3f} s；五次多项式方法对应时间分别约"
                f"{float(loop_a['生成时间_s']):.4f} s和"
                f"{float(immelmann_a['生成时间_s']):.4f} s。"
                "B样条方法明显更慢，但在当前离线轨迹生成任务中仍属于秒级，"
                "不会成为报告实验或训练数据预处理的主要瓶颈。",
            ),
            (
                "p",
                "单次运行时间会受计算机负载影响，因此它只用于比较数量级，不作为算法"
                "优劣的唯一依据。后续若要求在线实时更新，可预先离线生成动作库，"
                "在线阶段只做参数插值；也可减少节点数或使用五次多项式结果直接执行。"
                "若要求评估数值稳定性，应在不同初值、速度和半径条件下批量统计收敛率，"
                "而不是只引用本次成功结果。",
            ),
        ],
        [
            ("h3", "6.2.7 几何指标综合评估"),
            (
                "p",
                "四组轨迹均通过终端位置、动作平面、终端航迹转角、终端滚转角和最小曲率半径"
                "检查。五次多项式方法的优势是生成时间短，动作边界由五次多项式边界条件直接保证；"
                "B样条方法的优势是能够在相同边界下重新分配曲率，减少jerk和曲率变化率。"
                "两者的路径长度和动作时间基本一致，因此平滑性改善不是通过显著放大动作空间"
                "或延长飞行时间取得的。",
            ),
            ("figure", "key_metrics", "图6-9 双传统方法关键指标对比", 5.7),
            (
                "p",
                f"完整筋斗中，B样条方法的最大jerk由"
                f"{float(loop_a['最大jerk_mps3']):.3f}降至"
                f"{float(loop_b['最大jerk_mps3']):.3f} m/s³；殷麦曼机动中由"
                f"{float(immelmann_a['最大jerk_mps3']):.3f}降至"
                f"{float(immelmann_b['最大jerk_mps3']):.3f} m/s³。"
                "这种峰值与均方根同时下降的结果，说明改善覆盖了局部峰值和整段轨迹，"
                "而不只是把变化从一个时间点移动到另一个时间点。",
            ),
            (
                "status",
                "几何评估结论",
                "五次多项式方法和B样条方法均能可靠生成筋斗与殷麦曼。"
                "五次多项式方法适合快速、确定地构造动作；B样条方法以秒级优化时间换取更平滑的"
                "曲率和加速度变化，更适合作为后续控制跟踪与扩散模型数据集的高质量传统基线。",
            ),
        ],
        [
            ("h2", "6.3 实际飞机动力学的表现与评估"),
            ("h3", "6.3.1 飞机参数、气动数据与评价方法"),
            (
                "p",
                f"动力学结合性评价使用微小型固定翼无人机质量"
                f"{float(aircraft['mass_kg']):.1f} kg、机翼面积"
                f"{float(aircraft['wing_area_m2']):.2f} m²、翼展"
                f"{float(aircraft['wing_span_m']):.1f} m和平均气动弦长"
                f"{float(aircraft['mean_aerodynamic_chord_m']):.1f} m。"
                "转动惯量采用Ix=0.08、Iy=0.10、Iz=0.15 kg·m²。"
                "气动系数来自fixuav12222015.mat中的Tornado涡格法数据库，"
                "当前主要使用迎角—升力系数关系反查轨迹所需迎角。",
            ),
            (
                "p",
                "评价不把轨迹直接输入未经确认的全六自由度闭环模型，而是先做参考轨迹"
                "逆动力学。程序由速度和曲率计算切向与法向加速度，扣除重力后得到维持参考"
                "运动所需的合力，再换算为法向过载、气动力系数和升力裕度。"
                "殷麦曼机动顶部滚转时，还把所需法向力分解到机翼升力方向和机体侧向，"
                "用于检查滚转引入的侧向力需求。",
            ),
            ("formula", "aero_metrics", "过载、气动力系数与升力裕度指标"),
            (
                "table",
                ["类别", "评价指标", "当前阈值", "判定含义"],
                threshold_rows,
                [0.75, 2.05, 1.35, 2.35],
                8.0,
            ),
        ],
        [
            ("h3", "6.3.2 参考轨迹逆动力学与受力分解"),
            (
                "p",
                "对固定翼飞机而言，轨迹的几何可行并不等于气动力可实现。"
                "沿轨迹切向的加速度主要决定速度增减和净推进需求，沿轨迹法向的加速度"
                "决定速度方向改变和升力需求。重力在切向和法向上的投影随航迹转角变化，"
                "因此即使速度和曲率半径相同，筋斗底部、侧部和顶部的所需气动力也不同。",
            ),
            (
                "figure",
                "force_decomposition",
                "图6-10 参考轨迹逆动力学的受力分解",
                5.45,
            ),
            (
                "p",
                "程序首先根据轨迹计算a=Vdot·t+(V²/R)n，再用m(a-g)得到除重力外的"
                "所需合力。该合力在速度方向的分量记为净切向力需求，在垂直速度方向的"
                "分量用于计算法向过载和总法向力系数。殷麦曼机动滚转阶段还根据机翼法向"
                "把总法向力分成机翼升力分量和侧向分量，由此得到所需升力系数与侧向力系数。",
            ),
            (
                "p",
                "迎角需求通过VLM数据库中的CZ(α,β=0)关系反查。该步骤只在气动表覆盖的"
                "迎角范围内有效，不能外推到深失速或失速后状态。升力裕度以15°迎角处的"
                "正升力系数为当前极限代理，并额外除以1.2²形成速度裕度修正。"
                "这种处理适合早期规划筛查，但仍需由实测失速速度或可靠CLmax替换。",
            ),
            (
                "status",
                "评价等级",
                "本节判断的是“参考运动所需力和角速度是否落在当前假设包线内”，"
                "没有求出真实舵面偏角与油门，也没有证明闭环控制一定能跟踪。",
            ),
        ],
        [
            ("h3", "6.3.3 360度筋斗动力学表现"),
            (
                "p",
                f"五次多项式筋斗的最大法向过载为{float(loop_a['最大过载_g']):.3f} g，"
                f"B样条筋斗为{float(loop_b['最大过载_g']):.3f} g，均低于3.0 g筛查线。"
                "B样条方法将曲率分配得更均匀后，最大过载略有下降，但更重要的变化出现在"
                "气动力系数和迎角需求：最大估算迎角由"
                f"{float(loop_a['最大迎角需求_deg']):.3f}°降至"
                f"{float(loop_b['最大迎角需求_deg']):.3f}°。",
            ),
            ("figure", "loop_dynamics", "图6-11 360度筋斗动力学需求对比", 5.65),
            (
                "p",
                f"五次多项式方法计入1.2倍速度裕度后的最小升力裕度为"
                f"{float(loop_a['最小含速度裕度']):.3f}，仅略高于0.95判定线；"
                f"B样条方法提高到{float(loop_b['最小含速度裕度']):.3f}，"
                f"相对增加{loop_margin_gain:.2f}%。这意味着在当前VLM气动表和速度假设下，"
                "B样条轨迹对局部升力能力误差留出了更大余量。",
            ),
            (
                "p",
                f"航迹俯仰角速度峰值由{float(loop_a['最大俯仰角速度_degps']):.2f}°/s"
                f"降至{float(loop_b['最大俯仰角速度_degps']):.2f}°/s，"
                "均显著低于120°/s筛查值。筋斗不施加滚转指令，侧向力系数需求接近零。"
                "因此，就现有数据可验证的项目而言，两种方法都具备进入纵向闭环跟踪仿真的条件，"
                "B样条方法在升力裕度和俯仰变化平缓性上更有利。",
            ),
        ],
        [
            ("h3", "6.3.4 360度筋斗关键阶段分析"),
            (
                "p",
                "为了说明峰值从何而来，将完整筋斗按累计航迹转角分为底部进入、"
                "上升侧、顶部、下降侧和底部退出五个阶段。底部阶段曲率法向与重力反向，"
                "飞机需要额外升力同时克服重力并改变速度方向，通常形成较高正过载；"
                "顶部阶段曲率法向与重力同向，重力本身承担部分向心加速度，所需升力明显降低。",
            ),
            ("figure", "loop_phase", "图6-12 360度筋斗关键阶段动力学需求", 5.65),
            (
                "p",
                "五次多项式方法在上升侧和下降侧附近出现局部载荷与迎角峰值，"
                "与24 m曲率半径谷值位置相对应。B样条方法把半径限制在约30.34～34.88 m，"
                "使侧部峰值被削弱，载荷变化更接近单峰或缓变分布。"
                "这解释了为什么B样条筋斗的最大迎角和俯仰角速度同时下降。",
            ),
            (
                "p",
                "顶部附近估算迎角可能接近零或出现小幅负值，这不表示轨迹计算异常。"
                "当重力已经能够提供大部分向心加速度时，机翼只需产生较小升力，"
                "甚至可能需要轻微负升力保持规定曲率。实际飞行中，该区段对速度误差和姿态"
                "误差较敏感，应在闭环仿真中重点检查顶部速度保持和俯仰角超调。",
            ),
            (
                "status",
                "重点工况",
                "底部关注过载、迎角和推进需求；侧部关注曲率变化与俯仰角速度；"
                "顶部关注低升力状态下的速度保持和姿态误差。实飞试验不能只检查单一峰值。",
            ),
        ],
        [
            ("h3", "6.3.5 殷麦曼机动动力学表现"),
            (
                "p",
                f"五次多项式殷麦曼机动最大法向过载为{float(immelmann_a['最大过载_g']):.3f} g，"
                f"B样条殷麦曼机动为{float(immelmann_b['最大过载_g']):.3f} g。"
                "B样条结果的峰值略高，但差值很小，且两者均低于3.0 g。"
                "这一现象说明多目标优化并不保证每个单项指标都下降；B样条节点在同时降低jerk、"
                "曲率变化率和法向力系数峰值的过程中，对过载峰值作出了轻微折中。",
            ),
            ("figure", "immelmann_dynamics", "图6-13 殷麦曼机动动力学需求对比", 5.65),
            (
                "p",
                f"殷麦曼机动最大估算迎角由{float(immelmann_a['最大迎角需求_deg']):.3f}°降至"
                f"{float(immelmann_b['最大迎角需求_deg']):.3f}°，最小含速度裕度由"
                f"{float(immelmann_a['最小含速度裕度']):.3f}提高至"
                f"{float(immelmann_b['最小含速度裕度']):.3f}，相对增加{immelmann_margin_gain:.2f}%。"
                "虽然提升幅度小于完整筋斗，但B样条结果仍扩大了与0.95筛查线之间的距离。",
            ),
            (
                "p",
                f"顶部半滚转的最大机体系滚转角速度由"
                f"{float(immelmann_a['最大滚转角速度_degps']):.2f}°/s变为"
                f"{float(immelmann_b['最大滚转角速度_degps']):.2f}°/s，均低于240°/s。"
                f"最大侧向力系数需求从{float(immelmann_a['最大侧向力系数']):.3f}增加到"
                f"{float(immelmann_b['最大侧向力系数']):.3f}，但仍低于0.25。"
                "侧向力略增是纵向曲率分配改变后，滚转区间法向力水平发生变化所致，"
                "表明后续控制设计应联合检查滚转时序，而不能只优化纵向平面轨迹。",
            ),
        ],
        [
            ("h3", "6.3.6 殷麦曼机动顶部滚转耦合分析"),
            (
                "p",
                "殷麦曼机动在120°航迹转角后开始滚转，此时飞机已越过竖直上升阶段，"
                "法向过载总体下降。选择该区段滚转的目的，是避免在底部高载荷区同时要求"
                "大滚转角速度。滚转角通过五次函数从0°变化到180°，边界角速度和角加速度"
                "回到零，可减少进入反向水平飞行时的姿态突变。",
            ),
            (
                "figure",
                "roll_coupling",
                "图6-14 殷麦曼机动顶部半滚转与侧向力耦合",
                5.7,
            ),
            (
                "p",
                "滚转过程中，轨迹所需法向力方向仍由殷麦曼机动几何决定，但机翼升力方向随滚转"
                "不断旋转。若只保持原有总法向力，部分需求会投影到机体侧向，形成非零侧向力"
                "系数。五次多项式方法最大侧向力系数约0.087，B样条方法约0.096；后者略高，"
                "原因是其滚转区间内纵向法向力分布与五次多项式轨迹不同。",
            ),
            (
                "p",
                "当前程序把侧向力作为需求量筛查，并未进一步计算副翼、方向舵和升降舵的"
                "协调偏角。真实飞机完成殷麦曼动作时，滚转和俯仰通道会通过惯性耦合、"
                "气动交叉导数和舵面响应相互影响。后续控制器应采用姿态角速度反馈，"
                "并检查滚转过程中侧滑角、偏航角速度和高度误差，不能只跟踪滚转角。",
            ),
            (
                "status",
                "耦合结论",
                "两种殷麦曼机动轨姿的滚转率和侧向力需求均未超过当前筛查线，"
                "但B样条方法并未改善侧向力系数。滚转时序应在获得舵面数据后单独优化。",
            ),
        ],
        [
            ("h3", "6.3.7 与后续轨姿跟踪控制的接口"),
            (
                "p",
                "当前输出已经包含后续控制设计所需的基本参考量。位置和速度可进入外环"
                "轨迹跟踪器，累计航迹转角和滚转角可转换为姿态参考，机体系角速度可作为"
                "内环前馈量，加速度和法向过载可用于计算升力或俯仰控制前馈。"
                "因此，从数据接口角度看，两种方法都能够继续用于高机动轨姿跟踪控制研究。",
            ),
            (
                "figure",
                "control_interface",
                "图6-15 生成轨姿与后续闭环跟踪验证接口",
                5.7,
            ),
            (
                "p",
                "后续验证应至少建立位置/速度外环、姿态/角速度内环、执行机构模型和"
                "六自由度飞机模型。跟踪精度不能只用终端误差评价，还应统计全程位置均方根误差、"
                "最大位置误差、航迹角误差、滚转角误差、速度误差和约束违反持续时间。"
                "对筋斗应重点检查顶部和底部，对殷麦曼机动应重点检查滚转开始、滚转结束和反向退出。",
            ),
            (
                "p",
                "五次多项式轨迹可作为控制器初次联调的低复杂度参考，因为结果固定、问题容易复现；"
                "B样条轨迹可用于验证控制器对更平滑但数值生成轨迹的跟踪表现。"
                "若B样条轨迹的闭环误差反而更大，应检查姿态映射、控制前馈和滚转耦合，"
                "不能仅依据开环jerk较小就预判闭环一定更优。",
            ),
            (
                "status",
                "进入控制阶段前的缺口",
                "仍需补充推力曲线、舵面最大偏角、舵机速率、执行延迟、真实重心、结构过载"
                "和实测失速数据。缺少这些数据时，只能开展受限的模型级验证。",
            ),
        ],
        [
            ("h3", "6.3.8 综合结果、结论与适用边界"),
            (
                "table",
                ["动作", "方法", "几何结果", "主要动力学结果", "判定"],
                result_rows,
                [1.05, 0.72, 1.75, 2.5, 0.48],
                7.6,
            ),
            (
                "p",
                "综合结果表明，两种传统方法均能在当前统一边界下生成可用的筋斗与殷麦曼"
                "参考轨姿，并通过规划级几何和动力学筛查。五次多项式方法计算速度快、"
                "结果确定、参数与动作形状之间的关系直观，适合作为在线快速生成或故障回退基准。"
                "B样条约束优化方法计算时间较长，但能够在生成阶段直接处理路径长度、终端位置、"
                "曲率、过载和升力裕度，并显著降低jerk和曲率变化率。",
            ),
            (
                "p",
                "对完整筋斗，B样条方法同时降低最大过载、最大迎角和俯仰角速度，"
                "并把最小含速度裕度提高到1.010左右，综合改善较为明显。"
                "对殷麦曼机动，B样条方法降低jerk、曲率变化率、迎角和滚转角速度，"
                "但最大过载和侧向力系数略有增加，反映了多目标约束优化的真实折中。"
                "因此，报告不把B样条表述为对五次多项式方法的全面替代，而将其定位为更适合"
                "离线高质量轨迹生成的传统优化方法。",
            ),
            (
                "status",
                "本阶段结论",
                "两种方法生成的四条轨姿均可作为后续跟踪控制算法的参考输入。"
                "其中B样条结果更适合用于高质量训练样本和控制平滑性研究，五次多项式结果适合"
                "作为稳定基准。进入实物试验前仍必须完成推力、舵面、舵机、失速、结构载荷"
                "及六自由度闭环跟踪验证。",
            ),
            (
                "p",
                "本次动力学结果不能证明实机一定能够完成动作。VLM数据主要适用于附着流，"
                "15°迎角上限不是实测失速迎角；净切向力与机械功率尚未叠加可靠阻力模型，"
                "也没有与电机—螺旋桨实测能力比较；滚转率和侧向力阈值仍属于工程假设。"
                "后续获得真实推进与操纵参数后，应保持本次轨迹数据不变，依次完成控制量反算、"
                "六自由度开环验证、闭环跟踪误差统计和安全包线复核。",
            ),
            (
                "small",
                "参考文献：[1] Piegl L, Tiller W. The NURBS Book. Springer, 1997。\n"
                "[2] Betts J T. Practical Methods for Optimal Control and Estimation Using Nonlinear Programming. SIAM, 2010。\n"
                "[3] Stevens B L, Lewis F L, Johnson E N. Aircraft Control and Simulation. Wiley, 2015。\n"
                "[4] Beard R W, McLain T W. Small Unmanned Aircraft: Theory and Practice. Princeton University Press, 2012。\n"
                "[5] 项目内部资料：fixuav12222015.mat气动系数数据库及微小型固定翼飞机几何、质量与惯量参数。",
            ),
        ],
    ]
    if len(pages) < 20:
        raise RuntimeError(f"详细版报告页面单元不应少于20，当前为{len(pages)}")
    return pages


def formula_paths() -> dict[str, Path]:
    names = {
        "quintic_curve": "式6-1_五次多项式轨迹方程.png",
        "quintic_boundary": "式6-2_五次多项式边界约束.png",
        "bspline": "式6-3_B样条曲率参数化.png",
        "objective": "式6-4_优化目标函数.png",
        "constraints": "式6-5_几何与动力学约束.png",
        "time_law": "式6-6_速度与时间参数化.png",
        "inverse_dynamics": "式6-7_参考轨迹逆动力学.png",
        "aero_metrics": "式6-8_过载与气动系数.png",
    }
    paths = {key: FORMULA_DIR / name for key, name in names.items()}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少公式图片 {key}: {path}")
    return paths


def set_run_font(run, east_asia: str, ascii_font: str, size: float, bold=False):
    HELPER.set_run_font(run, east_asia, ascii_font, size, bold=bold)


def add_heading(document, text: str, level: int) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.space_before = HELPER.Pt(0 if level == 1 else 3)
    paragraph.paragraph_format.space_after = HELPER.Pt(6 if level <= 2 else 4)
    run = paragraph.add_run(text)
    set_run_font(
        run,
        "黑体",
        "Arial",
        {1: 16, 2: 14, 3: 12}[level],
        bold=True,
    )


def add_formula(document, path: Path, description: str) -> None:
    paragraph = document.add_paragraph(style="Report Equation")
    paragraph.alignment = HELPER.WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = HELPER.Pt(2)
    paragraph.paragraph_format.space_after = HELPER.Pt(4)
    run = paragraph.add_run()
    run.add_picture(str(path), width=HELPER.Inches(5.75))
    for properties in run._r.xpath(".//wp:docPr"):
        properties.set("title", description)
        properties.set("descr", description)


def add_header(section) -> None:
    section.different_first_page_header_footer = False
    section.header.is_linked_to_previous = False
    HELPER.clear_header_footer(section.header)
    paragraph = section.header.paragraphs[0]
    paragraph.alignment = HELPER.WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("任务二测试报告｜筋斗与殷麦曼传统轨姿生成")
    set_run_font(run, "宋体", "Arial", 8.5)
    run.font.color.rgb = HELPER.RGBColor.from_string("64748B")


def render_block(document, block, figures, formulas) -> None:
    kind = block[0]
    if kind == "h1":
        add_heading(document, block[1], 1)
    elif kind == "h2":
        add_heading(document, block[1], 2)
    elif kind == "h3":
        add_heading(document, block[1], 3)
    elif kind == "formula":
        add_formula(document, formulas[block[1]], block[2])
    else:
        HELPER.render_block(document, block, figures)


def build_report(pages, figures, formulas) -> dict[str, int]:
    actual_hash = hashlib.sha256(REFERENCE_DOCX.read_bytes()).hexdigest().upper()
    if actual_hash != REFERENCE_SHA256:
        raise RuntimeError("任务一参考报告哈希发生变化，停止生成。")
    for key, path in figures.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少报告插图 {key}: {path}")

    document = HELPER.Document(REFERENCE_DOCX)
    HELPER.clear_document_body(document)
    HELPER.normalize_styles(document)
    for section in document.sections:
        HELPER.configure_section(section)
        add_header(section)
        HELPER.clear_header_footer(section.footer)
        HELPER.add_footer(section)

    document.core_properties.title = "任务二测试报告：筋斗与殷麦曼双传统方法"
    document.core_properties.subject = "五次多项式与B样条约束优化对比测试"
    document.core_properties.comments = "规划级测试报告，待项目组审阅"

    for page_index, blocks in enumerate(pages):
        if page_index:
            HELPER.add_page_break(document)
        for block in blocks:
            render_block(document, block, figures, formulas)

    HELPER.set_update_fields(document)
    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT_DOCX)
    HELPER.remove_unreferenced_template_headers(OUTPUT_DOCX)
    HELPER.update_page_metadata(OUTPUT_DOCX, len(pages))
    return HELPER.structural_audit(OUTPUT_DOCX)


def prepare_assets() -> None:
    global plt
    import matplotlib.pyplot as matplotlib_pyplot

    plt = matplotlib_pyplot
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    frames = read_frames()
    make_figures(frames)
    make_formula_images()
    print(f"报告素材目录: {ASSET_DIR}")


def assemble_docx() -> None:
    global HELPER
    # The retained formatting helper defines optional plotting functions but
    # this assembly phase only uses its DOCX utilities.  Provide import stubs
    # so the document runtime does not need a second Matplotlib installation.
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        import types

        matplotlib_stub = types.ModuleType("matplotlib")
        pyplot_stub = types.ModuleType("matplotlib.pyplot")
        patches_stub = types.ModuleType("matplotlib.patches")
        patches_stub.FancyArrowPatch = type("FancyArrowPatch", (), {})
        patches_stub.FancyBboxPatch = type("FancyBboxPatch", (), {})
        matplotlib_stub.pyplot = pyplot_stub
        matplotlib_stub.patches = patches_stub
        sys.modules["matplotlib"] = matplotlib_stub
        sys.modules["matplotlib.pyplot"] = pyplot_stub
        sys.modules["matplotlib.patches"] = patches_stub
    HELPER = load_module("dual_method_docx_helper", HELPER_PATH)
    config = load_json(CONFIG_PATH)
    summary = pd.read_csv(SUMMARY_PATH, encoding="utf-8-sig")
    run_summary = load_json(RUN_SUMMARY_PATH)
    figures = figure_paths()
    formulas = formula_paths()
    pages = build_pages(config, summary, run_summary)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_report(pages, figures, formulas)
    qa_dir = REPORT_DIR / "_质量检查"
    qa_dir.mkdir(parents=True, exist_ok=True)
    with (qa_dir / "structural_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Created: {OUTPUT_DOCX}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare-assets",
        action="store_true",
        help="使用科学计算环境生成公式和报告插图",
    )
    mode.add_argument(
        "--assemble-docx",
        action="store_true",
        help="使用文档环境组装DOCX",
    )
    args = parser.parse_args()
    if args.prepare_assets:
        prepare_assets()
    else:
        assemble_docx()


if __name__ == "__main__":
    main()
