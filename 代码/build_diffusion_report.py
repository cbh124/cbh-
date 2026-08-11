"""Build the diffusion-model maneuver test report.

The report uses the same visual system and three-part technical structure as
the completed traditional-method report.  It explicitly separates learned
shape generation, hard geometric decoding, and aircraft-physics screening.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
OUTPUTS_DIR = PROJECT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "结果"
COMPARISON_DIR = RESULTS_DIR / "对比评估"
REPORT_DIR = PROJECT_DIR / "测试报告"
ASSET_DIR = REPORT_DIR / "插图"
FORMULA_DIR = ASSET_DIR / "公式"
QA_DIR = REPORT_DIR / "_质量检查_V1.1"
OUTPUT_DOCX = REPORT_DIR / "任务二测试报告_扩散模型部分_筋斗与殷麦曼_V1.1.docx"

CONFIG_PATH = PROJECT_DIR / "配置" / "diffusion_config.json"
ALL_METRICS_PATH = COMPARISON_DIR / "all_methods_metrics.csv"
UNIQUE_METRICS_PATH = COMPARISON_DIR / "diffusion_unique_metrics.csv"
ROBUSTNESS_PATH = COMPARISON_DIR / "seed_robustness_summary.csv"
DATA_SUMMARY_PATH = RESULTS_DIR / "训练数据" / "dataset_summary.json"
TRAINING_SUMMARY_PATH = RESULTS_DIR / "训练记录" / "training_summary.json"

HELPER_PATH = OUTPUTS_DIR / "任务二测试报告" / "build_task2_report.py"
REFERENCE_DOCX = Path(r"C:\Users\admin\Desktop\固定翼项目评审\任务一测试报告.docx")
REFERENCE_SHA256 = "9DBAC008E633AA9BCE54651AF9A92E0551D0AB45EB7E579029F6C64FF7B9FC30"

QUINTIC = "五次多项式方法"
BSPLINE = "B样条约束优化方法"
DIFFUSION = "条件潜空间扩散模型"
LOOP = "360度筋斗"
IMMELMANN = "殷麦曼机动（半筋斗接半滚转）"

plt = None
patches = None
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
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def row_for(rows, method=None, maneuver=None):
    matched = []
    for row in rows:
        if method is not None and row.get("方法") != method:
            continue
        if maneuver is not None and row.get("机动动作") != maneuver:
            continue
        matched.append(row)
    if len(matched) != 1:
        raise KeyError(f"结果不唯一: method={method}, maneuver={maneuver}")
    return matched[0]


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def pct(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def make_formula(name: str, latex: str, number: str, font_size: float = 18) -> Path:
    if plt is None:
        raise RuntimeError("公式图片只能在素材生成阶段创建。")
    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    path = FORMULA_DIR / f"{name}.png"
    latex = latex.replace("\\\\", "\\")
    fig = plt.figure(figsize=(10.8, 0.78))
    fig.patch.set_facecolor("white")
    fig.text(0.48, 0.52, f"${latex}$", ha="center", va="center", fontsize=font_size)
    fig.text(0.965, 0.52, f"({number})", ha="right", va="center", fontsize=12)
    fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def make_formula_images() -> dict[str, Path]:
    return {
        "hard_decode": make_formula(
            "式7-2_硬约束潜空间解码",
            r"\mathbf{P}=\mathbf{P}_0+\mathbf{N}\mathbf{z},\qquad"
            r"\mathbf{A}\mathbf{P}_0=\mathbf{b},\quad\mathbf{A}\mathbf{N}=\mathbf{0}",
            "7-2",
        ),
        "bspline_radius": make_formula(
            "式7-1_B样条半径与轨迹积分",
            r"\mathbf{P}=[R_1,\ldots,R_{12}]^{\mathrm{T}},\quad"
            r"R(\theta)=\mathbf{B}(\theta)\mathbf{P},\quad"
            r"x(\theta)=x_0+\int_{\theta_0}^{\theta}R(\xi)\cos\xi\,d\xi,\quad"
            r"z(\theta)=z_0+\int_{\theta_0}^{\theta}R(\xi)\sin\xi\,d\xi",
            "7-1",
            12,
        ),
        "forward": make_formula(
            "式7-3_前向加噪",
            r"\mathbf{z}_t=\sqrt{\bar\alpha_t}\,\mathbf{z}_0+"
            r"\sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon},\qquad"
            r"\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})",
            "7-3",
        ),
        "loss": make_formula(
            "式7-4_噪声预测损失",
            r"\mathcal{L}_{\mathrm{diff}}="
            r"\mathrm{E}_{\mathbf{z}_0,t,\boldsymbol{\epsilon}}"
            r"\left[\left\|\boldsymbol{\epsilon}-"
            r"\boldsymbol{\epsilon}_\theta(\mathbf{z}_t,t,\mathbf{c})"
            r"\right\|_2^2\right]",
            "7-4",
            16,
        ),
        "cfg": make_formula(
            "式7-5_无分类器引导",
            r"\hat{\boldsymbol{\epsilon}}="
            r"\boldsymbol{\epsilon}_\theta(\mathbf{z}_t,t,\emptyset)+w"
            r"\left[\boldsymbol{\epsilon}_\theta(\mathbf{z}_t,t,\mathbf{c})-"
            r"\boldsymbol{\epsilon}_\theta(\mathbf{z}_t,t,\emptyset)\right]",
            "7-5",
            15,
        ),
        "ddim": make_formula(
            "式7-6_DDIM确定性更新",
            r"\mathbf{z}_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat{\mathbf{z}}_0+"
            r"\sqrt{1-\bar\alpha_{t-1}}\,\hat{\boldsymbol{\epsilon}},\quad"
            r"\hat{\mathbf{z}}_0="
            r"\frac{\mathbf{z}_t-\sqrt{1-\bar\alpha_t}\hat{\boldsymbol{\epsilon}}}"
            r"{\sqrt{\bar\alpha_t}}",
            "7-6",
            14,
        ),
        "selection": make_formula(
            "式7-7_Best_of_N物理筛选目标",
            r"J=0.30J_{\mathrm{jerk,n}}+0.25J_{\dot\kappa,\mathrm{n}}+"
            r"0.15J_{\alpha,\mathrm{n}}+0.12J_{n,\mathrm{n}}+"
            r"0.13J_{\mathrm{margin,n}}+0.05J_{\mathrm{rough,n}}+"
            r"0.30P_{\mathrm{margin}}",
            "7-7",
            13,
        ),
        "inverse": make_formula(
            "式7-8_参考轨迹逆动力学",
            r"\mathbf{a}=\dot V\,\mathbf{e}_t+V^2\kappa\,\mathbf{e}_n,\quad"
            r"\mathbf{F}_{\mathrm{req}}=m(\mathbf{a}-\mathbf{g}),\quad"
            r"n=\frac{\|\mathbf{F}_{\mathrm{req}}\|}{mg}",
            "7-8",
            15,
        ),
        "aero": make_formula(
            "式7-9_气动需求与升力裕度",
            r"C_{N,\mathrm{req}}=\frac{F_N}{\frac{1}{2}\rho V^2S},\quad"
            r"M_L=\frac{C_{N,\max}}{C_{N,\mathrm{req}}(1.2)^2}",
            "7-9",
            17,
        ),
    }


def draw_box(ax, xy, wh, text, face, edge, fontsize=10):
    x, y = xy
    w, h = wh
    box = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.2, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def save_diagram(name: str, title: str, boxes, arrows, notes=None) -> Path:
    path = ASSET_DIR / name
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for item in boxes:
        draw_box(ax, item[0], item[1], item[2], item[3], item[4], item[5] if len(item) > 5 else 10)
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5))
    if notes:
        for x, y, text, color in notes:
            ax.text(x, y, text, ha="center", va="center", fontsize=9, color=color)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_diagrams() -> dict[str, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    framework = save_diagram(
        "图7-1_扩散模型统一测试框架.png",
        "条件潜空间扩散模型的训练、生成与统一评价框架",
        [
            ((0.03, 0.58), (0.18, 0.20), "动作条件\n筋斗 / 殷麦曼", "#E0F2FE", "#0284C7"),
            ((0.27, 0.58), (0.18, 0.20), "潜空间扩散模型\n学习形状分布", "#EDE9FE", "#7C3AED"),
            ((0.51, 0.58), (0.18, 0.20), "硬约束解码器\n边界严格满足", "#DCFCE7", "#16A34A"),
            ((0.75, 0.58), (0.20, 0.20), "飞机参数与VLM\n动力学筛选", "#FEF3C7", "#D97706"),
            ((0.27, 0.17), (0.18, 0.18), "传统方法基线\n五次 / B样条", "#F1F5F9", "#64748B"),
            ((0.57, 0.17), (0.18, 0.18), "几何与动力学\n统一指标对比", "#FEE2E2", "#DC2626"),
        ],
        [
            ((0.21, 0.68), (0.27, 0.68)), ((0.45, 0.68), (0.51, 0.68)),
            ((0.69, 0.68), (0.75, 0.68)), ((0.36, 0.58), (0.36, 0.35)),
            ((0.45, 0.26), (0.57, 0.26)), ((0.85, 0.58), (0.70, 0.35)),
        ],
    )
    dataset = save_diagram(
        "图7-2_专家样本构建流程.png",
        "训练样本不是任意曲线：先满足动作边界，再进行规划级物理筛查",
        [
            ((0.03, 0.58), (0.17, 0.20), "平滑随机潜变量\n9维", "#EDE9FE", "#7C3AED"),
            ((0.24, 0.58), (0.17, 0.20), "零空间解码\n12个半径节点", "#DCFCE7", "#16A34A"),
            ((0.45, 0.58), (0.17, 0.20), "轨迹与时间化\n速度规律一致", "#E0F2FE", "#0284C7"),
            ((0.66, 0.58), (0.17, 0.20), "几何/动力学筛查\n不合格则拒绝", "#FEF3C7", "#D97706"),
            ((0.42, 0.16), (0.22, 0.18), "6000条专家样本\n两动作各3000条", "#FEE2E2", "#DC2626"),
        ],
        [
            ((0.20, 0.68), (0.24, 0.68)), ((0.41, 0.68), (0.45, 0.68)),
            ((0.62, 0.68), (0.66, 0.68)), ((0.745, 0.58), (0.56, 0.34)),
        ],
        [(0.865, 0.45, "拒绝样本返回重采样", "#B91C1C")],
    )
    decoder = save_diagram(
        "图7-3_潜变量与硬约束解码.png",
        "网络负责多样性，解析解码器负责确定性边界",
        [
            ((0.05, 0.56), (0.18, 0.22), "扩散输出 z\n9维随机形状坐标", "#EDE9FE", "#7C3AED"),
            ((0.31, 0.56), (0.20, 0.22), "P = P0 + Nz\n映射到12个节点", "#DCFCE7", "#16A34A"),
            ((0.59, 0.56), (0.17, 0.22), "三次B样条\n半径函数 R(θ)", "#E0F2FE", "#0284C7"),
            ((0.80, 0.56), (0.16, 0.22), "积分得到 x,z\n并时间参数化", "#FEF3C7", "#D97706"),
            ((0.29, 0.15), (0.22, 0.18), "AP=b：长度、终点x、\n终点z严格满足", "#FEE2E2", "#DC2626"),
            ((0.62, 0.15), (0.22, 0.18), "边界后仍检查半径、\n载荷、迎角和裕度", "#F1F5F9", "#64748B"),
        ],
        [
            ((0.23, 0.67), (0.31, 0.67)), ((0.51, 0.67), (0.59, 0.67)),
            ((0.76, 0.67), (0.80, 0.67)), ((0.40, 0.56), (0.40, 0.33)),
            ((0.51, 0.24), (0.62, 0.24)),
        ],
    )
    selection = save_diagram(
        "图7-4_Best_of_N筛选流程.png",
        "一次条件输入生成一组候选，再用固定翼物理指标选择输出",
        [
            ((0.03, 0.58), (0.17, 0.20), "高斯噪声\n256个初始样本", "#EDE9FE", "#7C3AED"),
            ((0.24, 0.58), (0.17, 0.20), "60步DDIM\n批量反向去噪", "#E0F2FE", "#0284C7"),
            ((0.45, 0.58), (0.17, 0.20), "硬约束解码\n形成轨迹族", "#DCFCE7", "#16A34A"),
            ((0.66, 0.58), (0.17, 0.20), "剔除约束违反\n保留可行候选", "#FEF3C7", "#D97706"),
            ((0.41, 0.16), (0.22, 0.18), "综合质量排序\n输出Best-of-N轨迹", "#FEE2E2", "#DC2626"),
        ],
        [
            ((0.20, 0.68), (0.24, 0.68)), ((0.41, 0.68), (0.45, 0.68)),
            ((0.62, 0.68), (0.66, 0.68)), ((0.745, 0.58), (0.52, 0.34)),
        ],
    )
    dynamics = save_diagram(
        "图7-5_规划级动力学评估边界.png",
        "从参考轨姿到实机试验之间的验证层级",
        [
            ((0.03, 0.58), (0.17, 0.20), "本报告已完成\n参考轨姿生成", "#DCFCE7", "#16A34A"),
            ((0.24, 0.58), (0.17, 0.20), "本报告已完成\n逆动力学+VLM筛查", "#DCFCE7", "#16A34A"),
            ((0.45, 0.58), (0.17, 0.20), "后续工作\n六自由度闭环仿真", "#FEF3C7", "#D97706"),
            ((0.66, 0.58), (0.17, 0.20), "后续工作\n半实物与执行机构", "#FEF3C7", "#D97706"),
            ((0.83, 0.18), (0.14, 0.20), "最终目标\n实物飞行演示", "#FEE2E2", "#DC2626"),
        ],
        [
            ((0.20, 0.68), (0.24, 0.68)), ((0.41, 0.68), (0.45, 0.68)),
            ((0.62, 0.68), (0.66, 0.68)), ((0.745, 0.58), (0.87, 0.38)),
        ],
        [(0.33, 0.39, "规划级结论", "#166534"), (0.61, 0.39, "控制级验证", "#92400E")],
    )
    control = save_diagram(
        "图7-6_扩散轨姿控制接口.png",
        "扩散模型输出进入后续轨姿跟踪控制的接口",
        [
            ((0.02, 0.58), (0.17, 0.20), "离线扩散采样\n候选轨迹族", "#EDE9FE", "#7C3AED"),
            ((0.22, 0.58), (0.17, 0.20), "物理筛选\n确定参考轨姿", "#DCFCE7", "#16A34A"),
            ((0.42, 0.58), (0.17, 0.20), "位置/速度外环\n轨迹误差反馈", "#E0F2FE", "#0284C7"),
            ((0.62, 0.58), (0.17, 0.20), "姿态/角速度内环\n前馈+反馈", "#FEF3C7", "#D97706"),
            ((0.82, 0.58), (0.16, 0.20), "六自由度飞机\n执行机构模型", "#FEE2E2", "#DC2626"),
            ((0.42, 0.16), (0.17, 0.18), "跟踪精度统计\nRMS/峰值/约束", "#F1F5F9", "#64748B"),
        ],
        [
            ((0.19, 0.68), (0.22, 0.68)), ((0.39, 0.68), (0.42, 0.68)),
            ((0.59, 0.68), (0.62, 0.68)), ((0.79, 0.68), (0.82, 0.68)),
            ((0.90, 0.58), (0.57, 0.34)), ((0.42, 0.25), (0.31, 0.58)),
        ],
    )
    return {
        "framework": framework,
        "dataset": dataset,
        "decoder": decoder,
        "selection": selection,
        "dynamics_scope": dynamics,
        "control": control,
    }


def figure_paths() -> dict[str, Path]:
    paths = {
        "framework": ASSET_DIR / "图7-1_扩散模型统一测试框架.png",
        "dataset": ASSET_DIR / "图7-2_专家样本构建流程.png",
        "decoder": ASSET_DIR / "图7-3_潜变量与硬约束解码.png",
        "selection": ASSET_DIR / "图7-4_Best_of_N筛选流程.png",
        "dynamics_scope": ASSET_DIR / "图7-5_规划级动力学评估边界.png",
        "control": ASSET_DIR / "图7-6_扩散轨姿控制接口.png",
        "training": COMPARISON_DIR / "training_history.png",
        "denoising": COMPARISON_DIR / "denoising_process.png",
        "loop_geometry": COMPARISON_DIR / "loop_360_geometry_comparison.png",
        "loop_diversity": COMPARISON_DIR / "loop_360_candidate_diversity.png",
        "loop_dynamics": COMPARISON_DIR / "loop_360_dynamics_comparison.png",
        "immelmann_geometry": COMPARISON_DIR / "immelmann_geometry_comparison.png",
        "immelmann_diversity": COMPARISON_DIR / "immelmann_candidate_diversity.png",
        "immelmann_dynamics": COMPARISON_DIR / "immelmann_dynamics_comparison.png",
        "key_metrics": COMPARISON_DIR / "key_metrics_comparison.png",
        "robustness": COMPARISON_DIR / "seed_robustness.png",
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少报告插图 {key}: {path}")
    return paths


def formula_paths() -> dict[str, Path]:
    names = {
        "hard_decode": "式7-2_硬约束潜空间解码.png",
        "bspline_radius": "式7-1_B样条半径与轨迹积分.png",
        "forward": "式7-3_前向加噪.png",
        "loss": "式7-4_噪声预测损失.png",
        "cfg": "式7-5_无分类器引导.png",
        "ddim": "式7-6_DDIM确定性更新.png",
        "selection": "式7-7_Best_of_N物理筛选目标.png",
        "inverse": "式7-8_参考轨迹逆动力学.png",
        "aero": "式7-9_气动需求与升力裕度.png",
    }
    paths = {key: FORMULA_DIR / value for key, value in names.items()}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少公式图片 {key}: {path}")
    return paths


def build_pages(config, metrics, unique, robustness, data_summary, training_summary):
    ql = row_for(metrics, QUINTIC, LOOP)
    bl = row_for(metrics, BSPLINE, LOOP)
    dl = row_for(metrics, DIFFUSION, LOOP)
    qi = row_for(metrics, QUINTIC, IMMELMANN)
    bi = row_for(metrics, BSPLINE, IMMELMANN)
    di = row_for(metrics, DIFFUSION, IMMELMANN)
    ul = row_for(unique, maneuver=LOOP)
    ui = row_for(unique, maneuver=IMMELMANN)
    rl = next(row for row in robustness if row["maneuver_key"] == "loop_360")
    ri = next(row for row in robustness if row["maneuver_key"] == "immelmann")

    geometry_rows = []
    for action, rows in (("360度筋斗", (ql, bl, dl)), ("殷麦曼", (qi, bi, di))):
        for row in rows:
            geometry_rows.append([
                action,
                row["方法"].replace("方法", "").replace("条件潜空间", "潜空间"),
                f"{f(row, '轨迹长度_m'):.3f}",
                f"{f(row, '终端位置误差_m'):.3f}",
                f"{f(row, '最小曲率半径_m'):.2f}",
                f"{f(row, '均方根jerk_mps3'):.3f}",
                f"{f(row, '均方根曲率变化率_1pm_s'):.5f}",
            ])
    unique_rows = [
        [
            "360度筋斗", ul["候选数量"], f"{100*f(ul, '硬边界满足率'):.1f}%",
            f"{100*f(ul, '规划级可行率'):.1f}%", f"{f(ul, '曲率候选多样性_RMS_m'):.2f}",
            f"{100*f(ul, 'Best_of_N筛选收益'):.1f}%",
        ],
        [
            "殷麦曼", ui["候选数量"], f"{100*f(ui, '硬边界满足率'):.1f}%",
            f"{100*f(ui, '规划级可行率'):.1f}%", f"{f(ui, '曲率候选多样性_RMS_m'):.2f}",
            f"{100*f(ui, 'Best_of_N筛选收益'):.1f}%",
        ],
    ]
    dynamics_rows = []
    for action, rows in (("360度筋斗", (ql, bl, dl)), ("殷麦曼", (qi, bi, di))):
        for row in rows:
            dynamics_rows.append([
                action,
                row["方法"].replace("方法", "").replace("条件潜空间", "潜空间"),
                f"{f(row, '最大过载_g'):.3f}",
                f"{f(row, '最大迎角需求_deg'):.2f}",
                f"{f(row, '最小含速度裕度'):.3f}",
                f"{f(row, '最大俯仰角速度_degps'):.1f}",
                row["规划级结论"],
            ])

    pages = [
        [
            ("h1", "任务二测试报告：扩散模型轨姿生成方法"),
            ("h2", "360度筋斗与殷麦曼机动条件生成测试"),
            ("p", "本部分面向质量3.0 kg、机翼面积0.24 m²、翼展1.2 m、平均气动弦长0.2 m的微小型固定翼无人机，构建条件潜空间扩散模型，分别生成360度筋斗和殷麦曼机动的轨迹—姿态参考。测试沿用传统方法部分的动作尺度、速度规律、飞机惯量和VLM气动系数表，使扩散模型、五次多项式和B样条结果处于同一比较条件。"),
            ("status", "报告结构", "全文按7.1原理概述、7.2生成轨迹的几何对比与指标评估、7.3基于真实飞机参数的动力学测试对比展开。除统一指标外，增加候选多样性、可行率、Best-of-N收益、随机种子鲁棒性和去噪过程等扩散模型专属评价。"),
            ("figure", "framework", "图7-1 条件潜空间扩散模型统一测试框架", 5.75),
            ("p", "本报告中的“动力学测试”是利用真实飞机质量、几何、惯量和项目气动表开展的规划级逆动力学筛查，不是六自由度闭环跟踪试验，也不是实物飞行演示。该范围界定用于避免把参考轨迹可行性与控制系统可实现性混为一谈。"),
        ],
        [
            ("h2", "7.1 原理概述"),
            ("h3", "7.1.1 任务定义与总体技术思路"),
            ("p", "本项目要求针对同一架微小型固定翼无人机生成360度筋斗和殷麦曼两种高机动动作的轨迹—姿态参考。360度筋斗的纵向轨迹需要完成一周连续转向并回到规定终点；殷麦曼机动需要先完成180度半筋斗，到达约64 m的高度增量并以反向水平航向退出，同时在顶部阶段完成180度半滚转。最终结果不仅要给出空间位置，还要形成随时间变化的速度、航迹方向、滚转角和角速度等参考量，供后续动力学评估与控制器设计使用。"),
            ("p", "扩散模型没有直接替代轨迹几何和固定翼约束，而是承担“生成多种可行曲率分配”的任务。整体方法先用曲率半径描述筋斗或半筋斗的基本形状，再把轨迹长度和终点位置写成解析约束；扩散网络只学习约束允许范围内仍可变化的形状自由度。网络生成多个候选后，解析解码器将其还原为轨迹，最后利用曲率、平滑性、载荷、迎角和升力裕度等指标进行筛选。"),
            ("p", "采用这种技术路线的原因是：五次多项式和B样条优化通常针对给定条件输出一条确定轨迹，而扩散模型能够学习一组可行轨迹所形成的条件概率分布。相同动作条件下改变初始随机噪声，可以得到局部曲率分配不同的候选解，再根据飞机状态或评价侧重点选择输出。因此，本项目评价扩散模型时既比较最终单条轨迹，也考察候选可行率、多样性、Best-of-N筛选收益和随机种子稳定性。"),
            ("p", "当前模型属于“条件扩散生成器+解析几何解码器+物理筛选器”的混合框架，不是从任务指令直接输出舵面控制量的端到端飞控器。该定位使模型结构和每一步数据含义都能够检查，也明确了现阶段结果仍属于轨迹规划级输出。"),
        ],
        [
            ("h3", "7.1.2 曲率半径节点与轨迹几何表示"),
            ("p", "筋斗和殷麦曼的纵向主体均位于x-z平面内。以累计航迹转角θ作为自变量，在动作转角范围内预先设置12个角度节点，并在每个节点配置一个曲率半径值R_i。节点可写为(θ_i，R_i)，12个半径值组成节点向量P。R_i的单位为米；半径越小表示局部转弯越急、曲率越大，半径越大表示轨迹越平缓。完整筋斗采用12个周期角度位置并在360度处闭合；殷麦曼在0～180度范围内包含首尾共12个节点，因此形成11个角度区间。节点数12是兼顾形状表达能力、训练维数和数值稳定性的工程设置，并非三次B样条的数学要求。"),
            ("p", "三次B样条中的“三次”表示每个局部多项式片段的次数为3，与节点数12以及后续三个约束没有对应关系。程序把(θ_i，R_i)作为插值节点，并在给定周期或端点边界条件后求得三次样条的内部系数；因此R_i是半径插值值，不应与算法内部控制系数混为一谈。式中B(θ)表示完成边界条件处理后的等效样条插值权重。每个角度区间主要受附近若干节点共同影响，而不是一个节点单独决定一段圆弧。只有当R(θ)在某一区间保持常数时，该区间才是严格圆弧；一般情况下得到的是连续变曲率轨迹。"),
            ("formula", "bspline_radius", "三次B样条曲率半径与纵向平面轨迹积分"),
            ("p", "得到R(θ)后，利用ds=R(θ)dθ计算弧长增量，再由dx=cosθ·ds、dz=sinθ·ds逐步积分得到x-z轨迹。完整筋斗的积分区间为0～360度，殷麦曼半筋斗为0～180度。由此形成清晰的几何解码链：半径节点P→连续半径函数R(θ)→弧长增量ds→位置坐标x(θ)、z(θ)。"),
        ],
        [
            ("h3", "7.1.3 三个线性约束与九维潜变量"),
            ("p", "由于B样条半径函数可以写成R(θ)=B(θ)P，其中B(θ)为已知的样条基函数行向量，因此轨迹总长度L=∫R(θ)dθ、终点前向位移Δx=∫R(θ)cosθdθ和终点高度变化Δz=∫R(θ)sinθdθ都对节点向量P保持线性。离散积分后，这三个关系可统一写为AP=b。矩阵A的三行分别对应总长度、终点前向位置和终点高度，目标向量b则由动作定义给出。基准条件下，筋斗约束为长度约201.06 m且终点相对位移为(0，0)；殷麦曼约束为长度约100.53 m、前向位移0 m和高度增量64 m。"),
            ("p", "程序首先求出一个满足AP_0=b的特解P_0，再计算约束矩阵A的零空间基N。因为A包含三个独立约束、P包含12个节点，所以零空间维数为12-3=9。所有满足这三个等式的节点向量都可以写成P=P_0+Nz，其中z为9维潜变量。由AN=0可知，改变z只会改变允许的曲率分配，不会破坏总长度和终点位置。"),
            ("formula", "hard_decode", "满足三个线性几何约束的潜空间解码关系"),
            ("figure", "decoder", "图7-3 潜变量、硬约束解码器与轨迹输出关系", 5.7),
            ("p", "潜变量z不是九个具有固定物理名称的飞机参数。零空间的每个基向量通常会同时改变多个半径节点，因此单个z_j更适合理解为一种保持终端约束不变的全局形状变形模式。九个潜变量共同决定侧部、顶部和底部的半径如何重新分配。半径上下界、最大载荷和迎角等非线性条件无法由这三个线性等式保证，仍需在轨迹解码后检查。"),
        ],
        [
            ("h3", "7.1.4 扩散模型在潜空间中的作用及输入输出"),
            ("p", "扩散模型学习的是条件分布p(z|c)，也就是在给定动作和任务条件c时，哪些九维潜变量更接近合格专家轨迹。选择在潜空间而不是直接在数百个x、z离散点上扩散，有三方面作用：一是把学习对象压缩为少量形状自由度；二是通过P=P_0+Nz使所有生成结果先天满足三个线性边界；三是保留B样条和轨迹积分的物理可解释性。扩散网络负责产生形状变化，B样条负责把节点变成连续半径函数，两者承担不同任务。"),
            ("p", "训练阶段和生成阶段的“输入、输出”需要分别说明。训练时先从专家轨迹得到真实潜变量z_0，在随机扩散步t加入高斯噪声得到z_t。去噪网络的直接输入是带噪潜变量z_t、时间步t和条件向量c，单次网络调用的直接输出是所加噪声的估计ε_hat，而不是x-z轨迹。训练目标是使ε_hat接近真实噪声ε。"),
            ("p", "生成时不再输入专家潜变量，而是输入九维标准高斯噪声z_T和指定条件c。网络在多个反向时间步连续预测噪声并更新当前变量，最终得到去噪后的潜变量z_0。随后由零空间关系恢复12个半径节点，再经B样条与积分形成轨迹。因此，从网络层面看，模型逐步输出噪声估计；从完整生成系统看，扩散模块最终提供九维潜变量；从项目交付层面看，解析解码与时间参数化最终输出完整轨迹—姿态时序。"),
            ("numbered", [
                "指定筋斗或殷麦曼动作，并给出目标长度和终端高度条件。",
                "从九维高斯噪声出发，执行条件反向去噪并得到潜变量z。",
                "利用P=P_0+Nz恢复满足三个线性约束的12个半径节点。",
                "采用三次B样条获得R(θ)，积分形成x-z几何轨迹。",
                "建立速度、时间和姿态参考，完成物理筛选并选出最终轨迹。",
            ]),
        ],
        [
            ("h3", "7.1.5 专家训练样本的构建"),
            ("p", f"扩散模型需要学习合格潜变量的统计分布，因此首先构建程序化专家样本库。程序在九维零空间内生成平滑随机形状，经P=P_0+Nz恢复半径节点，再依次完成B样条插值、轨迹积分、时间参数化和规划级动力学计算。只有同时满足半径范围、平滑性、速度、过载、迎角、升力裕度和姿态角速度要求的样本才被保留。接受样本对应的潜变量作为扩散训练中的z_0。"),
            ("p", f"训练集共包含{data_summary['dataset_size']}条专家样本，其中筋斗和殷麦曼各3000条。半径节点限制在20.5～46.0 m；均方根加加速度不高于11.5 m/s³，曲率变化率均方根不高于0.012 (1/m)/s。为获得3000条合格筋斗样本共尝试5211次，接受率为{100*data_summary['acceptance_rate_by_action']['loop_360']:.2f}%；殷麦曼共尝试6375次，接受率为{100*data_summary['acceptance_rate_by_action']['immelmann']:.2f}%。殷麦曼接受率较低，说明半筋斗终点、顶部滚转与侧向需求形成了更窄的可行区域。"),
            ("figure", "dataset", "图7-2 专家样本构建与拒绝采样流程", 5.7),
            ("p", f"训练样本平均均方根加加速度为{data_summary['expert_rms_jerk_mean_mps3']:.3f} m/s³，平均曲率变化率为{data_summary['expert_curvature_rate_mean_1pm_s']:.5f} (1/m)/s。该样本库来自程序生成和规划级筛选，不是实飞数据。模型学习的是当前规则和气动筛查下的可行形状先验，尚未包含阵风、传感器误差、执行机构迟滞和真实闭环跟踪误差。"),
        ],
        [
            ("h3", "7.1.6 前向扩散、条件去噪与训练目标"),
            ("p", "训练前先按训练集统计量对专家潜变量z_0进行标准化，使九个方向具有相近数值尺度。前向扩散过程在随机时间步t向z_0加入高斯噪声ε，得到带噪变量z_t。累计噪声系数ᾱ_t决定原始信号和噪声各自所占比例：t较小时z_t仍接近专家形状，t增大后有效结构逐渐减弱，最终接近标准高斯分布。项目采用120个扩散步和余弦噪声日程，覆盖从轻微扰动到近似纯噪声的全过程。"),
            ("formula", "forward", "潜变量前向加噪分布"),
            ("p", "去噪网络采用残差多层感知机。时间步t先转换为时间嵌入，再与z_t及四维条件c共同进入网络。模型输出噪声估计ε_θ(z_t,t,c)，训练标签就是前向过程中实际加入的ε，损失函数为二者的均方误差。随机抽取不同t进行训练，相当于让网络在不同噪声强度下反复学习如何把样本指向专家潜变量分布。"),
            ("formula", "loss", "条件扩散模型的噪声预测损失"),
            ("p", "该损失衡量的是潜空间噪声预测精度，不能直接等同于轨迹误差或动力学可行性。验证损失下降只说明网络更好地恢复训练分布，最终仍必须将z解码为轨迹，并用终点误差、曲率、载荷和气动指标验证。"),
        ],
        [
            ("h3", "7.1.7 条件输入与无分类器引导"),
            ("p", "条件向量c共四维，由两维动作类别独热编码、目标长度变化量和终端高度变化量组成。动作编码[1，0]和[0，1]用于区分完整筋斗与殷麦曼；两个连续量用于描述相对基准动作尺度的小范围变化。对应约束目标变化时，程序同时更新特解P_0，使解析边界和网络条件保持一致。"),
            ("p", "训练过程中以0.12的概率丢弃条件，使同一个网络同时学习有条件和无条件噪声预测。采样时将两种预测进行组合：无条件分支提供总体专家形状先验，有条件分支把样本推向指定动作与任务尺度。这种方法称为无分类器引导，不需要额外训练动作分类器。"),
            ("formula", "cfg", "无分类器引导的条件噪声估计"),
            ("p", "采样时用引导系数w=1.45放大有条件预测相对无条件预测的差异。w过小会削弱动作条件，可能导致两类轨迹分布混合；w过大则容易压缩多样性并把样本推离训练分布。当前值用于在条件一致性与候选差异之间取得工程折中，不把它作为对所有场景都最优的固定参数。"),
            ("p", "本次最终指标来自两个基准动作条件，尚不能据此证明模型可在任意长度和高度范围外推。长度、高度、航点或障碍条件的扩展需要相应覆盖范围的训练样本，并应通过专门的条件插值和外推试验评价。"),
        ],
        [
            ("h3", "7.1.8 DDIM反向去噪与批量候选生成"),
            ("p", "推理阶段从256组九维标准高斯噪声同时开始，并采用60步DDIM完成反向去噪。每一步网络根据当前z_t、时间步和条件预测噪声，再计算对无噪潜变量的估计并更新到较低噪声等级。DDIM在给定初始噪声时采用确定性更新，因此同一条件、模型权重和随机种子能够复现结果，同时减少完整120步反向扩散的计算量。"),
            ("formula", "ddim", "DDIM反向去噪更新"),
            ("p", "当反向过程到达t=0时，每组初始噪声对应一个潜变量候选。相同动作条件和边界条件不会消除初始噪声带来的差异，因此候选可以具有不同的侧部、顶部和底部半径分配；同时，零空间解码保证这些形状变化不会破坏三个线性几何约束。批量采样由此把扩散模型学习到的条件分布转化为一组可供比较的轨迹形状。"),
            ("p", "DDIM只负责从噪声恢复潜变量，并不检查曲率半径、过载或气动裕度。每个潜变量仍需完整解码和评价，去噪收敛也不能代替飞机物理可行性判定。"),
        ],
        [
            ("h3", "7.1.9 Best-of-N物理筛选与最终输出"),
            ("figure", "selection", "图7-4 批量扩散采样与Best-of-N物理筛选", 5.7),
            ("p", "不同初始噪声可能到达条件分布中的不同区域，因此形成一组动作边界相同、局部曲率不同的候选。候选首先按硬边界、半径、速度包线、过载、迎角、升力裕度和姿态角速率剔除不可行解；随后对保留轨迹计算综合质量分数，主要权衡均方根加加速度、曲率变化率、最大迎角、最大过载、升力裕度倒数和半径粗糙度。最终选择分数最低的候选，而不是直接采用第一条随机样本。"),
            ("formula", "selection", "Best-of-N候选轨迹综合质量评分"),
            ("p", "式中带下标n的指标均按程序中的参考尺度归一化：加加速度除以8.0，曲率变化率除以0.006，最大迎角除以12度，最大过载除以2.5；升力裕度项采用1/max(M_L，0.1)，半径粗糙度除以2.0。P_margin=max(1-M_L，0)/0.05用于额外惩罚升力裕度低于1的候选。该表达与候选评分代码保持一致。"),
            ("p", "Best-of-N体现的是扩散模型的候选搜索能力。它不意味着任意一次随机采样都优于传统优化，而是利用生成分布扩大可选范围，再用统一、可解释的固定翼指标确定输出。如果一批候选中没有轨迹通过硬性筛查，系统应拒绝输出并记录失败原因。"),
        ],
        [
            ("h3", "7.1.10 从潜变量到轨迹—姿态时序"),
            ("p", "选中潜变量经过零空间解码和三次B样条积分后得到几何路径。随后沿用传统方法部分的同一速度规律：动作底部速度较高、顶部速度较低，并由dt=ds/V沿弧长积分构造时间轴。位置对时间求导得到速度、加速度和加加速度；累计航迹转角用于形成纵向姿态方向，并进一步计算俯仰角速度。"),
            ("p", "360度筋斗的滚转角指令保持0度。殷麦曼的纵向质心轨迹由扩散模型生成，但顶部半滚转目前采用确定性规则：当累计航迹转角由120度变化至180度时，使用五次平滑函数使滚转角从0度过渡到180度，并使区间两端的滚转角速度和角加速度接近零。也就是说，当前网络生成半筋斗几何形状，顶部滚转由解析姿态调度与其同步，尚不是神经网络直接联合生成。"),
            ("p", "最终统一输出t、x、y、z、速度与加速度分量、航迹转角、滚转角、机体系角速度、法向过载、估算迎角和升力裕度等时间序列。扩散模型、五次多项式和B样条方法使用相同的速度规律、姿态映射和评价器，使后续几何与动力学对比只反映轨迹生成方法的差异。"),
        ],
        [
            ("h3", "7.1.11 训练配置、收敛过程与模型产物"),
            ("table", ["类别", "参数", "设置", "说明"], [
                ["数据", "训练样本", "6000条", "筋斗与殷麦曼各3000条"],
                ["模型", "潜变量/条件维数", "9 / 4", "形状坐标与任务条件"],
                ["网络", "宽度/残差块", "160 / 5", "残差多层感知机"],
                ["扩散", "训练/采样步数", "120 / 60", "余弦日程与DDIM"],
                ["训练", "步数/批量", "6000 / 256", "90%训练，10%验证"],
                ["生成", "候选数/引导系数", "256 / 1.45", "Best-of-N筛选"],
            ], [0.62, 1.25, 0.95, 2.0], 8.1),
            ("figure", "training", "图7-5 条件潜空间扩散模型训练过程", 5.75),
            ("p", f"模型在CPU上训练{training_summary['train_steps']}步，用时{training_summary['training_time_s']:.1f} s。训练初期噪声预测损失快速下降，随后进入缓慢收敛阶段；最佳验证噪声损失为{training_summary['best_validation_noise_loss']:.4f}，出现在第{training_summary['best_training_step']}步。最终推理使用指数滑动平均权重，减小单次参数更新造成的波动。"),
            ("p", "训练与验证损失未出现持续分离，说明当前网络能够恢复程序化专家样本的潜变量分布。模型权重、训练日志、专家数据集和配置文件均单独保存，以便复现实验。需要再次强调，噪声预测损失只用于判断训练状态；模型是否具有工程价值，应由解码后的边界满足率、候选可行率、几何指标和动力学指标共同判定。"),
        ],
        [
            ("h2", "7.2 生成轨迹的几何对比与指标评估"),
            ("h3", "7.2.1 测试方法与评价指标"),
            ("p", "几何测试对两种动作分别运行五次多项式、B样条约束优化和条件潜空间扩散模型。三种方法使用相同动作长度、起终点、速度规律与采样间隔。传统方法各输出一条确定轨迹；扩散模型主要试验输出256条候选，并额外使用10个随机种子、每个种子64条候选进行鲁棒性测试。"),
            ("p", "统一指标包括轨迹长度、终端误差、最小曲率半径、均方根加加速度和均方根曲率变化率。扩散专属指标包括：硬边界满足率，用于判断解析解码是否可靠；规划级可行率，用于衡量候选分布落入当前包线的比例；曲率候选多样性，用候选半径曲线之间的均方根差描述；Best-of-N收益，用被选轨迹相对可行候选平均质量分数的下降比例描述；随机种子成功率与选中分数变异系数，用于评价重复生成稳定性。"),
            ("table", ["动作", "方法", "长度/m", "终端误差/m", "最小半径/m", "RMS加加速度", "曲率变化率RMS"], geometry_rows, [0.8, 1.05, 0.75, 0.9, 0.9, 1.0, 1.12], 7.5),
            ("status", "判据", "终端位置误差不大于0.05 m，最小曲率半径不小于20 m；平滑性指标用于相对比较，不把某一种方法预设为必然最优。"),
        ],
        [
            ("h3", "7.2.2 360度筋斗几何结果"),
            ("figure", "loop_geometry", "图7-6 360度筋斗三种方法几何与平滑性对比", 5.85),
            ("p", f"扩散模型选中轨迹长度为{f(dl, '轨迹长度_m'):.3f} m，闭合误差为{f(dl, '终端位置误差_m'):.3f} m，最小曲率半径为{f(dl, '最小曲率半径_m'):.2f} m。其平面轨迹与B样条基线整体接近圆形，但局部半径分配不同；半径始终高于20 m筛查线。硬约束解码使闭合条件在候选层面就得到保证，而不是依赖生成后再次平移终点。"),
            ("p", f"扩散轨迹均方根加加速度为{f(dl, '均方根jerk_mps3'):.3f} m/s³，相比B样条的{f(bl, '均方根jerk_mps3'):.3f} m/s³降低{-pct(f(dl, '均方根jerk_mps3'), f(bl, '均方根jerk_mps3')):.2f}%；曲率变化率均方根为{f(dl, '均方根曲率变化率_1pm_s'):.5f} (1/m)/s，相比B样条降低{-pct(f(dl, '均方根曲率变化率_1pm_s'), f(bl, '均方根曲率变化率_1pm_s')):.2f}%。与五次多项式相比，两项平滑指标改善更明显。"),
            ("status", "筋斗几何结论", "选中扩散轨迹满足闭合与最小半径要求，并在当前采样中得到略低于B样条的加加速度和更低的曲率变化率；该优势是Best-of-N筛选后的结果，不代表任意随机候选都优于B样条。"),
        ],
        [
            ("h3", "7.2.3 360度筋斗候选多样性与筛选收益"),
            ("figure", "loop_diversity", "图7-7 360度筋斗扩散候选轨迹族与筛选结果", 5.85),
            ("p", f"256条筋斗候选全部满足长度和终点硬边界，其中{100*f(ul, '规划级可行率'):.2f}%通过规划级约束。候选曲率半径曲线之间的RMS差异为{f(ul, '曲率候选多样性_RMS_m'):.2f} m，说明模型没有退化为只重复一条固定轨迹。图中浅色曲线表现为不同侧部和顶部半径分配，红色轨迹则是物理筛选后的输出。"),
            ("p", f"筋斗Best-of-N筛选收益为{100*f(ul, 'Best_of_N筛选收益'):.2f}%。该指标表示最终候选的综合质量分数相对可行候选平均水平下降约五分之一。扩散模型在这里的优势不是单次采样绝对准确，而是能够用批量生成扩大搜索覆盖面，再通过可解释的固定翼指标选择质量更高的样本。"),
            ("status", "多样性解释", "曲率差异不等同于“越大越好”。候选应在保持动作语义和物理可行的前提下提供适度差异；过高多样性若伴随低可行率，反而会增加筛选成本。"),
        ],
        [
            ("h3", "7.2.4 殷麦曼机动几何结果"),
            ("figure", "immelmann_geometry", "图7-8 殷麦曼机动三种方法几何与平滑性对比", 5.85),
            ("p", f"扩散模型殷麦曼轨迹长度为{f(di, '轨迹长度_m'):.3f} m，终端误差为{f(di, '终端位置误差_m'):.3f} m，最小曲率半径为{f(di, '最小曲率半径_m'):.2f} m。轨迹从动作底部进入半筋斗，在顶部以反向水平切线退出；几何终点和路径长度由硬约束解码保证。"),
            ("p", f"扩散轨迹均方根加加速度为{f(di, '均方根jerk_mps3'):.3f} m/s³，与B样条的{f(bi, '均方根jerk_mps3'):.3f} m/s³基本相同，变化为{pct(f(di, '均方根jerk_mps3'), f(bi, '均方根jerk_mps3')):+.2f}%。曲率变化率均方根为{f(di, '均方根曲率变化率_1pm_s'):.5f} (1/m)/s，相比B样条变化{pct(f(di, '均方根曲率变化率_1pm_s'), f(bi, '均方根曲率变化率_1pm_s')):+.2f}%。因此本动作上扩散结果与B样条处于同一水平，并非所有平滑指标都占优。"),
            ("status", "殷麦曼几何结论", "扩散模型可靠复现半筋斗几何与反向退出边界，显著优于五次多项式的曲率波动，但与B样条相比主要体现为可生成多解，而不是单条轨迹全面更优。"),
        ],
        [
            ("h3", "7.2.5 殷麦曼候选多样性与可行率"),
            ("figure", "immelmann_diversity", "图7-9 殷麦曼扩散候选轨迹族与筛选结果", 5.85),
            ("p", f"殷麦曼256条候选的硬边界满足率仍为100%，规划级可行率为{100*f(ui, '规划级可行率'):.2f}%，低于筋斗的{100*f(ul, '规划级可行率'):.2f}%。候选曲率多样性为{f(ui, '曲率候选多样性_RMS_m'):.2f} m，高于筋斗主要批次。这说明模型能够探索更多半筋斗局部形状，但一部分形状会在迎角、升力裕度或平滑性筛查中被剔除。"),
            ("p", f"殷麦曼Best-of-N收益为{100*f(ui, 'Best_of_N筛选收益'):.2f}%，略高于筋斗。较低可行率和较高筛选收益同时出现，表明该动作的候选质量分布更宽，筛选步骤对最终结果更重要。报告因此不只展示选中轨迹，还保留候选指标CSV和曲率半径样本文件，便于后续调整权重或加入舵面约束后重新排序。"),
            ("status", "改进方向", "若后续希望提高殷麦曼可行率，可增加该动作的高裕度专家样本、把滚转区间指标加入训练条件，或使用物理引导对去噪过程进行约束；不应简单删除不利指标。"),
        ],
        [
            ("h3", "7.2.6 去噪过程与生成机理观察"),
            ("figure", "denoising", "图7-10 两种动作DDIM反向去噪过程诊断", 5.8),
            ("p", "左图给出每个反向步预测的无噪潜变量范数，右图给出相邻去噪步预测变化。两种动作从高噪声端开始时预测变化很大，在前若干步迅速下降，随后保持较小变化，说明采样过程逐渐稳定到训练分布附近。潜变量范数并不要求单调下降，因为最终轨迹形状本身对应非零潜变量；真正应观察的是相邻预测变化是否收敛。"),
            ("p", "筋斗和殷麦曼曲线在早期下降趋势相似，说明同一网络学到了共享的固定翼曲率形状先验；中后期潜变量范数不同，则反映动作条件把样本引导到两个不同的条件分布。该图是对生成过程的诊断证据，不能单独证明动力学可行，最终判断仍以解码后轨迹和物理筛查为准。"),
            ("status", "模型独特性", "传统方法没有逐步去噪过程。保存去噪轨迹可以用于发现引导过强、采样发散或模式坍缩，是扩散模型测试中应保留的专属诊断。"),
        ],
        [
            ("h3", "7.2.7 随机种子鲁棒性"),
            ("figure", "robustness", "图7-11 扩散采样在10个随机种子下的鲁棒性", 5.85),
            ("p", f"鲁棒性测试对每个动作使用随机种子101～110，每个种子生成64条候选。两种动作的种子成功率均为100%，即每个随机种子至少得到一条规划级可行轨迹。筋斗平均可行率为{100*float(rl['mean_planning_feasible_rate']):.00f}%，最低为{100*float(rl['minimum_planning_feasible_rate']):.2f}%；殷麦曼平均可行率为{100*float(ri['mean_planning_feasible_rate']):.2f}%，最低为{100*float(ri['minimum_planning_feasible_rate']):.2f}%。"),
            ("p", f"筋斗与殷麦曼选中质量分数的变异系数分别为{100*float(rl['selected_score_coefficient_of_variation']):.2f}%和{100*float(ri['selected_score_coefficient_of_variation']):.2f}%，说明尽管候选形状会随随机种子变化，Best-of-N最终质量较稳定。64候选条件下平均筛选收益分别为{100*float(rl['mean_best_of_n_gain_ratio']):.2f}%和{100*float(ri['mean_best_of_n_gain_ratio']):.2f}%。"),
            ("status", "鲁棒性结论", "随机性带来了可选择性，但没有导致测试批次失效。实际工程中仍应固定并记录随机种子，同时保留候选筛查日志，以保证结果可复现。"),
        ],
        [
            ("h3", "7.2.8 扩散模型专属指标汇总与几何结论"),
            ("table", ["动作", "候选数", "硬边界满足率", "规划级可行率", "曲率多样性/m", "Best-of-N收益"], unique_rows, [1.0, 0.7, 1.1, 1.0, 1.0, 1.0], 8.1),
            ("figure", "key_metrics", "图7-12 三种方法关键指标总体对比", 5.8),
            ("p", "综合来看，扩散模型两条选中轨迹的几何质量接近B样条约束优化结果，并明显降低了五次多项式的加加速度和曲率波动。对筋斗，扩散模型在本批采样中得到最低曲率变化率；对殷麦曼，它与B样条接近但没有在每项指标上更优。扩散模型真正新增的能力是围绕同一动作生成可行轨迹族，并通过物理评分选择输出。"),
            ("p", f"计算代价也应同时报告。扩散模型主要批次的总生成与评价时间约为筋斗{f(dl, '生成时间_s'):.2f} s、殷麦曼{f(di, '生成时间_s'):.2f} s，远慢于五次多项式，但与B样条离线优化处于相近量级。扩散模型还需要约110 s的一次性CPU训练成本。当前定位应是离线高质量候选生成，而不是未经优化的机载实时规划。"),
            ("status", "几何总判定", "两种动作的最终扩散轨迹均通过几何筛查；候选多样性、筛选收益和种子鲁棒性证明模型具有传统单解方法不具备的生成特性。"),
        ],
        [
            ("h2", "7.3 基于真实飞机参数的动力学测试对比"),
            ("h3", "7.3.1 飞机参数、气动数据与评价范围"),
            ("p", "动力学评估使用当前已确认飞机参数：质量m=3.0 kg，机翼面积S=0.24 m²，翼展b=1.2 m，平均气动弦长c=0.2 m，转动惯量Ix=0.08、Iy=0.10、Iz=0.15 kg·m²。气动系数来自项目fixuav12222015数据中的VLM附着流表，并按迎角与侧向状态插值。三种轨迹方法调用同一套数据和评价函数。"),
            ("formula", "inverse", "参考轨迹逆动力学与法向过载估计"),
            ("formula", "aero", "法向气动系数需求与计入速度裕度的升力裕度"),
            ("p", "当前筛查线为：最小曲率半径不小于20 m，速度位于16.5～20.5 m/s，最大法向过载不高于3.0 g，估算迎角不高于15°，计入1.2倍速度裕度后的最小升力裕度不低于0.95，最大俯仰和滚转角速度分别不高于120°/s和240°/s，最大侧向力系数不高于0.25。"),
            ("status", "适用边界", "VLM表主要描述附着流，且尚缺可靠推力曲线、舵面限位、舵机速率和结构实测数据。因此本节只能判断参考运动需求是否落在当前假设包线内。"),
        ],
        [
            ("h3", "7.3.2 动力学测试方法与统一判据"),
            ("p", "对每条轨迹先根据速度方向和曲率计算切向加速度与法向加速度，再扣除重力得到飞机需要提供的合力。合力除以重量得到法向过载，根据动压和机翼面积转换为法向气动系数需求，随后在VLM表中反查所需迎角。由于实际控制存在速度跟踪误差，升力裕度按1.2倍速度不确定性折减。"),
            ("p", "殷麦曼动作还需要评价顶部滚转。当前滚转角在累计航迹转角120°～180°内平滑变化，其角速度与角加速度由统一时间轴计算。轨迹法向力投影到滚转后的机体系形成侧向力需求，用最大侧向力系数检查纵向轨迹与半滚转姿态之间的耦合。"),
            ("table", ["动作", "方法", "最大过载/g", "最大迎角/°", "最小升力裕度", "最大俯仰率/(°/s)", "判定"], dynamics_rows, [0.85, 1.05, 0.9, 0.9, 1.0, 1.15, 0.55], 7.5),
            ("p", "所有峰值都从完整时间序列统计，而不是只检查起点、顶部和终点。升力裕度曲线在所需升力接近零处可出现很大比值，因此图中对大于1.25的部分仅作显示截断；判定仍使用原始数据的全程最小值。"),
        ],
        [
            ("h3", "7.3.3 360度筋斗动力学表现"),
            ("figure", "loop_dynamics", "图7-13 360度筋斗三种方法动力学需求对比", 5.8),
            ("p", f"扩散轨迹最大法向过载为{f(dl, '最大过载_g'):.3f} g，低于3.0 g筛查线；最大估算迎角为{f(dl, '最大迎角需求_deg'):.3f}°，低于15°；最小升力裕度为{f(dl, '最小含速度裕度'):.3f}，高于0.95；最大俯仰角速度为{f(dl, '最大俯仰角速度_degps'):.2f}°/s，显著低于120°/s。四项指标均通过当前规划级筛查。"),
            ("p", f"与B样条相比，扩散轨迹最大过载由{f(bl, '最大过载_g'):.3f} g增至{f(dl, '最大过载_g'):.3f} g，迎角由{f(bl, '最大迎角需求_deg'):.3f}°增至{f(dl, '最大迎角需求_deg'):.3f}°，升力裕度由{f(bl, '最小含速度裕度'):.3f}降至{f(dl, '最小含速度裕度'):.3f}。这些差异较小且仍满足阈值，但说明Best-of-N在降低平滑性代价时对峰值载荷作了轻微折中。"),
            ("status", "筋斗动力学结论", "扩散轨迹满足当前飞机参数和VLM气动表下的规划级要求；其动力学表现接近B样条并明显平滑于五次多项式，但升力裕度没有超过B样条。"),
        ],
        [
            ("h3", "7.3.4 360度筋斗关键阶段分析"),
            ("p", "完整筋斗可分为底部拉起、竖直上升、顶部倒飞、竖直下降和底部改平五个阶段。底部速度较高且曲率较大，法向加速度与过载通常达到峰值；顶部速度较低，重力方向与轨迹法向关系发生变化，升力需求降低但速度保持能力更重要；侧部主要检验曲率变化和俯仰角速度。"),
            ("p", "扩散模型选中半径曲线在约60°附近达到约30 m低值，在150°～180°附近回升到约33 m，并在后半周形成相似但不完全镜像的分配。这种局部变化使曲率变化率低于两种传统方法，同时保持闭合边界。由于训练库允许非完全对称的平滑形状，模型能够在保持动作语义的前提下探索传统固定参数不易直接给出的局部折中。"),
            ("p", "目前评分函数只通过参考轨迹需求间接反映推进问题，没有将电机—螺旋桨最大推力、功率和响应延迟纳入硬约束。完整筋斗顶部是最可能出现速度不足的区段，因此后续六自由度仿真必须加入推进系统模型，并检查顶部最小空速、油门饱和持续时间和恢复余量。"),
            ("status", "控制关注点", "底部关注过载、迎角和升降舵需求；顶部关注空速、推力和倒飞姿态误差；整周关注轨迹闭合误差积累。"),
        ],
        [
            ("h3", "7.3.5 殷麦曼机动动力学表现"),
            ("figure", "immelmann_dynamics", "图7-14 殷麦曼机动三种方法动力学需求对比", 5.8),
            ("p", f"扩散殷麦曼轨迹最大过载为{f(di, '最大过载_g'):.3f} g，最大估算迎角为{f(di, '最大迎角需求_deg'):.3f}°，最小升力裕度为{f(di, '最小含速度裕度'):.3f}，最大俯仰角速度为{f(di, '最大俯仰角速度_degps'):.2f}°/s，均位于当前筛查范围。其最大滚转角速度为{f(di, '最大滚转角速度_degps'):.2f}°/s，最大侧向力系数为{f(di, '最大侧向力系数'):.3f}，也低于240°/s和0.25的暂定限制。"),
            ("p", f"与B样条相比，扩散结果最大过载略低{f(bi, '最大过载_g')-f(di, '最大过载_g'):.4f} g，但迎角增加{f(di, '最大迎角需求_deg')-f(bi, '最大迎角需求_deg'):.3f}°，升力裕度降低{f(bi, '最小含速度裕度')-f(di, '最小含速度裕度'):.003f}。滚转角速度增加约{f(di, '最大滚转角速度_degps')-f(bi, '最大滚转角速度_degps'):.2f}°/s，侧向力系数略低。各差异均小，三种方法均通过当前规划级判据。"),
            ("status", "殷麦曼动力学结论", "扩散轨迹与B样条的纵向及滚转需求处于同一水平，当前没有证据支持其在所有动力学峰值上更优；其优势仍是多候选生成和筛选能力。"),
        ],
        [
            ("h3", "7.3.6 殷麦曼顶部半滚转的结合性"),
            ("p", "殷麦曼质心轨迹由扩散模型生成，但顶部半滚转按照累计航迹转角触发。选择120°～180°区间，是为了避开底部高过载拉起阶段，并让滚转在反向水平退出前完成。五次平滑过渡保证滚转角在区间两端的角速度与角加速度接近零，减少姿态参考突变。"),
            ("p", "扩散轨迹局部曲率改变会影响滚转阶段的法向力大小，从而改变滚转后机体系侧向力需求。因此，即使滚转角时序规则完全相同，不同几何轨迹也会产生不同侧向力系数。当前扩散结果最大侧向力系数0.095，介于五次多项式0.083和B样条0.096之间。"),
            ("p", "该评估尚未根据副翼和方向舵气动导数反算舵偏角，也没有模拟滚转惯性、气动阻尼和侧滑反馈。真实控制器需要同时跟踪滚转角、滚转角速度、俯仰方向和速度，并抑制侧滑。后续若训练数据包含闭环或实飞姿态历史，可把滚转开始角、持续时间和峰值滚转率加入条件，逐步从“几何生成+规则姿态”升级为联合轨姿生成。"),
            ("status", "当前结论", "现有扩散轨迹能够为殷麦曼控制设计提供同步的质心轨迹和滚转参考，但还不能据此宣称实机一定可完成半滚转。"),
        ],
        [
            ("h3", "7.3.7 可行候选率的动力学含义"),
            ("p", "扩散模型的规划级可行率同时包含几何和平滑性及动力学筛查结果。筋斗主批次可行率96.48%，说明训练分布的大部分样本位于当前飞机包线内；殷麦曼可行率77.73%，说明约四分之一候选需要被拒绝。与传统方法只报告一个解是否通过相比，可行率能够描述整个生成分布与飞机约束之间的匹配程度。"),
            ("p", "可行率也能指导后续模型更新。如果真实舵面或推力限制加入后可行率明显下降，可以回溯每个候选的违反原因，补充相应工况的专家样本或改变条件输入，而不是只对最终轨迹作一次性手工修补。对于实机前的安全筛选，应设置“无可行候选则拒绝执行”的明确逻辑，并记录候选数、阈值和随机种子。"),
            ("p", "需要注意，可行率不是越高越好。若模型只重复极少数保守轨迹，可行率可能接近100%，但多样性和适应能力会下降。本项目同时报告曲率多样性、Best-of-N收益和选中质量分数变异系数，用多个维度区分“稳定可行”和“模式坍缩”。"),
            ("status", "扩散模型优势", "可行率把评价对象从一条选中轨迹扩展到整个候选分布，使模型与飞机包线的匹配程度可以量化，这是传统确定性单解方法没有的测试视角。"),
        ],
        [
            ("h3", "7.3.8 与后续轨姿跟踪控制的接口"),
            ("figure", "control", "图7-15 扩散轨姿进入后续闭环控制验证的接口", 5.75),
            ("p", "最终CSV统一输出t、x、y、z、速度与加速度分量、航迹转角、滚转角、机体系角速度、法向过载、估算迎角和升力裕度等量。位置和速度可进入外环轨迹跟踪器，航迹方向和滚转指令可形成姿态参考，角速度和加速度可作为内环与力控制前馈。数据接口已经满足后续控制算法开发的基本需要。"),
            ("p", "建议先用固定随机种子选出的当前轨迹开展控制联调，再逐步加入其他候选，检验控制器对形状变化的鲁棒性。跟踪指标应至少包括位置RMS/最大误差、速度误差、航迹角误差、滚转角误差、角速度误差、过载与迎角超限持续时间。筋斗重点统计顶部和底部，殷麦曼重点统计滚转开始、滚转结束和反向退出。"),
            ("status", "用途", "扩散模型的候选多样性可自然形成一组控制器测试轨迹，而不是只用单一参考验证一次；这有助于评估控制算法的鲁棒性。"),
        ],
        [
            ("h3", "7.3.9 从规划级筛查到实机演示的验证路径"),
            ("figure", "dynamics_scope", "图7-16 当前结果与实机演示之间的验证层级", 5.75),
            ("p", "现阶段已完成轨迹—姿态参考生成和基于真实飞机参数、VLM气动表的逆动力学筛查。下一阶段应在六自由度模型中加入纵横侧向气动导数、推力模型、重心位置和执行机构模型，设计轨迹外环与姿态角速度内环，比较期望轨姿与实际状态。闭环通过后再进入半实物或飞控在环测试。"),
            ("numbered", [
                "补齐推力—油门—空速曲线、舵面限位与舵机速率。",
                "在六自由度模型中验证两种动作的全程跟踪误差和控制饱和。",
                "采用缩小幅度、提高初始高度和安全员接管的渐进式试飞方案。",
                "先完成半筋斗或较保守筋斗，再逐步逼近报告中的完整动作尺度。",
                "用飞行日志更新气动参数与训练样本，并重新进行候选筛查。",
            ]),
            ("status", "安全声明", "规划级“通过”不是实飞放行结论。缺少推进、舵面、失速和结构实测数据时，不应直接把当前参考轨迹上传实机执行。"),
        ],
        [
            ("h3", "7.3.10 综合结论、优势与局限"),
            ("p", "本项目实现了同一条件潜空间扩散模型对360度筋斗和殷麦曼机动的轨迹生成。模型在9维曲率形状潜空间中学习6000条规划级专家样本，利用解析零空间解码严格满足动作长度与终点边界，使用DDIM批量生成候选，并由真实飞机参数和VLM气动表支持的统一评价器完成Best-of-N筛选。两条最终轨迹均通过当前几何和动力学筛查。"),
            ("p", "在单条轨迹指标上，扩散模型明显优于五次多项式的加加速度与曲率波动，并总体接近B样条约束优化。筋斗扩散轨迹的均方根加加速度和曲率变化率略优于B样条；殷麦曼两者基本相当，扩散模型并未在迎角、升力裕度或每一项平滑指标上全面领先。因此，结论不应写成扩散模型必然优于所有传统方法。"),
            ("p", "扩散模型的独特优势由分布级结果体现：两种动作硬边界满足率均为100%，主批次规划级可行率分别为96.48%和77.73%，曲率候选多样性约2.14 m和2.45 m，Best-of-N收益约20.78%和22.52%；10个随机种子均能生成可行结果，选中质量分数变异系数低于3%。这些证据表明模型能够稳定提供一组具有差异的可行候选，并用物理指标提高最终输出质量。"),
            ("p", "当前局限包括：训练样本来自程序化专家库而非实飞数据；动作条件种类仍少；殷麦曼滚转由确定性规则给出；VLM附着流模型不能覆盖失速和强非定常气动；尚未加入推进、舵面和六自由度闭环控制。后续工作应首先补齐飞机执行能力数据和闭环模型，再把控制可跟踪性、障碍或航点条件纳入训练与筛选。"),
            ("status", "本阶段判定", "条件潜空间扩散模型已形成可复现的筋斗与殷麦曼轨姿生成原型，能够支持后续控制设计和多候选鲁棒性试验；实物演示仍需按控制级验证路径逐级完成。"),
        ],
    ]
    pages = [
        [block for block in page if block[0] != "status"]
        for page in pages
    ]
    if len(pages) < 24:
        raise RuntimeError(f"扩散模型详细报告页面单元不足: {len(pages)}")
    return pages


def set_run_font(run, east_asia: str, ascii_font: str, size: float, bold=False):
    HELPER.set_run_font(run, east_asia, ascii_font, size, bold=bold)


def add_heading(document, text: str, level: int) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.space_before = HELPER.Pt(0 if level == 1 else 3)
    paragraph.paragraph_format.space_after = HELPER.Pt(6 if level <= 2 else 4)
    run = paragraph.add_run(text)
    set_run_font(run, "黑体", "Arial", {1: 16, 2: 14, 3: 12}[level], bold=True)


def add_formula(document, path: Path, description: str) -> None:
    paragraph = document.add_paragraph(style="Report Equation")
    paragraph.alignment = HELPER.WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = HELPER.Pt(2)
    paragraph.paragraph_format.space_after = HELPER.Pt(4)
    run = paragraph.add_run()
    run.add_picture(str(path), width=HELPER.Inches(5.75))
    for prop in run._r.xpath(".//wp:docPr"):
        prop.set("title", description)
        prop.set("descr", description)


def add_header(section) -> None:
    section.different_first_page_header_footer = False
    section.header.is_linked_to_previous = False
    HELPER.clear_header_footer(section.header)
    paragraph = section.header.paragraphs[0]
    paragraph.alignment = HELPER.WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("任务二测试报告｜扩散模型高机动轨姿生成")
    set_run_font(run, "宋体", "Arial", 8.5)
    run.font.color.rgb = HELPER.RGBColor.from_string("64748B")


def add_numbered_restart(document, items) -> None:
    """Add a real numbered list that restarts at 1 for each procedure."""
    base_id = HELPER.ensure_list_numbering(document)["number"]
    numbering = document.part.numbering_part.element
    base_num = next(
        node
        for node in numbering.findall(HELPER.qn("w:num"))
        if int(node.get(HELPER.qn("w:numId"))) == base_id
    )
    abstract_id = base_num.find(HELPER.qn("w:abstractNumId")).get(
        HELPER.qn("w:val")
    )
    num_ids = [
        int(node.get(HELPER.qn("w:numId")))
        for node in numbering.findall(HELPER.qn("w:num"))
    ]
    new_id = max(num_ids, default=0) + 1
    num = HELPER.OxmlElement("w:num")
    num.set(HELPER.qn("w:numId"), str(new_id))
    abstract_ref = HELPER.OxmlElement("w:abstractNumId")
    abstract_ref.set(HELPER.qn("w:val"), abstract_id)
    num.append(abstract_ref)
    override = HELPER.OxmlElement("w:lvlOverride")
    override.set(HELPER.qn("w:ilvl"), "0")
    start = HELPER.OxmlElement("w:startOverride")
    start.set(HELPER.qn("w:val"), "1")
    override.append(start)
    num.append(override)
    numbering.append(num)
    for item in items:
        paragraph = document.add_paragraph(style="Report Body")
        HELPER.apply_list_numbering(paragraph, new_id)
        paragraph.paragraph_format.space_after = HELPER.Pt(2)
        paragraph.paragraph_format.line_spacing = 1.15
        run = paragraph.add_run(item)
        set_run_font(run, "宋体", "Times New Roman", 10)


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
    elif kind == "numbered":
        add_numbered_restart(document, block[1])
    else:
        HELPER.render_block(document, block, figures)


def prepare_assets() -> None:
    global plt, patches
    import matplotlib.pyplot as matplotlib_pyplot
    import matplotlib.patches as matplotlib_patches

    plt = matplotlib_pyplot
    patches = matplotlib_patches
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    make_diagrams()
    make_formula_images()
    print(f"报告素材目录: {ASSET_DIR}")


def assemble_docx() -> None:
    global HELPER
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

    HELPER = load_module("diffusion_report_docx_helper", HELPER_PATH)
    actual_hash = hashlib.sha256(REFERENCE_DOCX.read_bytes()).hexdigest().upper()
    if actual_hash != REFERENCE_SHA256:
        raise RuntimeError("任务一参考报告哈希发生变化，停止生成。")

    config = load_json(CONFIG_PATH)
    metrics = load_csv(ALL_METRICS_PATH)
    unique = load_csv(UNIQUE_METRICS_PATH)
    robustness = load_csv(ROBUSTNESS_PATH)
    data_summary = load_json(DATA_SUMMARY_PATH)
    training_summary = load_json(TRAINING_SUMMARY_PATH)
    figures = figure_paths()
    formulas = formula_paths()
    pages = build_pages(config, metrics, unique, robustness, data_summary, training_summary)

    document = HELPER.Document(REFERENCE_DOCX)
    HELPER.clear_document_body(document)
    HELPER.normalize_styles(document)
    for section in document.sections:
        HELPER.configure_section(section)
        add_header(section)
        HELPER.clear_header_footer(section.footer)
        HELPER.add_footer(section)

    document.core_properties.title = "任务二测试报告：扩散模型轨姿生成方法"
    document.core_properties.subject = "360度筋斗与殷麦曼机动条件潜空间扩散模型测试"
    document.core_properties.comments = "规划级测试报告，待项目组审阅"

    for index, blocks in enumerate(pages):
        if index:
            HELPER.add_page_break(document)
        for block in blocks:
            render_block(document, block, figures, formulas)

    HELPER.set_update_fields(document)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT_DOCX)
    HELPER.remove_unreferenced_template_headers(OUTPUT_DOCX)
    HELPER.update_page_metadata(OUTPUT_DOCX, len(pages))
    audit = HELPER.structural_audit(OUTPUT_DOCX)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    (QA_DIR / "structural_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created: {OUTPUT_DOCX}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-assets", action="store_true")
    mode.add_argument("--assemble-docx", action="store_true")
    args = parser.parse_args()
    if args.prepare_assets:
        prepare_assets()
    else:
        assemble_docx()


if __name__ == "__main__":
    main()
