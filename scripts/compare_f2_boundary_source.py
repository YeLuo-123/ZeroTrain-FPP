#!/usr/bin/env python3
"""Compare frequency-2 boundary completion with Exp3 internal superposition."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fringe_repair.f2_boundary import (
    InternalSuperposition,
    boundary_from_order,
    circular_error,
    complete_boundary_rows,
    demodulate_three_step,
    detect_boundary_rows,
    make_boundary_gap,
    make_scene,
    order_from_boundary,
    render_three_step,
    unresolved_order_from_observations,
    unwrap_high_from_frequency_two,
    wrap_positive,
)


METHODS = {
    "traditional": "Traditional 3-step",
    "source_ips": "Source IPS (12 internal)",
    "boundary": "F2 boundary completion",
    "hybrid": "IPS + boundary completion",
    "oracle": "Oracle boundary (3-step)",
}
SCENARIOS = {
    "harmonic": "5th harmonic only",
    "gap": "boundary gap only",
    "combined": "harmonic + boundary gap",
}
SCENARIOS_ZH = {
    "harmonic": "仅五次谐波",
    "gap": "仅分界线缺失",
    "combined": "五次谐波与分界线缺失同时存在",
}


def one_acquisition(
    scene: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    source: bool,
    harmonic_ratio: float,
    gap_mask: np.ndarray | None,
    high_frequency: int,
    superposition: InternalSuperposition,
) -> dict[str, np.ndarray]:
    up = scene["projector_coordinate"]
    background = scene["background"]
    low_modulation = scene["modulation"].copy()
    high_modulation = scene["modulation"].copy()
    if gap_mask is not None:
        # The experiment targets a broken low-frequency phase-order boundary.
        # High-frequency observations retain partial contrast, as is required
        # for boundary recovery to be useful rather than inventing all data.
        low_modulation[gap_mask] *= 0.035
        high_modulation[gap_mask] *= 0.40

    low_true = 2.0 * np.pi * 2 * up
    high_true = 2.0 * np.pi * high_frequency * up
    low_images = render_three_step(
        low_true,
        background,
        low_modulation,
        rng,
        harmonic_ratio=harmonic_ratio,
        read_noise_sigma=1.5,
        source_superposition=source,
        superposition=superposition,
    )
    high_images = render_three_step(
        high_true,
        background,
        high_modulation,
        rng,
        harmonic_ratio=harmonic_ratio,
        read_noise_sigma=1.5,
        source_superposition=source,
        superposition=superposition,
    )
    low_wrapped, _, low_quality = demodulate_three_step(
        low_images, source_superposition=source, superposition=superposition
    )
    high_wrapped, _, high_quality = demodulate_three_step(
        high_images, source_superposition=source, superposition=superposition
    )
    return {
        "low_true": low_true,
        "high_true": high_true,
        "low_wrapped": low_wrapped,
        "high_wrapped": high_wrapped,
        "low_quality": low_quality,
        "high_quality": high_quality,
        "low_images": low_images,
    }


def evaluate_method(
    acquired: dict[str, np.ndarray],
    order_gt: np.ndarray,
    boundary_gt_columns: np.ndarray,
    *,
    high_frequency: int,
    completion: bool,
    oracle: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    height, width = order_gt.shape
    observed = detect_boundary_rows(acquired["low_wrapped"], acquired["low_quality"])
    if oracle:
        boundary_columns = boundary_gt_columns.copy()
    elif completion:
        boundary_columns = complete_boundary_rows(observed, width)
    else:
        boundary_columns = observed.copy()

    if oracle or completion:
        order = order_from_boundary(boundary_columns, width)
    else:
        order = unresolved_order_from_observations(observed, width)

    low_absolute = acquired["low_wrapped"] + 2.0 * np.pi * order
    high_absolute, high_order = unwrap_high_from_frequency_two(
        low_absolute, acquired["high_wrapped"], high_frequency
    )
    high_wrapped_true = wrap_positive(acquired["high_true"])
    high_order_gt = np.rint(
        (acquired["high_true"] - high_wrapped_true) / (2.0 * np.pi)
    ).astype(np.int32)

    phase_error = high_absolute - acquired["high_true"]
    wrapped_low_error = circular_error(
        acquired["low_wrapped"], wrap_positive(acquired["low_true"])
    )
    boundary_valid = np.isfinite(boundary_columns)
    boundary_error = np.abs(
        boundary_columns[boundary_valid] - boundary_gt_columns[boundary_valid]
    )
    metrics = {
        "boundary_coverage": float(boundary_valid.mean()),
        "boundary_mae_px": float(boundary_error.mean()) if len(boundary_error) else float("nan"),
        "boundary_within_3px": (
            float((boundary_error <= 3.0).mean()) if len(boundary_error) else 0.0
        ),
        "k2_accuracy": float((order == order_gt).mean()),
        "low_wrapped_mae_rad": float(np.mean(np.abs(wrapped_low_error))),
        "high_phase_mae_rad": float(np.mean(np.abs(phase_error))),
        "high_phase_rmse_rad": float(np.sqrt(np.mean(phase_error**2))),
        "high_bad_0_2": float((np.abs(phase_error) > 0.2).mean()),
        "high_order_accuracy": float((high_order == high_order_gt).mean()),
        "catastrophic_order_rate": float((np.abs(high_order - high_order_gt) >= 2).mean()),
    }
    arrays = {
        "observed_boundary": observed,
        "boundary_columns": boundary_columns,
        "order": order,
        "low_absolute": low_absolute,
        "high_absolute": high_absolute,
        "phase_error": phase_error,
        "high_order": high_order,
    }
    return metrics, arrays


def simulate_case(
    seed: int,
    scenario: str,
    *,
    height: int,
    width: int,
    high_frequency: int,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    scene_rng = np.random.default_rng(seed)
    scene = make_scene(height, width, scene_rng)
    low_true = 2.0 * np.pi * 2 * scene["projector_coordinate"]
    order_gt = np.floor(low_true / (2.0 * np.pi)).astype(np.int32)
    order_gt = np.clip(order_gt, 0, 1)
    boundary_gt = boundary_from_order(order_gt)
    boundary_gt_columns = np.argmax(boundary_gt, axis=1).astype(np.float64)

    gap_mask = None
    gap_rows = None
    if scenario in {"gap", "combined"}:
        gap_mask, gap_rows = make_boundary_gap(order_gt, scene_rng)
    harmonic_ratio = 0.25 if scenario in {"harmonic", "combined"} else 0.0
    superposition = InternalSuperposition()

    standard = one_acquisition(
        scene,
        np.random.default_rng(seed + 10_000),
        source=False,
        harmonic_ratio=harmonic_ratio,
        gap_mask=gap_mask,
        high_frequency=high_frequency,
        superposition=superposition,
    )
    source = one_acquisition(
        scene,
        np.random.default_rng(seed + 20_000),
        source=True,
        harmonic_ratio=harmonic_ratio,
        gap_mask=gap_mask,
        high_frequency=high_frequency,
        superposition=superposition,
    )

    configurations = {
        "traditional": (standard, False, False),
        "source_ips": (source, False, False),
        "boundary": (standard, True, False),
        "hybrid": (source, True, False),
        "oracle": (standard, True, True),
    }
    metrics: dict[str, dict[str, float]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, (acquired, completion, oracle) in configurations.items():
        metrics[name], arrays[name] = evaluate_method(
            acquired,
            order_gt,
            boundary_gt_columns,
            high_frequency=high_frequency,
            completion=completion,
            oracle=oracle,
        )
    details: dict[str, object] = {
        "scene": scene,
        "order_gt": order_gt,
        "boundary_gt": boundary_gt,
        "boundary_gt_columns": boundary_gt_columns,
        "gap_mask": gap_mask,
        "gap_rows": gap_rows,
        "standard": standard,
        "source": source,
        "arrays": arrays,
    }
    return metrics, details


def aggregate(all_metrics: dict[str, dict[str, list[dict[str, float]]]]) -> dict:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for scenario, methods in all_metrics.items():
        result[scenario] = {}
        for method, rows in methods.items():
            keys = rows[0]
            result[scenario][method] = {}
            for key in keys:
                values = np.array([row[key] for row in rows], dtype=np.float64)
                finite = values[np.isfinite(values)]
                result[scenario][method][key] = (
                    float(finite.mean()) if len(finite) else float("nan")
                )
                result[scenario][method][f"{key}_std"] = (
                    float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
                )
    return result


def plot_aggregate(metrics: dict, output: Path) -> None:
    method_order = ["traditional", "source_ips", "boundary", "hybrid", "oracle"]
    colors = ["#6c757d", "#2f6f9f", "#e07a2d", "#2a9d67", "#8e5aa9"]
    panels = [
        ("high_phase_mae_rad", "High absolute-phase MAE (rad)", False),
        ("k2_accuracy", "Frequency-2 order accuracy (%)", True),
        ("catastrophic_order_rate", "Catastrophic high-order rate (%)", True),
        ("boundary_coverage", "Resolved boundary rows (%)", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    x = np.arange(len(SCENARIOS))
    width = 0.15
    for axis, (metric, title, percent) in zip(axes.flat, panels):
        for index, (method, color) in enumerate(zip(method_order, colors)):
            values = [metrics[s][method][metric] for s in SCENARIOS]
            if percent:
                values = np.asarray(values) * 100.0
            axis.bar(
                x + (index - 2) * width,
                values,
                width,
                color=color,
                label=METHODS[method],
            )
        axis.set_xticks(x, list(SCENARIOS.values()), rotation=12, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if metric == "high_phase_mae_rad":
            axis.set_yscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Frequency-2 boundary completion vs. Exp3 internal superposition",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_representative(details: dict[str, object], output: Path) -> None:
    scene = details["scene"]
    order_gt = details["order_gt"]
    boundary_gt_columns = details["boundary_gt_columns"]
    gap_mask = details["gap_mask"]
    standard = details["standard"]
    arrays = details["arrays"]
    methods = ["traditional", "source_ips", "boundary", "hybrid"]

    fig, axes = plt.subplots(4, 5, figsize=(16, 13.5))
    axes[0, 0].imshow(scene["projector_coordinate"], cmap="viridis")
    axes[0, 0].set_title("Normalized projector coordinate")
    axes[1, 0].imshow(wrap_positive(standard["low_true"]), cmap="twilight", vmin=0, vmax=2*np.pi)
    axes[1, 0].set_title("Frequency-2 wrapped GT")
    axes[2, 0].imshow(order_gt, cmap="gray", vmin=0, vmax=1)
    axes[2, 0].set_title("Frequency-2 order GT")
    axes[3, 0].imshow(gap_mask, cmap="gray" if gap_mask is not None else "viridis")
    axes[3, 0].set_title("Simulated boundary-gap mask")

    vmax_error = 2.0 * np.pi
    for column, method in enumerate(methods, start=1):
        data = arrays[method]
        acquired = details["source"] if method in {"source_ips", "hybrid"} else standard
        axes[0, column].imshow(acquired["low_wrapped"], cmap="twilight", vmin=0, vmax=2*np.pi)
        axes[0, column].set_title(METHODS[method])
        axes[1, column].imshow(acquired["low_quality"], cmap="magma", vmin=0, vmax=0.85)
        observed = data["observed_boundary"]
        completed = data["boundary_columns"]
        rows = np.arange(len(observed))
        axes[1, column].plot(boundary_gt_columns, rows, "w--", lw=1.2, label="GT")
        axes[1, column].plot(observed, rows, color="#46d6ff", lw=1.0, label="detected")
        if np.isfinite(completed).all():
            axes[1, column].plot(completed, rows, color="#ffea55", lw=1.0, label="completed")
        axes[2, column].imshow(data["order"] != order_gt, cmap="Reds", vmin=0, vmax=1)
        axes[2, column].set_title(f"K2 error: {(data['order'] != order_gt).mean()*100:.2f}%")
        image = axes[3, column].imshow(
            np.abs(data["phase_error"]),
            cmap="inferno",
            vmin=0,
            vmax=vmax_error,
        )
        axes[3, column].set_title(
            f"|high phase error|, MAE={np.mean(np.abs(data['phase_error'])):.3f}"
        )

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    legend_handles = [
        Line2D([0], [0], color="white", linestyle="--", lw=1.5, label="GT"),
        Line2D([0], [0], color="#46d6ff", lw=1.3, label="detected"),
        Line2D([0], [0], color="#ffea55", lw=1.3, label="completed"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.925,
        bottom=0.04,
        top=0.875,
        wspace=0.055,
        hspace=0.28,
    )
    colorbar_axis = fig.add_axes((0.94, 0.065, 0.012, 0.17))
    fig.colorbar(image, cax=colorbar_axis, label="rad")
    fig.suptitle("Representative combined-degradation scene", fontsize=15, y=0.995)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_profiles(details: dict[str, object], output: Path) -> None:
    boundary_gt = details["boundary_gt_columns"]
    arrays = details["arrays"]
    gap_rows = details["gap_rows"]
    methods = ["traditional", "source_ips", "boundary", "hybrid"]
    colors = ["#6c757d", "#2f6f9f", "#e07a2d", "#2a9d67"]
    rows = np.arange(len(boundary_gt))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(boundary_gt, rows, "k--", lw=2, label="GT")
    for method, color in zip(methods, colors):
        axes[0].plot(
            arrays[method]["boundary_columns"],
            rows,
            color=color,
            lw=1.2,
            label=METHODS[method],
        )
    if gap_rows is not None:
        axes[0].axhspan(gap_rows[0], gap_rows[1], color="#f4a261", alpha=0.18)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Boundary column x")
    axes[0].set_ylabel("Image row y")
    axes[0].set_title("Boundary trajectory and missing interval")
    axes[0].grid(alpha=0.25)

    if gap_rows is None:
        profile_row = len(rows) // 2
    else:
        profile_row = (gap_rows[0] + gap_rows[1]) // 2
    true_phase = details["standard"]["high_true"][profile_row]
    axes[1].plot(true_phase, "k--", lw=2, label="GT")
    for method, color in zip(methods, colors):
        axes[1].plot(
            arrays[method]["high_absolute"][profile_row],
            color=color,
            lw=1.1,
            label=METHODS[method],
        )
    axes[1].set_xlabel("Image column x")
    axes[1].set_ylabel("Absolute phase (rad)")
    axes[1].set_title(f"High-frequency phase profile at damaged row y={profile_row}")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(metrics: dict, output: Path, scenes: int, high_frequency: int) -> None:
    lines = [
        "# 基频2分界线补全与源代码内部相移叠加方案仿真对比",
        "",
        "## 1. 实验目的",
        "",
        "本实验验证基频为2时，通过补全包裹相位级次分界线恢复二值级次"
        "$K_2\\in\\{0,1\\}$，是否能够降低后续高频相位展开中的灾难性级次错误。"
        "同时严格复现 `Exp3_Simulation.m` 的12步内部相移投影次数、整数取整以及"
        "$c_2/c_3$ 解调补偿，与分界线补全方案进行比较。",
        "",
        "两种方法解决的误差来源不同：内部相移叠加抑制五次谐波，分界线补全恢复"
        "低频级次拓扑。实验因此还包含二者结合的 Hybrid 组。",
        "",
        "## 2. 协议",
        "",
        f"- 场景：每种退化 {scenes} 个随机二维光滑表面；",
        f"- 频率：低频 $f_l=2$，高频 $f_h={high_frequency}$；",
        "- 相机采集：低频、高频各三幅外部相移图；",
        "- 源代码方案：每幅相机图像内部叠加12种相移状态，投影次数"
        "`[20,19,15,10,5,1,0,1,5,10,15,19]`；",
        "- 退化：五次谐波、低频分界线低调制度缺口、二者共同存在；",
        "- 分界线补全：高置信度跳变种子、全局/局部伪边界剔除及保形插值；",
        "- Oracle：使用真实分界线，但仍使用带噪声的标准三步包裹相位。",
        "",
        "当前环境没有 MATLAB/Octave，因此实验由 Python 对源代码公式进行数值等价"
        "复现，不是直接执行 `.m` 文件。该实验验证算法机制和相对趋势，不代替真实"
        "相机—投影仪采集。",
        "",
        "## 3. 平均结果",
        "",
    ]
    for scenario, scenario_label in SCENARIOS_ZH.items():
        lines.extend(
            [
                f"### {scenario_label}",
                "",
                "| 方法 | 边界覆盖率 | 边界≤3px | K2准确率 | 高频相位MAE/rad | 高频级次准确率 | 灾难性级次错误率 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            row = metrics[scenario][method]
            lines.append(
                f"| {METHODS[method]} "
                f"| {100*row['boundary_coverage']:.2f}% "
                f"| {100*row['boundary_within_3px']:.2f}% "
                f"| {100*row['k2_accuracy']:.2f}% "
                f"| {row['high_phase_mae_rad']:.4f} "
                f"| {100*row['high_order_accuracy']:.2f}% "
                f"| {100*row['catastrophic_order_rate']:.2f}% |"
            )
        lines.append("")

    combined = metrics["combined"]
    traditional = combined["traditional"]
    source = combined["source_ips"]
    boundary = combined["boundary"]
    hybrid = combined["hybrid"]
    gap = metrics["gap"]
    source_reduction = (
        1.0 - source["high_phase_mae_rad"] / traditional["high_phase_mae_rad"]
    )
    boundary_reduction = (
        1.0 - boundary["high_phase_mae_rad"] / traditional["high_phase_mae_rad"]
    )
    hybrid_reduction = (
        1.0 - hybrid["high_phase_mae_rad"] / traditional["high_phase_mae_rad"]
    )
    lines.extend(
        [
            "## 4. 结果解释",
            "",
            f"- 在组合退化下，源代码内部相移方案将低频包裹相位MAE从"
            f" `{traditional['low_wrapped_mae_rad']:.4f}` 降至"
            f" `{source['low_wrapped_mae_rad']:.4f} rad`，说明其五次谐波抑制机制有效；"
            "",
            f"- 仅使用分界线补全时，边界覆盖率由"
            f" `{100*traditional['boundary_coverage']:.2f}%` 提高到"
            f" `{100*boundary['boundary_coverage']:.2f}%`，K2准确率由"
            f" `{100*traditional['k2_accuracy']:.2f}%` 提高到"
            f" `{100*boundary['k2_accuracy']:.2f}%`；",
            "",
            f"- 在仅分界线缺失时，补全方案将K2准确率从"
            f" `{100*gap['traditional']['k2_accuracy']:.2f}%` 提高到"
            f" `{100*gap['boundary']['k2_accuracy']:.2f}%`，高频相位MAE从"
            f" `{gap['traditional']['high_phase_mae_rad']:.4f}` 降至"
            f" `{gap['boundary']['high_phase_mae_rad']:.4f} rad`；",
            "",
            f"- 在组合退化下，相对于传统三步，高频相位MAE经源代码方案、单独补线和"
            f" Hybrid 分别降低 `{100*source_reduction:.2f}%`、"
            f" `{100*boundary_reduction:.2f}%` 和 `{100*hybrid_reduction:.2f}%`。"
            f" Hybrid 的K2准确率为 `{100*hybrid['k2_accuracy']:.2f}%`，灾难性高频"
            f"级次错误率为 `{100*hybrid['catastrophic_order_rate']:.2f}%`，说明"
            "谐波抑制和边界补全具有互补性；",
            "",
            "- Oracle 使用真实边界，但采用仍含谐波的普通三步观测。因此在谐波与组合"
            "退化下，Hybrid 可以优于该 Oracle；Oracle 只代表“普通三步观测下边界"
            "完全正确”的上限，不是内部叠加观测的上限；",
            "",
            f"- 单独分界线补全在组合退化下虽然把K2准确率提高到"
            f" `{100*boundary['k2_accuracy']:.2f}%`，高频级次准确率仍只有"
            f" `{100*boundary['high_order_accuracy']:.2f}%`。这说明谐波造成的连续低频"
            "相位偏差会被频率比放大，仅补线不能替代相位质量改善。",
            "",
            "## 5. 结论与证据边界",
            "",
            "本实验能够验证“缺失的基频2级次边界是高频展开错误的重要来源，以及物理"
            "边界补全能否恢复该级次拓扑”。它不能证明神经网络已经有效，因为当前补全器"
            "是确定性的保形插值，不是训练模型；也不能证明真实反光、遮挡和运动场景的性能。下一步"
            "应以同一输入输出协议替换为边界—级次多任务网络，并在真实双频三步数据上验证。",
            "",
            "具体限制包括：场景假设每行只有一条可表示为 $x_\\Gamma(y)$ 的平滑边界；"
            "缺口处高频仍保留部分调制度；没有模拟完全无观测的大面积遮挡、多个不连通"
            "物体、边界分叉、相移间运动和传感器饱和；也没有相机—投影仪标定，因此"
            "报告的是相位误差而不是毫米深度误差。",
            "",
            "源代码方案每个相机曝光包含120次内部微投影。仿真假设微投影在线性相机"
            "积分中准确相加，未计入投影刷新时间、同步误差、运动和饱和。因此其相位优势"
            "不能直接等价为真实动态测量优势。",
            "",
            "## 6. 产物",
            "",
            "- `metrics.json`：全部均值和标准差；",
            "- `aggregate_metrics.png`：三类退化的总体指标；",
            "- `representative_combined.png`：组合退化代表性结果；",
            "- `boundary_and_phase_profiles.png`：分界线轨迹及损坏行高频相位剖面。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/f2_boundary_vs_source")
    parser.add_argument("--scenes", type=int, default=30)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--high-frequency", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    collected: dict[str, dict[str, list[dict[str, float]]]] = {
        scenario: defaultdict(list) for scenario in SCENARIOS
    }
    for scenario_index, scenario in enumerate(SCENARIOS):
        for index in range(args.scenes):
            seed = args.seed + scenario_index * 100_000 + index
            metrics, details = simulate_case(
                seed,
                scenario,
                height=args.height,
                width=args.width,
                high_frequency=args.high_frequency,
            )
            for method, row in metrics.items():
                collected[scenario][method].append(row)
    metrics = aggregate(collected)
    superposition = InternalSuperposition()
    payload = {
        "protocol": {
            "scenes_per_scenario": args.scenes,
            "height": args.height,
            "width": args.width,
            "low_frequency": 2,
            "high_frequency": args.high_frequency,
            "seed": args.seed,
            "source_internal_counts": superposition.counts.tolist(),
            "source_total_projection_count": superposition.total_count,
            "source_c2_c3": list(superposition.c2_c3),
            "source_fifth_harmonic_spectral_amplitude": superposition.spectral_amplitude(5),
            "implementation": "Python numerical equivalent of Exp3_Simulation.m",
        },
        "metrics": metrics,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_aggregate(metrics, output / "aggregate_metrics.png")
    hybrid_rows = collected["combined"]["hybrid"]
    hybrid_mae = np.array([row["high_phase_mae_rad"] for row in hybrid_rows])
    representative_index = int(np.argmin(np.abs(hybrid_mae - np.median(hybrid_mae))))
    representative_seed = args.seed + 2 * 100_000 + representative_index
    _, representative = simulate_case(
        representative_seed,
        "combined",
        height=args.height,
        width=args.width,
        high_frequency=args.high_frequency,
    )
    plot_representative(representative, output / "representative_combined.png")
    plot_profiles(representative, output / "boundary_and_phase_profiles.png")
    write_report(
        metrics,
        output / "comparison_report_zh.md",
        args.scenes,
        args.high_frequency,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
