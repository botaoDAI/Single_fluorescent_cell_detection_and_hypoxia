"""
Plot hypoxie vs normoxie density time courses from exported txt files.

Data layout (per condition directory):
    - One txt per initial density (Well1..Well6), e.g. "0_Gy_Well1_3exps.txt".
    - Two columns per file: mean density and std density (cells/µm^2).

The script builds one combined plot with 12 curves (6 hypoxie + 6 normoxie),
then makes one highlight plot per initial density where the focal pair keeps
full opacity and the other 10 curves are dimmed.
添加了到一定maximum的截止功能
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ------- Configuration -------
# Data roots
HYPOXIE_DIR = Path("../Cell_Radiation_Proliferation_Model/results txt for model 1015/hypoxie")
NORMOXIE_DIR = Path("../Cell_Radiation_Proliferation_Model/results txt for model 1015/normoxie")

# Choose which set of files to read (same template for both conditions).
# Switch to the 10 Gy set by replacing the template line below with:
# FILE_TEMPLATE = "Well{idx}_Incucyte_F98_10Gy_2025_10_15_smooth=35.txt"
FILE_TEMPLATE = "0_Gy_Well{idx}_3exps.txt"

# Initial density dictionary (x10^5/ml) keyed by file number.
INITIAL_DENSITIES = {
    1: 0.11,
    2: 0.22,
    3: 0.44,
    4: 0.88,
    5: 1.76,
    6: 3.52,
}

# Frame-to-time conversion (hours between frames)
TIME_STEP_HOURS = 1.5

# Where to save figures
OUTPUT_DIR = Path("./results 151025/hypoxie_normoxie_plots")

# 根据开关只画到 y 第一次达到 1.5e-3 的功能
CUT_AT_THRESHOLD = True  # 默认关闭以保持原行为
THRESHOLD_VALUE = 1.5e-3


def load_mean_std(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a txt file (two columns: mean, std)."""
    arr = np.loadtxt(path)
    if arr.ndim == 1:  # single row edge-case
        arr = arr.reshape(1, -1)
    mean = arr[:, 0]
    std = arr[:, 1] if arr.shape[1] > 1 else np.full_like(mean, np.nan)
    return mean, std


def load_condition_series(base_dir: Path, wells: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
    """Return per-well mean/std arrays for one condition."""
    data = {}
    expected_len = None
    for idx in wells:
        file_path = base_dir / FILE_TEMPLATE.format(idx=idx)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        mean, std = load_mean_std(file_path)
        data[idx] = {"mean": mean, "std": std, "path": file_path}
        expected_len = expected_len or len(mean)
        if len(mean) != expected_len:
            raise ValueError(f"Length mismatch for well {idx}: {len(mean)} vs {expected_len}")
    return data


def build_time_axis(n_points: int) -> np.ndarray:
    return np.arange(n_points, dtype=float) * TIME_STEP_HOURS


def maybe_truncate(time_axis: np.ndarray, y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Optionally truncate a curve at the first point reaching the threshold."""
    if not CUT_AT_THRESHOLD:
        return time_axis, y_values
    hit_indices = np.where(y_values >= THRESHOLD_VALUE)[0]  # 首次达到/超过阈值
    if hit_indices.size == 0:
        return time_axis, y_values  # 未达到则画全程
    end_idx = int(hit_indices[0])
    return time_axis[: end_idx + 1], y_values[: end_idx + 1]


def combined_plot(
    hypoxie: Dict[int, Dict[str, np.ndarray]],
    normoxie: Dict[int, Dict[str, np.ndarray]],
    time_axis: np.ndarray,
    output_path: Path,
) -> None:
    colors = plt.get_cmap("tab10").colors
    plt.figure(figsize=(15, 8))

    for i, idx in enumerate(sorted(INITIAL_DENSITIES.keys())):
        color = colors[i % len(colors)]
        density = INITIAL_DENSITIES[idx]
        h_x, h_y = maybe_truncate(time_axis, hypoxie[idx]["mean"])
        n_x, n_y = maybe_truncate(time_axis, normoxie[idx]["mean"])
        plt.plot(
            h_x,
            h_y,
            linestyle="--",
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=f"Well {idx} hypoxie ({density} x10^5/ml)",
        )
        plt.plot(
            n_x,
            n_y,
            linestyle="-",
            marker="o",
            markersize=3,
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=f"Well {idx} normoxie ({density} x10^5/ml)",
        )

    plt.xlabel("Time (h)", fontsize=22, fontweight="bold")
    plt.ylabel("Cell density (cells/µm²)", fontsize=22, fontweight="bold")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=13, ncol=2, frameon=True, framealpha=0.95)
    plt.title("Hypoxie vs Normoxie (all initial densities)", fontsize=22, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def highlight_plot(
    focus_idx: int,
    hypoxie: Dict[int, Dict[str, np.ndarray]],
    normoxie: Dict[int, Dict[str, np.ndarray]],
    time_axis: np.ndarray,
    output_path: Path,
    faded_alpha: float = 0.15,
) -> None:
    colors = plt.get_cmap("tab10").colors
    plt.figure(figsize=(15, 8))

    for i, idx in enumerate(sorted(INITIAL_DENSITIES.keys())):
        color = colors[i % len(colors)]
        density = INITIAL_DENSITIES[idx]
        emphasis = 1.0 if idx == focus_idx else faded_alpha
        linewidth = 2.2 if idx == focus_idx else 1.3
        h_x, h_y = maybe_truncate(time_axis, hypoxie[idx]["mean"])
        n_x, n_y = maybe_truncate(time_axis, normoxie[idx]["mean"])
        plt.plot(
            h_x,
            h_y,
            linestyle="--",
            color=color,
            linewidth=linewidth,
            alpha=emphasis,
            label=f"Well {idx} hypoxie ({density} x10^5/ml)",
        )
        plt.plot(
            n_x,
            n_y,
            linestyle="-",
            marker="o",
            markersize=3,
            color=color,
            linewidth=linewidth,
            alpha=emphasis,
            label=f"Well {idx} normoxie ({density} x10^5/ml)",
        )

    density_text = INITIAL_DENSITIES[focus_idx]
    plt.xlabel("Time (h)", fontsize=22, fontweight="bold")
    plt.ylabel("Cell density (cells/µm²)", fontsize=22, fontweight="bold")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=13, ncol=2, frameon=True, framealpha=0.95)
    plt.title(
        f"Well {focus_idx} ({density_text} x10^5/ml): hypoxie vs normoxie",
        fontsize=22,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    wells = sorted(INITIAL_DENSITIES.keys())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hypoxie_data = load_condition_series(HYPOXIE_DIR, wells)
    normoxie_data = load_condition_series(NORMOXIE_DIR, wells)

    n_points = len(next(iter(hypoxie_data.values()))["mean"])
    time_axis = build_time_axis(n_points)

    template_label = Path(FILE_TEMPLATE.format(idx="X")).stem.replace(" ", "_")
    combined_path = OUTPUT_DIR / f"hypoxie_normoxie_all_wells_{template_label}.pdf"
    combined_plot(hypoxie_data, normoxie_data, time_axis, combined_path)

    for idx in wells:
        out_path = OUTPUT_DIR / f"hypoxie_normoxie_highlight_well{idx}_{template_label}.pdf"
        highlight_plot(idx, hypoxie_data, normoxie_data, time_axis, out_path)

    print(f"Saved combined plot to: {combined_path}")
    print(f"Saved per-well highlight plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
