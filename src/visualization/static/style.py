"""Shared visualization style: colors, fonts, and figure defaults."""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Colorblind-friendly palette for conditions
CONDITION_COLORS = {
    "normal": "#4477AA",       # Blue
    "no_image": "#EE6677",     # Red/coral
    "wrong_image": "#CCBB44",  # Yellow/olive
    "gaussian_noise": "#AA3377",  # Purple
    "heavy_blur": "#66CCEE",   # Cyan
    "shuffled_patches": "#228833",  # Green
}

# Benchmark display names
BENCHMARK_NAMES = {
    "mmmu": "MMMU",
    "mmbench": "MMBench",
    "scienceqa": "ScienceQA",
    "pope": "POPE",
    "textvqa": "TextVQA",
    "gqa": "GQA",
}

# Condition display names
CONDITION_NAMES = {
    "normal": "Normal",
    "no_image": "No Image",
    "wrong_image": "Wrong Image",
    "gaussian_noise": "Noise",
    "heavy_blur": "Blur",
    "shuffled_patches": "Shuffled",
}

# Random chance baselines
RANDOM_CHANCE = {
    "mmmu": 0.25,
    "mmbench": 0.25,
    "scienceqa": 0.25,
    "pope": 0.50,
    "textvqa": 0.0,
    "gqa": 0.0,
}

# Benchmark ordering: high language prior first, then low
BENCHMARK_ORDER = ["mmmu", "mmbench", "scienceqa", "pope", "textvqa", "gqa"]

# High-level category
HIGH_PRIOR = {"mmmu", "mmbench", "scienceqa"}
LOW_PRIOR = {"pope", "textvqa", "gqa"}


def apply_style():
    """Apply consistent matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def save_figure(fig, output_dir: str, name: str):
    """Save figure as both PNG and PDF."""
    import os
    fig.savefig(os.path.join(output_dir, f"{name}.png"))
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"))
    plt.close(fig)
