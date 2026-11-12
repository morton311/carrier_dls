"""Shared matplotlib style for the literature-review figures."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
FIG_DIR.mkdir(exist_ok=True)

# Text width of the review (article, 1in margins) in inches.
TEXTWIDTH = 6.5

# Categorical slots (fixed order; first three are validated all-pairs).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# Single-hue ordinal ramp (blue 250 -> 700).
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"
NEUTRAL = "#f0efec"

# Diverging map for signed fields: blue <-> neutral gray <-> red.
DIVERGING = LinearSegmentedColormap.from_list(
    "bgr", ["#0d366b", BLUE, "#9ec5f4", NEUTRAL, "#f4b4b3", RED, "#8f1f1e"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#fcfcfb"] + BLUE_RAMP)


def setup():
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    })


def save(fig, name):
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved {path}")


def panel_label(ax, label, x=-0.02, y=1.02):
    ax.text(x, y, rf"\textbf{{({label})}}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=INK)
