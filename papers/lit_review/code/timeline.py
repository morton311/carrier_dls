"""Timeline of dimensionality-reduction developments relevant to fluid dynamics.

Produces figs/timeline.pdf. Lanes are labelled, so color is a secondary cue.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

LANES = [
    ("Linear \\& modal", S.BLUE, [
        (1901, "PCA (Pearson)"), (1936, "Eckart--Young"), (1967, "POD (Lumley)"),
        (1987, "snapshot POD"), (2005, "balanced POD"), (2010, "DMD"),
        (2016, "OpInf"), (2018, "SPOD"),
    ]),
    ("Kernel \\& manifold\nlearning", S.ORANGE, [
        (1998, "kernel PCA"), (2000, "Isomap, LLE"), (2006, "diffusion maps"),
        (2017, "quadratic manifolds"), (2021, "kernel POD-NN"), (2023, "shear-flow manifolds"),
    ]),
    ("Sparse, local \\&\nmultiscale", S.AQUA, [
        (1996, "PUM / sparse coding"), (2002, "RB element"),
        (2012, "local ROBs"), (2014, "cluster ROM"), (2016, "sparse-coding ROM"),
        (2019, "mPOD"), (2020, "DLS"), (2021, "DD-LSPG"),
    ]),
    ("Neural\nautoencoders", S.VIOLET, [
        (1991, "NLPCA (Kramer)"), (2002, "NN-POD, wall turb."), (2006, "deep AE"),
        (2014, "VAE"), (2017, "$\\beta$-VAE"), (2020, "CAE manifold ROM"),
        (2022, "$\\beta$-VAE modes"), (2023, "NIF, CROM"), (2024, "graph AE"),
    ]),
    ("Latent dynamics\n\\& operators", S.MAGENTA, [
        (1997, "LSTM"), (2016, "SINDy"), (2017, "Transformer"), (2018, "Koopman AE"),
        (2019, "SINDy-AE"), (2021, "DeepONet, FNO"), (2022, "LaSDI, LED"),
        (2024, "$\\beta$-VAE + Transf."),
    ]),
]

SEGMENTS = [(1895, 0.0), (1985, 1.0), (2010, 2.5), (2026.5, 6.6)]


def tx(y):
    """Piecewise-linear time axis: early history compressed, 2010 onward expanded."""
    ys, xs = zip(*SEGMENTS)
    return np.interp(y, ys, xs)


def main():
    S.setup()
    fig, ax = plt.subplots(figsize=(S.TEXTWIDTH, 4.6))
    ax.grid(False)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    n = len(LANES)
    ax.set_xlim(-0.02, float(tx(2026.5)))
    ax.set_ylim(-0.55, n - 1 + 0.55)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    levels = [0.14, -0.14, 0.31, -0.31]
    for li, (name, color, events) in enumerate(LANES):
        y0 = n - 1 - li
        ax.axhspan(y0 - 0.47, y0 + 0.47, color=S.NEUTRAL, lw=0, zorder=0)
        ax.plot([0, tx(2026)], [y0, y0], color=S.GRID, lw=0.8, zorder=1)
        ax.text(-0.08, y0, name, ha="right", va="center", fontsize=7.5, color=S.INK)
        placed, stems = [], []          # label boxes (x0, x1, level); outer stems (x, sign)
        for yr, lab in sorted(events):
            x = float(tx(yr))
            for dy in levels:
                t = ax.text(x, y0 + dy, lab, ha="center", va="bottom" if dy > 0 else "top",
                            fontsize=6, color=S.INK, zorder=4)
                bb = t.get_window_extent(renderer).padded(4)
                (x0, _), (x1, _) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
                sign, outer = np.sign(dy), abs(dy) > 0.2
                clash = any(x0 < q1 and q0 < x1 for q0, q1, d in placed if d == dy)
                if outer:   # our stem must not cut an inner label on the same side
                    clash |= any(q0 < x < q1 for q0, q1, d in placed if d == sign * levels[0])
                else:       # our label must not cover an earlier outer stem
                    clash |= any(x0 < xs < x1 for xs, sg in stems if sg == sign)
                if not clash:
                    placed.append((x0, x1, dy))
                    if outer:
                        stems.append((x, sign))
                    break
                t.remove()
            else:
                raise RuntimeError(f"no free label slot for {lab!r}")
            ax.plot([x, x], [y0, y0 + dy * 0.9], color=color, lw=0.6, zorder=2)
            ax.plot(x, y0, "o", color=color, ms=4.2, mec="white", mew=0.7, zorder=3)
    ticks = [1900, 1950, 1985, 2000, 2010, 2015, 2020, 2025]
    ax.set_xticks(tx(ticks)); ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticks([])
    for yb in (1985, 2010):
        xb = float(tx(yb))
        ax.plot([xb, xb], [-0.55, n - 1 + 0.5], color=S.INK_2, lw=0.5, ls=(0, (2, 2)), zorder=0.5)
    ax.tick_params(axis="x", labelsize=7.5)
    S.save(fig, "timeline")


if __name__ == "__main__":
    main()
