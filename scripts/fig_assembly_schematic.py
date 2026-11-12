"""Schematic of how element predictions are combined in the element-token rollout.

(a) Geometry: each element predicts its own copy of every node it touches
    (exploded view); the mass-weighted projection returns one value per node.
(b) One rollout step as implemented: runner._predict_dofs_forward_spatial ->
    runner._reassemble_tokens, with the operators built once in runner._assembly_ops.

Everything is laid out in inches on one full-figure axes.

    python scripts/fig_assembly_schematic.py      # writes figures/assembly_schematic.pdf/.png
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

COLORS = ['#2a78d6', '#e8743b', '#1baf7a', '#8a5cf6']      # elements e1..e4
INK = '0.15'
MUTED = '0.40'
BOX_FC = '#f4f6fa'
BOX_EC = '0.60'
SOLVE_FC = '#e4f4ec'
TRAIN_FC = '#fdf1e6'
LOOP = '#2a78d6'
H = 5.1                                                       # figure height, inches

FS_TITLE, FS_BODY, FS_CODE, FS_NOTE = 7.6, 6.8, 5.6, 6.4


def box(ax, x, y, w, h, title, body, code=None, fc=BOX_FC):
    import matplotlib.patches as mp
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.05',
                                   fc=fc, ec=BOX_EC, lw=0.8))
    ax.text(x + 0.07, y + h - 0.07, title, ha='left', va='top', fontsize=FS_TITLE, color=INK, weight='bold')
    if code:
        ax.text(x + w - 0.06, y + h - 0.08, code, ha='right', va='top', fontsize=FS_CODE, color=MUTED)
    ax.text(x + 0.07, y + h - 0.23, body, ha='left', va='top', fontsize=FS_BODY, color=INK, linespacing=1.45)


def arrow(ax, p, q, color=INK, ls='-', rad=0.0, lw=1.0):
    ax.annotate('', xy=q, xytext=p,
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, ls=ls, mutation_scale=8,
                                connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def panel_geometry(ax):
    import matplotlib.patches as mp
    cx = 1.08                                                  # panel centre line
    ax.text(0.05, H - 0.08, r'(a) Shared nodes', fontsize=9, color=INK, va='top')

    # exploded view: four elements pulled apart; each holds its own copy of its corners
    s, g, cy = 0.46, 0.20, H - 1.32
    ax.text(cx, cy + s + g / 2 + 0.30, r'element predictions $\hat{\mathbf{d}}_e$', ha='center',
            fontsize=FS_BODY, color=INK)
    ax.text(cx, cy + s + g / 2 + 0.15, r'every element keeps its own copy (circled: 4 copies)',
            ha='center', fontsize=FS_NOTE, color=MUTED)
    lows = [(cx - g / 2 - s, cy + g / 2), (cx + g / 2, cy + g / 2),
            (cx - g / 2 - s, cy - g / 2 - s), (cx + g / 2, cy - g / 2 - s)]
    for k, (x0, y0) in enumerate(lows):
        ax.add_patch(mp.Rectangle((x0, y0), s, s, fc=COLORS[k], alpha=0.12, ec=COLORS[k], lw=1.0))
        ax.text(x0 + s / 2, y0 + s / 2, rf'$e_{k + 1}$', ha='center', va='center', fontsize=FS_BODY,
                color=COLORS[k])
        for dx in (0, s):
            for dy in (0, s):
                ax.plot(x0 + dx, y0 + dy, 'o', ms=3.6, color=COLORS[k], zorder=3)
    ax.add_patch(mp.Circle((cx, cy), 0.23, fc='none', ec=MUTED, lw=0.7, ls='--'))

    # projection arrow
    top, bot = cy - s - g / 2 - 0.12, cy - s - g / 2 - 0.62
    arrow(ax, (cx, top), (cx, bot), lw=1.3)
    ax.text(cx + 0.08, (top + bot) / 2 + 0.07, r'mass-weighted', fontsize=FS_NOTE, color=INK, va='center')
    ax.text(cx + 0.08, (top + bot) / 2 - 0.07, r'projection', fontsize=FS_NOTE, color=INK, va='center')

    # assembled mesh: one value per node
    a, cy2 = 0.40, bot - 0.12 - 0.40
    for k, (i, j) in enumerate([(0, 1), (1, 1), (0, 0), (1, 0)]):
        ax.add_patch(mp.Rectangle((cx - a + i * a, cy2 - a + j * a), a, a, fc=COLORS[k], alpha=0.12,
                                  ec='0.35', lw=0.8))
    for i in range(3):
        for j in range(3):
            ax.plot(cx - a + i * a, cy2 - a + j * a, 'o', ms=4.2, color=INK, zorder=3)
    ax.add_patch(mp.Circle((cx, cy2), 0.10, fc='none', ec=MUTED, lw=0.7, ls='--'))
    ax.text(cx, cy2 - a - 0.17, r'assembled coefficients $\mathbf{d}^{\star}$', ha='center',
            fontsize=FS_BODY, color=INK)
    ax.text(cx, cy2 - a - 0.32, r'one value per node, $C^0$ field', ha='center', fontsize=FS_NOTE, color=MUTED)

    # the two limiting metrics
    y = cy2 - a - 0.55
    lines = [r'steps 3--5 solve one least-squares problem:',
             r'$\mathbf{d}^{\star}=\arg\min_{\mathbf{d}}\sum_e\|\mathbf{G}_e\mathbf{d}-\hat{\mathbf{d}}_e\|^2_{\mathbf{M}_{e,\lambda}}$',
             r'$\lambda\to\infty$: $\mathbf{M}_{e,\lambda}\propto\mathbf{I}$, nodal average',
             r'$\lambda=0$: pure $L^2$ projection of the',
             r'\quad piecewise field $\{\mathbf{B}\hat{\mathbf{d}}_e\}$; unstable',
             r'$\lambda=10^{-2}$ (used): field metric + ridge',
             r'copies that already agree are unchanged']
    for k, t in enumerate(lines):
        ax.text(0.08, y - 0.165 * k, t, fontsize=FS_NOTE, color=INK, ha='left', va='top')


def panel_pipeline(ax, W):
    ax.text(2.42, H - 0.08, r'(b) One rollout step, as implemented', fontsize=9, color=INK, va='top')
    x, w, h, gap = 2.55, 2.25, 0.68, 0.11
    top = H - 0.32
    ys = [top - h - k * (h + gap) for k in range(6)]          # lower edges, top to bottom
    spec = [
        (r'1. Gather tokens', r'\texttt{lltogl}',
         r'$\mathbf{x}_e=[\mathbf{G}_e\mathbf{d}^{(t-\tau+1)},\dots,\mathbf{G}_e\mathbf{d}^{(t)}]$'
         '\n' r'one token per element; $\mathbf{G}_e$ picks its 4 nodes', BOX_FC),
        (r'2. Spatial transformer', r'\texttt{SpatialTransformer}',
         r'$\hat{\mathbf{d}}_e=\mathbf{G}_e\mathbf{d}^{(t)}+f_\theta(\mathbf{x}_1,\dots,\mathbf{x}_E)_e$'
         '\n' r'each element predicted on its own', BOX_FC),
        (r'3. Mass-matrix weighting', r'\texttt{\_reassemble\_tokens}',
         r'$\mathbf{r}_e=\mathbf{M}_{e,\lambda}\hat{\mathbf{d}}_e$'
         '\n' r'$\mathbf{M}_{e,\lambda}=\mathbf{B}^{\top}\mathbf{W}_{\!e}\mathbf{B}'
         r'+\lambda\,(\mathrm{tr}\,\mathbf{M}/n_e)\,\mathbf{I}$'
         '\n' r'$\mathbf{W}_{\!e}$: point weights, 0 outside the domain', BOX_FC),
        (r'4. Scatter-add to nodes', r'\texttt{index\_add}',
         r'$\mathbf{b}=\sum_e\mathbf{G}_e^{\top}\mathbf{r}_e$'
         '\n' r'every copy of a node contributes', BOX_FC),
        (r'5. Global solve', r'\texttt{\_assembly\_ops}',
         r'$\mathbf{d}^{(t+1)}=\mathbf{A}^{-1}\mathbf{b}$, \ '
         r'$\mathbf{A}=\sum_e\mathbf{G}_e^{\top}\mathbf{M}_{e,\lambda}\mathbf{G}_e$'
         '\n' r'$\mathbf{A}^{-1}$ built once; couples neighbouring nodes', SOLVE_FC),
        (r'6. Regather tokens', r'\texttt{\_reassemble\_tokens}',
         r'$\mathbf{d}_e^{(t+1)}=\mathbf{G}_e\mathbf{d}^{(t+1)}$, appended to the'
         '\n' r'input window (oldest frame dropped)', BOX_FC),
    ]
    for y, (title, code, body, fc) in zip(ys, spec):
        box(ax, x, y, w, h, title, body, code, fc)
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, (x + w / 2, a), (x + w / 2, b + h))

    # feedback loop 6 -> 1
    lx = x - 0.13
    ax.plot([x, lx, lx], [ys[5] + h / 2, ys[5] + h / 2, ys[0] + h / 2], color=LOOP, lw=1.0)
    arrow(ax, (lx, ys[0] + h / 2), (x, ys[0] + h / 2), color=LOOP)
    ax.text(lx - 0.06, (ys[0] + ys[5]) / 2 + h / 2, r'next step $t+1$', rotation=90, fontsize=FS_NOTE,
            color=LOOP, ha='center', va='center')

    # side column: training (steps 3-6 inside the rollout) and the decoded output
    sx = x + w + 0.18
    sw = W - sx - 0.04
    ty, th = ys[4], ys[3] + h - ys[4]
    box(ax, sx, ty, sw, th, r'Training',
        r'\texttt{rollout.reassemble}:'
        '\n' r'steps 3--6 run inside the'
        '\n' r'16-step training rollout'
        '\n' r'loss on regathered tokens,'
        '\n' r'$\sum_e\|\mathbf{R}_\lambda(\mathbf{d}_e-\mathbf{d}_e^{\mathrm{true}})\|^2$'
        '\n' r'gradients pass through $\mathbf{A}^{-1}$', fc=TRAIN_FC)
    box(ax, sx, ys[5], sw, h, r'Output field',
        r'$\mathbf{u}_e=\mathbf{B}\,\mathbf{d}^{(t+1)}_e$'
        '\n' r'per element (decode)')
    arrow(ax, (x + w, ys[5] + h * 0.40), (sx, ys[5] + h * 0.40))
    arrow(ax, (x + w, ys[5] + h * 0.80), (sx + 0.35, ty), color='#c2601f', ls='--')


def main():
    import matplotlib.pyplot as plt
    import lib.plotting as pl
    plt.rcParams['figure.constrained_layout.use'] = False
    W = pl.width
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')
    panel_geometry(ax)
    panel_pipeline(ax, W)
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/assembly_schematic.pdf')
    fig.savefig('figures/assembly_schematic.png', dpi=220)
    print('wrote figures/assembly_schematic.pdf/.png')


if __name__ == '__main__':
    main()
