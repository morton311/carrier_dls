"""Schematic of the element-token forecasting model.

One forecast step: the coefficient history is gathered into one token per element, the spatial transformer predicts
every element's update, and the mass-weighted projection returns one coefficient vector that is fed back.

Everything is laid out in inches on one full-figure axes.

    python scripts/fig_model_schematic.py      # writes figures/model_schematic.pdf/.png
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

INK, MUTED = '0.15', '0.42'
BLUE = '#2a78d6'
BOX_FC, BOX_EC = '#f4f6fa', '0.60'
MODEL_FC, PROJ_FC = '#eaf1fb', '#e4f4ec'
H = 2.25                                                      # figure height, inches
FS_TITLE, FS_BODY, FS_NOTE = 7.4, 6.6, 6.2


def rbox(ax, x, y, w, h, fc=BOX_FC, ec=BOX_EC, lw=0.8, z=1):
    import matplotlib.patches as mp
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.05', fc=fc, ec=ec, lw=lw,
                                   zorder=z))


def arrow(ax, p, q, color=INK, ls='-', lw=1.0, rad=0.0):
    ax.annotate('', xy=q, xytext=p,
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, ls=ls, mutation_scale=8,
                                connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def mesh(ax, x, y, s, n=4, fc='white', ec='0.45', hi=None, z=2):
    """An n x n element mesh of side s with its lower-left corner at (x, y); hi = (i, j) element to highlight."""
    import matplotlib.patches as mp
    ax.add_patch(mp.Rectangle((x, y), s, s, fc=fc, ec=ec, lw=0.7, zorder=z))
    a = s / n
    for k in range(1, n):
        ax.plot([x + k * a, x + k * a], [y, y + s], color=ec, lw=0.4, zorder=z)
        ax.plot([x, x + s], [y + k * a, y + k * a], color=ec, lw=0.4, zorder=z)
    if hi is not None:
        i, j = hi
        ax.add_patch(mp.Rectangle((x + i * a, y + j * a), a, a, fc=BLUE, alpha=0.30, ec=BLUE, lw=1.0, zorder=z))
        for dx in (0, a):
            for dy in (0, a):
                ax.plot(x + i * a + dx, y + j * a + dy, 'o', ms=2.2, color=BLUE, zorder=z + 1)


def tokens(ax, x, yc, w, n=6, hi=2, hat=False):
    """A column of n element tokens centred on yc, with an ellipsis; returns (bottom, top)."""
    import matplotlib.patches as mp
    h, g = 0.15, 0.045
    tot = n * h + (n - 1) * g + 0.12                          # extra room for the dots
    y = yc + tot / 2
    for k in range(n):
        y -= h
        if k == n - 1:                                        # dots before the last token
            ax.text(x + w / 2, y + h + 0.012, r'$\vdots$', ha='center', va='center', fontsize=6, color=MUTED)
            y -= 0.12
        on = k == hi
        ax.add_patch(mp.Rectangle((x, y), w, h, fc=BLUE if on else 'white', alpha=1.0, ec=BLUE if on else '0.45',
                                  lw=0.7, zorder=2))
        if on:
            ax.add_patch(mp.Rectangle((x, y), w, h, fc='white', alpha=0.65, ec='none', zorder=2))
            ax.add_patch(mp.Rectangle((x, y), w, h, fc='none', ec=BLUE, lw=1.0, zorder=3))
        y -= g
    return yc - tot / 2, yc + tot / 2


def panel_step(ax, W):
    yc = 1.27                                                 # centre line of the pipeline

    # 1. coefficient history: tau stacked meshes
    s, off, x0 = 0.70, 0.065, 0.06
    for k in range(3, -1, -1):
        mesh(ax, x0 + k * off, yc - s / 2 - 0.10 + k * off, s, hi=(1, 2) if k == 0 else None, z=2 + (3 - k) * 2,
             fc='white' if k == 0 else '0.95')
    ax.text(x0 + (s + 3 * off) / 2, yc + s / 2 + 0.28, 'coefficient history', ha='center', fontsize=FS_TITLE,
            color=INK)
    ax.text(x0 + (s + 3 * off) / 2, yc - s / 2 - 0.26, r'$\mathbf{d}^{(t-\tau+1)},\dots,\mathbf{d}^{(t)}$',
            ha='center', fontsize=FS_BODY, color=INK)
    ax.text(x0 + (s + 3 * off) / 2, yc - s / 2 - 0.42, r'$\tau=4$ snapshots', ha='center', fontsize=FS_NOTE,
            color=MUTED)

    # 2. element tokens
    tx, tw = 1.16, 0.24
    arrow(ax, (x0 + s + 3 * off + 0.03, yc), (tx - 0.03, yc))
    tokens(ax, tx, yc, tw)
    ax.text(tx + tw / 2, yc - 0.76, 'element', ha='center', fontsize=FS_TITLE, color=INK)
    ax.text(tx + tw / 2, yc - 0.90, r'tokens $\mathbf{x}_e$', ha='center', fontsize=FS_TITLE, color=INK)

    # 3. spatial transformer
    mx, mw, mh = 1.58, 2.80, 1.22
    my = yc - mh / 2
    rbox(ax, mx, my, mw, mh, fc=MODEL_FC, ec=BLUE, lw=0.9)
    ax.text(mx + 0.08, my + mh - 0.07, r'Spatial transformer $f_\theta$', ha='left', va='top', fontsize=FS_TITLE,
            color=INK, weight='bold')
    ax.text(mx + mw - 0.08, my + mh - 0.08, 'shared weights for all elements', ha='right', va='top',
            fontsize=FS_NOTE, color=MUTED)
    arrow(ax, (tx + tw + 0.03, yc), (mx, yc))
    sy, sh = my + 0.10, 0.80
    sc = sy + sh / 2
    ex, ew = mx + 0.10, 0.72                                  # embedding
    lx, lw_ = ex + ew + 0.12, 0.98                            # encoder layers
    hx, hw = lx + lw_ + 0.20, 0.54                            # head
    rbox(ax, ex, sy, ew, sh, fc='white')
    for k, t in enumerate(['linear', 'embedding', r'$+$ Fourier', 'features of', 'centroid']):
        ax.text(ex + ew / 2, sy + sh - 0.13 - 0.135 * k, t, ha='center', va='center', fontsize=FS_BODY, color=INK)
    for k in (2, 1):                                          # shadow copies: four layers
        rbox(ax, lx + 0.035 * k, sy + 0.035 * k, lw_, sh, fc='white', z=1)
    rbox(ax, lx, sy, lw_, sh, fc='white', z=2)
    ax.text(lx + lw_ / 2, sy + sh - 0.12, r'encoder layer $\times 4$', ha='center', va='center', fontsize=FS_BODY,
            color=INK, weight='bold', zorder=3)
    rbox(ax, lx + 0.07, sy + 0.30, lw_ - 0.14, 0.27, fc=BOX_FC, z=3)
    ax.text(lx + lw_ / 2, sy + 0.495, 'attention across', ha='center', va='center', fontsize=FS_NOTE, color=INK,
            zorder=4)
    ax.text(lx + lw_ / 2, sy + 0.375, 'all elements', ha='center', va='center', fontsize=FS_NOTE, color=INK,
            zorder=4)
    rbox(ax, lx + 0.07, sy + 0.06, lw_ - 0.14, 0.18, fc=BOX_FC, z=3)
    ax.text(lx + lw_ / 2, sy + 0.15, 'SwiGLU feed-forward', ha='center', va='center', fontsize=FS_NOTE - 0.4,
            color=INK, zorder=4)
    rbox(ax, hx, sy, hw, sh, fc='white')
    for k, t in enumerate(['linear', 'head', '(zero', 'init.)']):
        ax.text(hx + hw / 2, sy + sh - 0.17 - 0.15 * k, t, ha='center', va='center', fontsize=FS_BODY, color=INK)
    arrow(ax, (ex + ew, sc), (lx, sc))
    arrow(ax, (lx + lw_ + 0.07, sc), (hx, sc))

    # residual connection around the model
    px = mx + mw + 0.15
    import matplotlib.patches as mp
    ry = my + mh + 0.16
    ax.plot([tx + tw + 0.10, tx + tw + 0.10, px], [yc, ry, ry], color=MUTED, lw=0.9, zorder=0)
    arrow(ax, (px, ry), (px, yc + 0.075), color=MUTED, lw=0.9)
    ax.text((mx + px) / 2, ry + 0.07, r'residual: last frame $\mathbf{G}_e\mathbf{d}^{(t)}$', ha='center',
            fontsize=FS_NOTE, color=MUTED)
    ax.plot([mx + mw, px - 0.075], [yc, yc], color=INK, lw=1.0)
    ax.add_patch(mp.Circle((px, yc), 0.075, fc='white', ec=INK, lw=0.8, zorder=3))
    ax.text(px, yc, r'$+$', ha='center', va='center', fontsize=7, color=INK, zorder=4)

    # 4. element predictions
    ox, ow = px + 0.19, 0.24
    arrow(ax, (px + 0.075, yc), (ox - 0.03, yc))
    tokens(ax, ox, yc, ow)
    ax.text(ox + ow / 2, yc - 0.76, 'element', ha='center', fontsize=FS_TITLE, color=INK)
    ax.text(ox + ow / 2, yc - 0.90, r'predictions $\hat{\mathbf{d}}_e$', ha='center', fontsize=FS_TITLE,
            color=INK)

    # 5. mass-weighted projection and the next state
    qx, qw, qh = ox + ow + 0.17, 0.62, 0.62
    arrow(ax, (ox + ow + 0.03, yc), (qx, yc))
    rbox(ax, qx, yc - qh / 2, qw, qh, fc=PROJ_FC)
    for k, t in enumerate(['mass-', 'weighted', 'projection']):
        ax.text(qx + qw / 2, yc + 0.17 - 0.15 * k, t, ha='center', va='center', fontsize=FS_BODY, color=INK)
    s2 = 0.50
    nx = W - s2 - 0.06
    arrow(ax, (qx + qw, yc), (nx - 0.03, yc))
    mesh(ax, nx, yc - s2 / 2, s2, hi=(1, 2))
    ax.text(nx + s2 / 2, yc + s2 / 2 + 0.26, 'next state', ha='center', fontsize=FS_TITLE, color=INK)
    ax.text(nx + s2 / 2, yc + s2 / 2 + 0.10, r'$\mathbf{d}^{(t+1)}$', ha='center', fontsize=FS_BODY, color=INK)
    ax.text(nx + s2 / 2, yc - s2 / 2 - 0.15, 'one value', ha='center', fontsize=FS_NOTE, color=MUTED)
    ax.text(nx + s2 / 2, yc - s2 / 2 - 0.28, 'per node', ha='center', fontsize=FS_NOTE, color=MUTED)

    # feedback
    fy = my - 0.56
    fx0, fx1 = nx + s2 / 2, x0 + 0.20
    ax.plot([fx0, fx0, fx1], [yc - s2 / 2 - 0.36, fy, fy], color=BLUE, lw=1.0)
    arrow(ax, (fx1, fy), (fx1, yc - s / 2 - 0.50), color=BLUE)
    ax.text((mx + mx + mw) / 2, fy + 0.07, 'appended to the history; the oldest snapshot is dropped',
            ha='center', fontsize=FS_NOTE, color=BLUE)


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
    panel_step(ax, W)
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/model_schematic.pdf')
    fig.savefig('figures/model_schematic.png', dpi=220)
    print('wrote figures/model_schematic.pdf/.png')


if __name__ == '__main__':
    main()
