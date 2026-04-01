import os, glob, argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


def setup_style():
    plt.rcParams.update({'grid.linewidth': 0.3,
                         'text.usetex': True})


def histogram_to_pdf_from_samples(H, bin_edges):
    H = np.asarray(H, dtype=float)
    H = H[np.isfinite(H)]
    counts, _ = np.histogram(H, bins=bin_edges)
    widths = np.diff(bin_edges)
    total = counts.sum()

    if total <= 0:
        pdf = np.full(len(widths), np.nan, dtype=float)
    else:
        pdf = counts / total / widths

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, pdf


def discover_file(summary_dir, tracer, zone):
    summary_dir = Path(summary_dir)
    tracer = tracer.lower()
    zone = zone.lower()

    patterns = [str(summary_dir / f'{tracer}_{zone}_entropy_from_r_classification.npz'),
                str(summary_dir / f'{tracer}_{zone}_entropy_from_classification.npz'),
                str(summary_dir / f'{tracer}_*{zone}*_entropy_from_r_classification.npz'),
                str(summary_dir / f'{tracer}_*{zone}*_entropy_from_classification.npz')]

    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))

    matches = sorted(set(matches))
    return matches[0] if matches else None


def load_H_obj(path):
    d = np.load(path, allow_pickle=True)
    H = d['H_obj']
    H = H[np.isfinite(H)]
    return H


def plot_joint(summary_dir, outdir, bins=35, xmin=0.0, xmax=0.56):
    colors = {'BGS': 'crimson',
              'LRG': 'green',
              'ELG': 'darkorange',
              'QSO': 'deepskyblue'}

    tracers = ['BGS', 'LRG', 'ELG', 'QSO']

    bin_edges = np.linspace(xmin, xmax, bins + 1)

    fig, ax = plt.subplots()
    ax.grid(lw=0.3)

    plotted = False

    for tracer in tracers:
        path_sgc = discover_file(summary_dir, tracer, 'SGC')
        path_ngc = discover_file(summary_dir, tracer, 'NGC')

        if path_sgc is None and path_ngc is None:
            continue

        color = colors[tracer]

        if path_sgc is not None and path_ngc is not None:
            H_sgc = load_H_obj(path_sgc)
            H_ngc = load_H_obj(path_ngc)

            x_sgc, pdf_sgc = histogram_to_pdf_from_samples(H_sgc, bin_edges)
            x_ngc, pdf_ngc = histogram_to_pdf_from_samples(H_ngc, bin_edges)

            x = x_sgc
            pdf_low = np.minimum(pdf_sgc, pdf_ngc)
            pdf_high = np.maximum(pdf_sgc, pdf_ngc)
            pdf_med = 0.5 * (pdf_sgc + pdf_ngc)

            ax.fill_between(x, pdf_low, pdf_high,
                            color=color, alpha=0.4, zorder=2,
                            label=rf'{tracer} $\pm 1\sigma$')

            ax.plot(x, pdf_med, color=color, lw=1.6, zorder=4,
                    label=rf'{tracer} median')
        else:
            path = path_sgc if path_sgc is not None else path_ngc
            H = load_H_obj(path)
            x, pdf = histogram_to_pdf_from_samples(H, bin_edges)

            ax.plot(x, pdf, color=color, lw=2.4, zorder=4,
                    label=rf'{tracer} median')

            region = 'SGC' if path_sgc is not None else 'NGC'

        plotted = True

    if not plotted:
        raise RuntimeError()

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'$H$')
    ax.set_ylabel(r'PDF')

    leg = ax.legend(# loc='lower center',
                    # bbox_to_anchor=(0.5, -0.12),
                    ncol=1,
                    # frameon=True,
                    # fancybox=True
                    )

    fig.tight_layout()
    outpath = Path(outdir) / 'joint_entropy_pdf_style.png'
    fig.savefig(outpath, dpi=360, bbox_inches='tight')
    plt.close(fig)

    print(f'{outpath}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-dir', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--bins', type=int, default=35)
    parser.add_argument('--xmin', type=float, default=0.0)
    parser.add_argument('--xmax', type=float, default=1.0)
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    plot_joint(summary_dir=args.summary_dir,
               outdir=args.outdir,
               bins=args.bins,
               xmin=args.xmin,
               xmax=args.xmax)


if __name__ == '__main__':
    main()