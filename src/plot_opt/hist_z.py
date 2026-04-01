import os, re, glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
plt.style.use('dark_background')


def setup_style():
    plt.rcParams.update({'text.usetex': True})


def safe_upper(x):
    return str(x).strip().upper()


def tracer_aliases(tracer):
    t = safe_upper(tracer)
    mapping = {'BGS': ('BGS', 'BGS_ANY', 'BGS_BRIGHT'),
               'BGS_ANY': ('BGS_ANY', 'BGS', 'BGS_BRIGHT'),
               'BGS_BRIGHT': ('BGS_BRIGHT', 'BGS', 'BGS_ANY'),
               'ELG': ('ELG', 'ELG_LOPNOTQSO', 'ELG_LOPnotqso'),
               'ELG_LOPNOTQSO': ('ELG_LOPNOTQSO', 'ELG', 'ELG_LOPnotqso'),
               'LRG': ('LRG'),
               'QSO': ('QSO')}
    return mapping.get(t, (t,))


def get_columns(path):
    with fits.open(path, memmap=True) as hdul:
        return list(hdul[1].columns.names)


def find_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def iter_fits_chunks(path, columns, chunk_rows=500_000):
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[1]
        data = hdu.data
        if data is None:
            return

        nrows = int(hdu.header.get('NAXIS2', 0))
        if nrows == 0:
            return

        use_cols = [c for c in columns if c in hdu.columns.names]

        for start in range(0, nrows, chunk_rows):
            stop = min(start + chunk_rows, nrows)
            block = data[start:stop]
            yield {col: np.asarray(block[col]) for col in use_cols}


def discover_files(base, tracer, zone):
    base = Path(base)
    tracer_dir = tracer.lower()
    zone_dir = zone.lower()
    zone_up = safe_upper(zone)

    aliases_up = [safe_upper(a) for a in tracer_aliases(tracer)]
    class_root = base / 'classification' / tracer_dir / zone_dir

    files = []
    for a in aliases_up:
        patterns = [str(class_root / f'zone_{zone_up}_{a}_iter*.fits.gz'),
                    str(class_root / f'zone_{zone_up}_{a}_iter*.fits')]
        for pat in patterns:
            files.extend(glob.glob(pat))

    return sorted(set(files))


def load_redshift_for_tracer(base, tracer, zones, chunk_rows=500_000):
    base = Path(base)
    z_all = []

    aliases_up = [safe_upper(a) for a in tracer_aliases(tracer)]

    for zone in zones:
        zone_up = safe_upper(zone)

        raw_files = []
        for a in aliases_up:
            patterns = [str(base / 'raw' / f'zone_{zone_up}_{a}.fits.gz'),
                        str(base / 'raw' / f'zone_{zone_up}_{a}.fits')]
            for pat in patterns:
                raw_files.extend(glob.glob(pat))

        raw_files = sorted(set(raw_files))

        if len(raw_files) == 0:
            continue

        found_any = False

        for raw_path in raw_files:
            cols = get_columns(raw_path)
            z_col = find_col(cols, ('Z', 'z'))
            isdata_col = find_col(cols, ('ISDATA', 'isdata'))
            randiter_col = find_col(cols, ('RANDITER', 'randiter'))

            if z_col is None:
                continue

            wanted = [z_col]
            for c in (isdata_col, randiter_col):
                if c is not None:
                    wanted.append(c)

            chunks = []
            for chunk in iter_fits_chunks(raw_path, wanted, chunk_rows=chunk_rows):
                z = np.asarray(chunk[z_col], dtype=np.float32)

                mask = np.isfinite(z)

                if isdata_col is not None:
                    mask &= np.asarray(chunk[isdata_col]).astype(bool)
                elif randiter_col is not None:
                    mask &= (np.asarray(chunk[randiter_col]) == -1)

                if np.any(mask):
                    chunks.append(z[mask])

            if chunks:
                z_all.append(np.concatenate(chunks))
                found_any = True

    return np.concatenate(z_all) if z_all else np.array([], dtype=np.float32)


def plot_histogram(base, zones, outdir, bins=30, zmin=0.0, zmax=3.5, chunk_rows=500_000):
    colors = {'BGS': 'crimson',
              'LRG': 'green',
              'ELG': 'darkorange',
              'QSO': 'deepskyblue'}

    tracers = ['BGS', 'LRG', 'ELG', 'QSO']

    fig, ax = plt.subplots()
    ax.grid(lw=0.3, alpha=0.5)

    bin_edges = np.linspace(zmin, zmax, bins + 1)
    widths = np.diff(bin_edges)

    plotted = False

    for tracer in tracers:
        z = load_redshift_for_tracer(base, tracer, zones, chunk_rows=chunk_rows)

        if z.size == 0:
            continue

        counts, _ = np.histogram(z, bins=bin_edges)
        y = counts / widths

        ax.bar(bin_edges[:-1], y, width=widths, align='edge',
               color=colors[tracer], alpha=0.85,
               edgecolor=colors[tracer], linewidth=0.3,
               label=rf'{tracer} object')

        plotted = True

    if not plotted:
        raise RuntimeError()
    ax.set_xlim(zmin, zmax)
    ax.set_xlabel(r'$Z$')
    ax.set_ylabel(r'$N_{\mathrm{Gal}}/\Delta Z$')
    # ax.set_yscale('log')

    leg = ax.legend(loc='upper right')
    fig.tight_layout()

    outpath = Path(outdir) / 'redshift_distribution_by_tracer.png'
    fig.savefig(outpath, dpi=360, bbox_inches='tight')
    plt.close(fig)
    print(f'\n------------ {outpath}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--zones', nargs='+', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--bins', type=int, default=30)
    parser.add_argument('--zmin', type=float, default=0.0)
    parser.add_argument('--zmax', type=float, default=3.5)
    parser.add_argument('--chunk-rows', type=int, default=500_000)
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    plot_histogram(base=args.base, zones=args.zones,
                   outdir=args.outdir, bins=args.bins,
                   zmin=args.zmin, zmax=args.zmax,
                   chunk_rows=args.chunk_rows)


if __name__ == '__main__':
    main()