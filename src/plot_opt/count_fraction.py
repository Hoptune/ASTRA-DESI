import re, glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def safe_upper(x):
    return str(x).strip().upper()


def tracer_aliases(tracer):
    t = safe_upper(tracer)
    mapping = {'BGS': ('BGS', 'BGS_ANY', 'BGS_BRIGHT'),
               'BGS_ANY': ('BGS_ANY', 'BGS', 'BGS_BRIGHT'),
               'BGS_BRIGHT': ('BGS_BRIGHT', 'BGS', 'BGS_ANY'),
               'ELG': ('ELG', 'ELG_LOPNOTQSO', 'ELG_LOPnotqso'),
               'ELG_LOPNOTQSO': ('ELG_LOPNOTQSO', 'ELG', 'ELG_LOPnotqso'),
               'LRG': ('LRG',),
               'QSO': ('QSO',)}
    return mapping.get(t, (t,))


def parse_iter(path):
    m = re.search(r'iter(\d+)', str(path))
    return int(m.group(1)) if m else -1


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


def r_from_counts(ndata, nrand):
    ndata = np.asarray(ndata, dtype=np.float32)
    nrand = np.asarray(nrand, dtype=np.float32)
    denom = ndata + nrand
    r = np.full_like(denom, np.nan, dtype=np.float32)
    valid = np.isfinite(denom) & (denom > 0)
    r[valid] = (ndata[valid] - nrand[valid]) / denom[valid]
    return r


def classify_from_r(r):
    out = np.full(r.shape, -1, dtype=np.int8)

    out[np.isfinite(r) & (r >= -1.0) & (r <= -0.25)] = 0   # Void
    out[np.isfinite(r) & (r > -0.25) & (r <= 0.25)] = 1    # Sheet
    out[np.isfinite(r) & (r > 0.25) & (r <= 0.65)] = 2     # Filament
    out[np.isfinite(r) & (r > 0.65) & (r <= 1.0)] = 3      # Knot

    return out


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

    files = sorted(set(files), key=parse_iter)
    return [(parse_iter(f), f) for f in files if parse_iter(f) >= 0]


def one_iteration_fractions(path, chunk_rows=500_000):
    cols = get_columns(path)

    ndata_col = find_col(cols, ('NDATA', 'ndata'))
    nrand_col = find_col(cols, ('NRAND', 'nrand'))
    isdata_col = find_col(cols, ('ISDATA', 'isdata'))

    if ndata_col is None or nrand_col is None or isdata_col is None:
        raise ValueError()

    wanted = [ndata_col, nrand_col, isdata_col]

    counts_obj = np.zeros(4, dtype=np.int64)
    counts_rand = np.zeros(4, dtype=np.int64)

    for chunk in iter_fits_chunks(path, wanted, chunk_rows=chunk_rows):
        ndata = np.asarray(chunk[ndata_col], dtype=np.float32)
        nrand = np.asarray(chunk[nrand_col], dtype=np.float32)
        isdata = np.asarray(chunk[isdata_col]).astype(bool)

        r = r_from_counts(ndata, nrand)
        env = classify_from_r(r)

        valid = env >= 0
        if np.any(valid & isdata):
            counts_obj += np.bincount(env[valid & isdata], minlength=4).astype(np.int64)
        if np.any(valid & (~isdata)):
            counts_rand += np.bincount(env[valid & (~isdata)], minlength=4).astype(np.int64)

    frac_obj = counts_obj / counts_obj.sum() if counts_obj.sum() > 0 else np.full(4, np.nan)
    frac_rand = counts_rand / counts_rand.sum() if counts_rand.sum() > 0 else np.full(4, np.nan)

    return frac_obj, frac_rand


def zone_mean_fractions(base, tracer, zone, chunk_rows=500_000, iter_min=None, iter_max=None):
    files = discover_files(base, tracer, zone)

    if iter_min is not None:
        files = [(it, p) for it, p in files if it >= iter_min]
    if iter_max is not None:
        files = [(it, p) for it, p in files if it <= iter_max]

    if len(files) == 0:
        return None

    obj_list = []
    rand_list = []

    for it, path in files:
        frac_obj, frac_rand = one_iteration_fractions(path, chunk_rows=chunk_rows)
        obj_list.append(frac_obj)
        rand_list.append(frac_rand)

    return {'zone': safe_upper(zone),
            'n_iter': len(files),
            'object_mean': np.nanmean(np.vstack(obj_list), axis=0),
            'random_mean': np.nanmean(np.vstack(rand_list), axis=0)}


def build_count_fraction_table(base, zones, tracers, chunk_rows=500_000, iter_min=None, iter_max=None):
    rows = []

    for tracer in tracers:
        zone_results = []
        for zone in zones:
            zres = zone_mean_fractions(base=base,
                                       tracer=tracer,
                                       zone=zone,
                                       chunk_rows=chunk_rows,
                                       iter_min=iter_min,
                                       iter_max=iter_max)
            if zres is not None:
                zone_results.append(zres)

        if len(zone_results) == 0:
            continue

        obj_zone = np.vstack([z['object_mean'] for z in zone_results])
        rand_zone = np.vstack([z['random_mean'] for z in zone_results])

        obj_mean = np.nanmean(obj_zone, axis=0)
        rand_mean = np.nanmean(rand_zone, axis=0)

        obj_std = np.nanstd(obj_zone, axis=0, ddof=1) if obj_zone.shape[0] > 1 else np.zeros(4)
        rand_std = np.nanstd(rand_zone, axis=0, ddof=1) if rand_zone.shape[0] > 1 else np.zeros(4)

        rows.append({'Catalog': 'Object',
                     'Tracer': tracer,
                     'Void': f'{100*obj_mean[0]:.2f} ± {100*obj_std[0]:.2f}',
                     'Sheet': f'{100*obj_mean[1]:.2f} ± {100*obj_std[1]:.2f}',
                     'Filament': f'{100*obj_mean[2]:.2f} ± {100*obj_std[2]:.2f}',
                     'Knot': f'{100*obj_mean[3]:.2f} ± {100*obj_std[3]:.2f}'})

        rows.append({'Catalog': 'Random',
                     'Tracer': tracer,
                     'Void': f'{100*rand_mean[0]:.2f} ± {100*rand_std[0]:.2f}',
                     'Sheet': f'{100*rand_mean[1]:.2f} ± {100*rand_std[1]:.2f}',
                     'Filament': f'{100*rand_mean[2]:.2f} ± {100*rand_std[2]:.2f}',
                     'Knot': f'{100*rand_mean[3]:.2f} ± {100*rand_std[3]:.2f}'})

    return pd.DataFrame(rows, columns=['Catalog', 'Tracer', 'Void', 'Sheet', 'Filament', 'Knot'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--zones', nargs='+', required=True)
    parser.add_argument('--tracers', nargs='+', default=['BGS', 'LRG', 'ELG', 'QSO'])
    parser.add_argument('--chunk-rows', type=int, default=500_000)
    parser.add_argument('--iter-min', type=int, default=None)
    parser.add_argument('--iter-max', type=int, default=None)
    args = parser.parse_args()

    df = build_count_fraction_table(base=args.base,
                                    zones=args.zones,
                                    tracers=args.tracers,
                                    chunk_rows=args.chunk_rows,
                                    iter_min=args.iter_min,
                                    iter_max=args.iter_max,)

    print('')
    print(df.to_string(index=False))
    print('')


if __name__ == '__main__':
    main()