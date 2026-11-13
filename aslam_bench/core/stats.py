"""Análisis estadístico para comparar métodos de exploración.

Diseño pareado: cada (modo de localización, mapa, pose inicial, repetición) es un
"bloque" en el que se ejecutan todos los métodos con las mismas condiciones.

  * Resumen: mediana, IQR, media, desviación típica, n.
  * Pares de métodos: Wilcoxon de rangos con signo (pareado), corrección de Holm
    sobre todas las comparaciones de la familia, tamaño del efecto Cliff's δ.
  * Éxito: proporción con IC de Wilson; McNemar exacto (pareado).
  * ≥ 3 métodos: Friedman sobre los bloques + rangos medios (Nemenyi CD).
  * Tiempos censurados (no se alcanza el nivel de cobertura): se imputan como
    t_max (conservador para el método que falla) y se informa la fracción censurada;
    además se ofrece la curva de Kaplan-Meier del tiempo hasta el 95 %.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

BLOCK = ['loc_mode', 'map', 'start_id', 'rep']

# métrica -> (etiqueta, mejor = 'lower' | 'higher')
METRICS = {
    'T90': ('Tiempo hasta 90 % (s)', 'lower'),
    'T95': ('Tiempo hasta 95 % (s)', 'lower'),
    'D90': ('Distancia hasta 90 % (m)', 'lower'),
    'D95': ('Distancia hasta 95 % (m)', 'lower'),
    'auc': ('AUC cobertura-tiempo', 'higher'),
    'coverage_final': ('Cobertura final', 'higher'),
    'path_length': ('Longitud de trayectoria (m)', 'lower'),
    'rotation_total': ('Rotación total (rad)', 'lower'),
    'efficiency_m2_per_m': ('Eficiencia (m²/m)', 'higher'),
    'occ_recall': ('Recall de ocupadas', 'higher'),
    'occ_precision': ('Precisión de ocupadas', 'higher'),
    'free_iou': ('IoU espacio libre', 'higher'),
    'adnn': ('ADNN (m)', 'lower'),
    'ate_rmse': ('ATE RMSE (m)', 'lower'),
    'decision_time_mean': ('Tiempo de decisión medio (s)', 'lower'),
    'n_collisions': ('Colisiones', 'lower'),
}


def impute_censored(df, t_max_col='t_max', default_t_max=900.0):
    """Sustituye T/D censurados: T -> t_max; D -> longitud total recorrida."""
    df = df.copy()
    t_max = df[t_max_col] if t_max_col in df else default_t_max
    for c in [c for c in df.columns if c.startswith('T') and c[1:].isdigit()]:
        cens = df[c].isna()
        df[c + '_censored'] = cens
        df.loc[cens, c] = t_max if np.isscalar(t_max) else t_max[cens]
        d = 'D' + c[1:]
        if d in df:
            df.loc[df[d].isna(), d] = df.loc[df[d].isna(), 'path_length']
    return df


def cliffs_delta(x, y):
    """δ = P(X > Y) - P(X < Y). |δ|<0.147 despreciable, <0.33 pequeño, <0.474 mediano, resto grande."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def cliffs_label(d):
    a = abs(d)
    if not np.isfinite(a):
        return ''
    return 'despreciable' if a < 0.147 else 'pequeño' if a < 0.33 else 'mediano' if a < 0.474 else 'grande'


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for r, k in enumerate(order):
        running = max(running, min(1.0, (m - r) * p[k]))
        out[k] = running
    return out


def summary_table(df, metrics, by=('loc_mode', 'map', 'method')):
    rows = []
    for key, g in df.groupby(list(by)):
        row = dict(zip(by, key))
        row['n'] = len(g)
        row['success_rate'] = g['success'].mean() if 'success' in g else np.nan
        lo, hi = wilson_ci(int(g['success'].sum()), len(g)) if 'success' in g else (np.nan, np.nan)
        row['success_ci_lo'], row['success_ci_hi'] = lo, hi
        for m in metrics:
            if m not in g:
                continue
            v = pd.to_numeric(g[m], errors='coerce').dropna()
            row[f'{m}_median'] = v.median() if len(v) else np.nan
            row[f'{m}_q1'] = v.quantile(0.25) if len(v) else np.nan
            row[f'{m}_q3'] = v.quantile(0.75) if len(v) else np.nan
            row[f'{m}_mean'] = v.mean() if len(v) else np.nan
            row[f'{m}_std'] = v.std() if len(v) > 1 else np.nan
            if f'{m}_censored' in g:
                row[f'{m}_censored_frac'] = g[f'{m}_censored'].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(df, metrics, methods=None, group_by=('loc_mode', 'map')):
    """Wilcoxon pareado para cada par de métodos, métrica y grupo (+ 'ALL' agrupando mapas).

    Devuelve un DataFrame con p, p_holm (corrigiendo por toda la familia de un mismo
    loc_mode), mediana de diferencias, Cliff's δ y quién gana.
    """
    methods = methods or sorted(df['method'].unique())
    rows = []
    groups = []
    for key, g in df.groupby(list(group_by)):
        groups.append((dict(zip(group_by, key)), g))
    for lm, g in df.groupby('loc_mode'):
        groups.append(({'loc_mode': lm, 'map': 'ALL'}, g))
    for gkey, g in groups:
        for a, b in itertools.combinations(methods, 2):
            ga = g[g['method'] == a].set_index(BLOCK)
            gb = g[g['method'] == b].set_index(BLOCK)
            common = ga.index.intersection(gb.index)
            for m in metrics:
                if m not in g:
                    continue
                x = pd.to_numeric(ga.loc[common, m], errors='coerce')
                y = pd.to_numeric(gb.loc[common, m], errors='coerce')
                ok = x.notna() & y.notna()
                x, y = x[ok].to_numpy(), y[ok].to_numpy()
                n = len(x)
                p = float('nan')
                if n >= 5 and np.any(x != y):
                    try:
                        p = float(stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided').pvalue)
                    except ValueError:
                        p = float('nan')
                d = cliffs_delta(x, y)
                better = METRICS.get(m, (m, 'lower'))[1]
                med = float(np.median(x - y)) if n else float('nan')
                winner = ''
                if n and np.isfinite(med) and med != 0:
                    winner = a if (med < 0) == (better == 'lower') else b
                rows.append(dict(**gkey, metric=m, method_a=a, method_b=b, n_pairs=n,
                                 median_a=float(np.median(x)) if n else np.nan,
                                 median_b=float(np.median(y)) if n else np.nan,
                                 median_diff=med, cliffs_delta=d, effect=cliffs_label(d),
                                 p_value=p, better_is=better, favours=winner))
    out = pd.DataFrame(rows)
    if len(out):
        out['p_holm'] = np.nan
        for (lm, scope), idx in out.groupby(['loc_mode', out['map'] == 'ALL']).groups.items():
            out.loc[idx, 'p_holm'] = holm(out.loc[idx, 'p_value'].to_numpy())
        out['significant_0.05'] = out['p_holm'] < 0.05
    return out


def mcnemar_success(df, methods=None):
    methods = methods or sorted(df['method'].unique())
    rows = []
    for lm, g in df.groupby('loc_mode'):
        for a, b in itertools.combinations(methods, 2):
            ga = g[g['method'] == a].set_index(BLOCK)['success']
            gb = g[g['method'] == b].set_index(BLOCK)['success']
            c = ga.index.intersection(gb.index)
            sa, sb = ga.loc[c].astype(bool), gb.loc[c].astype(bool)
            n01, n10 = int((~sa & sb).sum()), int((sa & ~sb).sum())
            p = float(stats.binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
            rows.append(dict(loc_mode=lm, method_a=a, method_b=b, n_pairs=len(c),
                             success_a=float(sa.mean()), success_b=float(sb.mean()),
                             only_a=n10, only_b=n01, p_mcnemar_exact=p))
    return pd.DataFrame(rows)


def friedman(df, metric, methods=None):
    """Friedman + rangos medios y diferencia crítica de Nemenyi (α = 0,05)."""
    methods = methods or sorted(df['method'].unique())
    if len(methods) < 3:
        return None
    rows = []
    for lm, g in df.groupby('loc_mode'):
        wide = g.pivot_table(index=BLOCK[1:], columns='method', values=metric).dropna()
        wide = wide[methods]
        if len(wide) < 3:
            continue
        better = METRICS.get(metric, (metric, 'lower'))[1]
        stat, p = stats.friedmanchisquare(*[wide[m].to_numpy() for m in methods])
        ranks = (wide if better == 'lower' else -wide).rank(axis=1).mean()
        k, n = len(methods), len(wide)
        q = stats.studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2)
        cd = q * math.sqrt(k * (k + 1) / (6 * n))
        rows.append(dict(loc_mode=lm, metric=metric, n_blocks=n, chi2=stat, p=p, cd_nemenyi=cd,
                         **{f'rank_{m}': ranks[m] for m in methods}))
    return pd.DataFrame(rows)


def kaplan_meier(times, observed):
    """Estimador KM de S(t) = P(no haber alcanzado el nivel antes de t)."""
    t = np.asarray(times, float)
    e = np.asarray(observed, bool)
    order = np.argsort(t)
    t, e = t[order], e[order]
    uniq = np.unique(t[e])
    s, xs, ys = 1.0, [0.0], [1.0]
    for u in uniq:
        at_risk = (t >= u).sum()
        d = ((t == u) & e).sum()
        s *= 1 - d / at_risk
        xs.append(u)
        ys.append(s)
    return np.array(xs), np.array(ys)
