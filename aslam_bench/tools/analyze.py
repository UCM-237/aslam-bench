"""Calcula métricas de todos los episodios de una campaña y genera el análisis estadístico.

  python -m aslam_bench.tools.analyze results/iter1_sim2d

Salida en <campaña>/analysis/:
  metrics.csv            una fila por episodio (todas las métricas)
  curves.csv             curvas de cobertura (t, cobertura, distancia) por episodio
  summary.csv            mediana/IQR/media/desv. por (modo, mapa, método)
  tests_wilcoxon.csv     pruebas pareadas por par de métodos, métrica y mapa (+ ALL)
  tests_mcnemar.csv      éxito pareado
  friedman.csv           solo si hay >= 3 métodos
  fig_*.png              curvas de cobertura, boxplots, Kaplan-Meier
  report.md              informe legible con las tablas principales
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from ..core import stats as st  # noqa: E402
from ..core.config import resource_dir  # noqa: E402
from ..core.metrics import compute_campaign_metrics  # noqa: E402

# Paleta categórica validada (slots 1..8, orden fijo; el color sigue al método, no al rango)
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

KEY_METRICS = ['T90', 'T95', 'D95', 'auc', 'coverage_final', 'path_length', 'rotation_total',
               'occ_recall', 'free_iou', 'adnn', 'ate_rmse', 'decision_time_mean', 'n_collisions']


def _style():
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 160, 'font.size': 9,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'lines.linewidth': 2.0, 'figure.facecolor': '#fcfcfb',
        'axes.facecolor': '#fcfcfb'})


def method_colors(methods):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(methods))}


def _grid_curve(curves, t_grid, xcol='t'):
    """Interpola cada episodio a una rejilla común (manteniendo el último valor)."""
    out = []
    for _, c in curves.groupby(['start_id', 'rep']):
        c = c.sort_values(xcol)
        x, y = c[xcol].to_numpy(), c['coverage'].to_numpy()
        yi = np.interp(t_grid, x, y, left=0.0, right=y[-1])
        out.append(yi)
    return np.array(out)


def fig_coverage(curves, metrics, colors, lm, out, xcol='t', xlabel='Tiempo (s)'):
    maps = sorted(curves['map'].unique())
    fig, axes = plt.subplots(1, len(maps), figsize=(4.2 * len(maps), 3.2), sharey=True, squeeze=False)
    xmax = curves[xcol].max() if xcol == 't' else curves.groupby('map')[xcol].max().max()
    for ax, m in zip(axes[0], maps):
        cm = curves[curves['map'] == m]
        xm = cm[xcol].max()
        grid = np.linspace(0, xm, 200)
        for meth in sorted(cm['method'].unique()):
            arr = _grid_curve(cm[cm['method'] == meth], grid, xcol)
            if not len(arr):
                continue
            med, q1, q3 = np.median(arr, 0), np.percentile(arr, 25, 0), np.percentile(arr, 75, 0)
            ax.fill_between(grid, q1, q3, color=colors[meth], alpha=0.18, linewidth=0)
            ax.plot(grid, med, color=colors[meth], label=f'{meth} (n={len(arr)})')
        ax.axhline(0.95, color=INK2, linewidth=0.8, linestyle='--')
        ax.set_title(m, color=INK)
        ax.set_xlabel(xlabel)
        ax.text(0.99, 0.955, 'objetivo 95 %', transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                fontsize=7, color=INK2)
        ax.set_ylim(0, 1.01)
    axes[0][0].set_ylabel('Cobertura del área alcanzable')
    axes[0][-1].legend(loc='lower right')
    fig.suptitle(f'Cobertura vs {xlabel.split(" ")[0].lower()} — mediana y rango intercuartílico ({lm})', color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_boxplots(df, colors, lm, out, metrics=('T95', 'D95', 'auc', 'ate_rmse')):
    maps = sorted(df['map'].unique())
    methods = sorted(df['method'].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.3), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        pos, data, cols = [], [], []
        width = 0.8 / len(methods)
        for i, mp in enumerate(maps):
            for k, meth in enumerate(methods):
                v = pd.to_numeric(df[(df['map'] == mp) & (df['method'] == meth)][m], errors='coerce').dropna()
                data.append(v.to_numpy())
                pos.append(i + (k - (len(methods) - 1) / 2) * width)
                cols.append(colors[meth])
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, linewidth=1.5),
                        flierprops=dict(marker='o', markersize=3, markeredgecolor=INK2))
        for patch, c in zip(bp['boxes'], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
            patch.set_edgecolor(INK2)
        ax.set_xticks(range(len(maps)))
        ax.set_xticklabels(maps, rotation=15)
        ax.set_title(st.METRICS.get(m, (m,))[0], color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m], alpha=0.75) for m in methods]
    fig.legend(handles, methods, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f'({lm}; T95/D95 censurados imputados a t_max / distancia total)', y=-0.02, color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_km(raw, colors, lm, out):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for meth in sorted(raw['method'].unique()):
        g = raw[raw['method'] == meth]
        obs = g['T95'].notna().to_numpy()
        t = np.where(obs, g['T95'], g['duration'])   # censura en el instante en que acabó el episodio
        xs, ys = st.kaplan_meier(t, obs)
        ax.step(np.append(xs, g['t_max'].max()), np.append(ys, ys[-1]), where='post', color=colors[meth],
                label=f'{meth} ({obs.sum()}/{len(obs)} alcanzan 95 %)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('P(no haber llegado al 95 %)')
    ax.set_ylim(0, 1.02)
    ax.set_title(f'Tiempo hasta 95 % de cobertura, Kaplan-Meier ({lm})', fontsize=9)
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '–'
    if isinstance(v, (bool, np.bool_)):
        return 'sí' if v else 'no'
    if isinstance(v, (int, np.integer)):
        return str(v)
    if abs(v) >= 100:
        return f'{v:.0f}'
    return f'{v:.{nd}f}'


def _pfmt(p):
    if not np.isfinite(p):
        return '–'
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def write_report(an_dir, camp, raw, summary, tests, mcn, fried, figs):
    L = []
    L.append(f'# Informe de la campaña `{camp["name"]}`\n')
    L.append(f'Backend: **{camp["backend"]}** · métodos: {", ".join(sorted(raw["method"].unique()))} · '
             f'mapas: {", ".join(sorted(raw["map"].unique()))} · episodios: {len(raw)}\n')
    L.append('Diseño pareado: en cada bloque (modo de localización, mapa, pose inicial, repetición) se '
             'ejecutan todos los métodos en las mismas condiciones. Los tiempos y distancias hasta un nivel '
             'de cobertura que no se alcanza se imputan a `t_max` y a la distancia total recorrida '
             '(se indica la fracción censurada). Las pruebas son Wilcoxon de rangos con signo pareadas con '
             'corrección de Holm por familia (por modo de localización; por mapa y global por separado) y '
             'tamaño del efecto Cliff\'s δ (A − B; δ<0 ⇒ A tiende a valores menores).\n')
    for lm in sorted(raw['loc_mode'].unique()):
        L.append(f'\n## Modo de localización: `{lm}`\n')
        for f in figs.get(lm, []):
            L.append(f'![{os.path.basename(f)}]({os.path.basename(f)})\n')
        s = summary[summary['loc_mode'] == lm]
        L.append('\n### Resumen por mapa (mediana [Q1–Q3])\n')
        cols = ['T95', 'D95', 'auc', 'coverage_final', 'path_length', 'occ_recall', 'ate_rmse', 'decision_time_mean']
        L.append('| Mapa | Método | n | Éxito (IC95 %) | ' + ' | '.join(st.METRICS[c][0] for c in cols) + ' |')
        L.append('|' + '---|' * (4 + len(cols)))
        for _, r in s.sort_values(['map', 'method']).iterrows():
            cells = []
            for c in cols:
                cell = f'{_fmt(r.get(c + "_median"))} [{_fmt(r.get(c + "_q1"))}–{_fmt(r.get(c + "_q3"))}]'
                cf = r.get(c + '_censored_frac')
                if cf is not None and np.isfinite(cf) and cf > 0:
                    cell += f' ({cf:.0%} cens.)'
                cells.append(cell)
            L.append(f'| {r["map"]} | {r["method"]} | {r["n"]} | {r["success_rate"]:.0%} '
                     f'({r["success_ci_lo"]:.0%}–{r["success_ci_hi"]:.0%}) | ' + ' | '.join(cells) + ' |')
        t = tests[(tests['loc_mode'] == lm) & (tests['map'] == 'ALL')]
        if len(t):
            L.append('\n### Comparación pareada global (todos los mapas)\n')
            L.append('| Métrica | A | B | pares | mediana A | mediana B | δ de Cliff | efecto | p (Holm) | favorece |')
            L.append('|---|---|---|---|---|---|---|---|---|---|')
            for _, r in t.iterrows():
                sig = ' **✓**' if r['p_holm'] < 0.05 else ''
                L.append(f'| {st.METRICS.get(r["metric"], (r["metric"],))[0]} | {r["method_a"]} | {r["method_b"]} | '
                         f'{r["n_pairs"]} | {_fmt(r["median_a"])} | {_fmt(r["median_b"])} | {_fmt(r["cliffs_delta"])} | '
                         f'{r["effect"]} | {_pfmt(r["p_holm"])}{sig} | {r["favours"] or "–"} |')
        t = tests[(tests['loc_mode'] == lm) & (tests['map'] != 'ALL') & tests['metric'].isin(['T95', 'D95', 'auc'])]
        if len(t):
            L.append('\n### Por mapa (T95, D95, AUC)\n')
            L.append('| Mapa | Métrica | A | B | pares | δ de Cliff | p (Holm) | favorece |')
            L.append('|---|---|---|---|---|---|---|---|')
            for _, r in t.iterrows():
                sig = ' **✓**' if r['p_holm'] < 0.05 else ''
                L.append(f'| {r["map"]} | {r["metric"]} | {r["method_a"]} | {r["method_b"]} | {r["n_pairs"]} | '
                         f'{_fmt(r["cliffs_delta"])} | {_pfmt(r["p_holm"])}{sig} | {r["favours"] or "–"} |')
        m = mcn[mcn['loc_mode'] == lm] if len(mcn) else mcn
        if len(m):
            L.append('\n### Tasa de éxito (alcanzar 95 % de cobertura), McNemar exacto\n')
            L.append('| A | B | pares | éxito A | éxito B | solo A | solo B | p |')
            L.append('|---|---|---|---|---|---|---|---|')
            for _, r in m.iterrows():
                L.append(f'| {r["method_a"]} | {r["method_b"]} | {r["n_pairs"]} | {r["success_a"]:.0%} | '
                         f'{r["success_b"]:.0%} | {r["only_a"]} | {r["only_b"]} | {_pfmt(r["p_mcnemar_exact"])} |')
        term = raw[raw['loc_mode'] == lm].groupby(['method', 'termination']).size().unstack(fill_value=0)
        L.append('\n### Motivos de terminación\n')
        L.append('| Método | ' + ' | '.join(term.columns) + ' |')
        L.append('|---|' + '---|' * len(term.columns))
        for meth, r in term.iterrows():
            L.append(f'| {meth} | ' + ' | '.join(str(v) for v in r.to_numpy()) + ' |')
    if fried is not None and len(fried):
        L.append('\n## Friedman (≥ 3 métodos)\n')
        L.append(fried.to_markdown(index=False) if hasattr(fried, 'to_markdown') else fried.to_string())
    L.append('\n---\nTablas completas: `summary.csv`, `tests_wilcoxon.csv`, `tests_mcnemar.csv`; datos por '
             'episodio: `metrics.csv`, `curves.csv`.\n')
    with open(os.path.join(an_dir, 'report.md'), 'w') as f:
        f.write('\n'.join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('campaign_dir')
    ap.add_argument('--maps', default=None)
    ap.add_argument('--recompute', action='store_true', help='recalcular metrics.csv aunque exista')
    a = ap.parse_args(argv)
    with open(os.path.join(a.campaign_dir, 'campaign.yaml')) as f:
        saved = yaml.safe_load(f)
    camp, cfg = saved['campaign'], saved['platform']
    maps_dir = a.maps or resource_dir('maps')
    an = os.path.join(a.campaign_dir, 'analysis')
    os.makedirs(an, exist_ok=True)
    mpath, cpath = os.path.join(an, 'metrics.csv'), os.path.join(an, 'curves.csv')
    if a.recompute or not os.path.exists(mpath):
        raw, curves = compute_campaign_metrics(a.campaign_dir, cfg, maps_dir, mpath, cpath)
    else:
        raw, curves = pd.read_csv(mpath), pd.read_csv(cpath)
    print(f'{len(raw)} episodios con métricas')

    df = st.impute_censored(raw)
    metrics = [m for m in KEY_METRICS if m in df]
    summary = st.summary_table(df, metrics)
    summary.to_csv(os.path.join(an, 'summary.csv'), index=False)
    tests = st.paired_tests(df, metrics)
    tests.to_csv(os.path.join(an, 'tests_wilcoxon.csv'), index=False)
    mcn = st.mcnemar_success(df)
    mcn.to_csv(os.path.join(an, 'tests_mcnemar.csv'), index=False)
    fried = None
    if df['method'].nunique() >= 3:
        fried = pd.concat([st.friedman(df, m) for m in ('T95', 'D95', 'auc')], ignore_index=True)
        fried.to_csv(os.path.join(an, 'friedman.csv'), index=False)

    _style()
    colors = method_colors(df['method'].unique())
    figs = {}
    for lm in sorted(df['loc_mode'].unique()):
        c = curves[curves['loc_mode'] == lm]
        f1 = os.path.join(an, f'fig_coverage_time_{lm}.png')
        fig_coverage(c, df, colors, lm, f1)
        f2 = os.path.join(an, f'fig_coverage_distance_{lm}.png')
        fig_coverage(c, df, colors, lm, f2, xcol='distance', xlabel='Distancia recorrida (m)')
        f3 = os.path.join(an, f'fig_boxplots_{lm}.png')
        # con pose oráculo el ATE es ~0 por construcción: se muestra la cobertura final
        bm = ('T95', 'D95', 'auc', 'coverage_final' if lm == 'oracle' else 'ate_rmse')
        fig_boxplots(df[df['loc_mode'] == lm], colors, lm, f3, metrics=bm)
        f4 = os.path.join(an, f'fig_km_{lm}.png')
        fig_km(raw[raw['loc_mode'] == lm], colors, lm, f4)
        figs[lm] = [f1, f2, f3, f4]
    write_report(an, camp, raw, summary, tests, mcn, fried, figs)
    print(f'Análisis en {an}/report.md')


if __name__ == '__main__':
    main()
