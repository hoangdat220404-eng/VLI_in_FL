"""Offline comparison of label-free choices of the VLI* divisor G*.

Input: .npz files written by `main.py --dump_signal FILE` (one row per window:
G = summed bias gradient of the window, N = true label counts).
For every estimator, VLI* is recomputed with that G* exactly as in the attack
(VLI_star_closed_form -> simplex projection -> integer counts), so the numbers
match what --scheme VLI_star / All would report.

Usage (from VLI_in_FL/):
    python scripts/eval_gstar.py signals/*.npz
    python scripts/eval_gstar.py signals/*.npz --per_file      # one table per setting

Selection criterion printed at the end: regret = (best label-free rule of that
setting) - irec(rule), per setting (file), then the WORST setting. A rule that is
stable across settings has a small worst-setting regret, not just a good average.
All windows are kept, including IID-like ones with no absent class.
ORACLE (mean G over the true absent set) is a reference only, never an attack.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from client.attacks import VLI_star_closed_form, vli_star_gstar, _project_to_simplex  # noqa: E402


# ---------------------------------------------------------------- estimators
def _pos(G):
    p = np.sort(G[G > 0.0])[::-1]
    if p.size == 0:
        raise ValueError('no positive component')
    return p


def est_upper_mean(G, n=None):
    """Mean of the positives that lie at or above mean+ (one trimming step)."""
    p = _pos(G)
    return float(p[p >= p.mean()].mean())


def est_upper_median(G, n=None):
    p = _pos(G)
    return float(np.median(p[p >= p.mean()]))


def est_quantile(q):
    def f(G, n=None):
        return float(np.quantile(_pos(G), q))
    return f


def est_top_fraction(frac):
    def f(G, n=None):
        p = _pos(G)
        k = max(1, int(round(frac * p.size)))
        return float(p[:k].mean())
    return f


def est_blend(w):
    def f(G, n=None):
        p = _pos(G)
        return float(w * p[0] + (1.0 - w) * p.mean())
    return f


def est_geo_upper(G, n=None):
    """Geometric mean of positives >= mean+ (absent-class spread is multiplicative)."""
    p = _pos(G)
    u = p[p >= p.mean()]
    return float(np.exp(np.log(u).mean()))


def est_fixed_point(G, n):
    """Smallest G* (scanning up from mean+) whose predicted-absent set has mean G = G*."""
    p = _pos(G)
    grid = np.linspace(p.mean(), p[0], 200)
    for g in grid:
        raw = (n / G.size) * (1.0 - G / g)
        proj = _project_to_simplex(raw, n)
        zero = proj < 0.5
        if zero.any() and G[zero].mean() <= g:
            return float(g)
    return float(p[0])


def est_mixture(G, n=None, s_floor=0.1):
    """Two-component fit of the positive G_j, following the VLI* model itself.

    absent classes   : G_j = G* x (multiplicative noise)  -> log-normal(mu, s), K of them
    present, N_j<n/C : G_j = G* (1 - C N_j / n) in (0, G*) -> Uniform(0, G*)
    The two components share G* = E[log-normal] = exp(mu + s^2/2): the edge of the
    uniform part and the centre of the absent cluster are the same number, which is
    what lets the fit tell 'few absent classes' (IID) from 'many spread ones'.
    No mixing weight is fixed in advance, so K is estimated, not assumed.
    """
    from scipy.optimize import minimize
    p = _pos(G)
    if p.size < 3:
        return float(p[0])
    lx = np.log(p)

    def unpack(t):
        mu, ls, lp = t
        s = s_floor + np.exp(ls)
        pi = 1.0 / (1.0 + np.exp(-lp))
        return mu, s, pi, np.exp(mu + 0.5 * s * s)

    def nll(t):
        mu, s, pi, gs = unpack(t)
        ln_pdf = np.exp(-0.5 * ((lx - mu) / s) ** 2) / (p * s * np.sqrt(2 * np.pi))
        uni = (p < gs) / gs
        return -np.sum(np.log(pi * ln_pdf + (1 - pi) * uni + 1e-300))

    best = None
    for mu0 in (np.log(p[0]), np.log(p.mean()), np.log(np.median(p))):
        for s0 in (0.15, 0.4, 0.8):
            for pi0 in (-1.0, 0.0, 1.0):
                r = minimize(nll, [mu0, np.log(s0), pi0], method='Nelder-Mead',
                             options=dict(maxiter=600, xatol=1e-4, fatol=1e-6))
                if best is None or r.fun < best.fun:
                    best = r
    return float(unpack(best.x)[3])


def build_estimators():
    E = OrderedDict()
    for mode in ('max', 'mean+', 'med+', 'gap', 'otsu'):
        E[mode] = (lambda m: (lambda G, n=None: vli_star_gstar(G, m)[0]))(mode)
    E['upmean'] = est_upper_mean
    E['upmed'] = est_upper_median
    E['geoup'] = est_geo_upper
    for q in (0.6, 0.7, 0.8, 0.9):
        E['q%.0f' % (100 * q)] = est_quantile(q)
    for fr in (0.25, 0.5, 0.75):
        E['top%.0f' % (100 * fr)] = est_top_fraction(fr)
    for w in (0.25, 0.5):
        E['blend%.0f' % (100 * w)] = est_blend(w)
    E['fixpt'] = est_fixed_point
    E['mix'] = est_mixture
    return E


# ---------------------------------------------------------------- evaluation
def irec_of(G, N, g):
    n = int(N.sum())
    pred = VLI_star_closed_form(G, n_samples=n, gmax=g)
    return float(np.minimum(pred, N).sum()) / n


def uniform_irec(N):
    n = float(N.sum())
    return float(np.minimum(n / N.size, N).sum()) / n


def load(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['meta']))
    return d['G'], d['N'], meta


def label_of(meta, path):
    ck = os.path.basename(meta.get('check_point_path') or 'untrained')
    return '%s a=%g acc=%.1f (%s)' % (meta['dataset'], meta['alpha'], meta['attacked_acc'], ck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--per_file', action='store_true')
    a = ap.parse_args()

    E = build_estimators()
    names = list(E) + ['ORACLE']
    rows = []                                   # (label, {name: mean irec}, {name: mean lambda}, n_windows)
    for path in sorted(a.files):
        G_all, N_all, meta = load(path)
        irec = {k: [] for k in names}
        lam = {k: [] for k in names}
        used = no_absent = 0
        for G, N in zip(G_all, N_all):
            absent = N == 0
            if not (G > 0).any():
                continue                         # no rule is defined without a positive component
            used += 1
            # Windows with NO absent class (IID-like) are kept: the attacker cannot know
            # this, so every rule must still answer. ORACLE is undefined there and skipped.
            no_absent += int(absent.sum() == 0)
            u = uniform_irec(N)
            gvals = {}
            for k, f in E.items():
                try:
                    gvals[k] = f(G, int(N.sum()))
                except ValueError:
                    gvals[k] = None
            gvals['ORACLE'] = float(G[absent].mean()) if absent.any() else None
            for k in names:
                if gvals[k] is None or gvals[k] <= 0:
                    continue
                r = irec_of(G, N, gvals[k])
                irec[k].append(r)
                lam[k].append((r - u) / (1.0 - u) if u < 1 else np.nan)
        if not used:
            print('skip %s: no usable window' % path)
            continue
        mi = {k: float(np.mean(v)) if v else np.nan for k, v in irec.items()}
        ml = {k: float(np.nanmean(v)) if v else np.nan for k, v in lam.items()}
        rows.append((label_of(meta, path) + ' noabs=%d/%d' % (no_absent, used), mi, ml, used))
        if a.per_file:
            print('\n== %s  [%d windows]' % (rows[-1][0], used))
            for k in names:
                print('  %-11s irec=%.4f  lambda=%+.3f' % (k, mi[k], ml[k]))

    if not rows:
        return
    # ---- one line per setting, irec of every estimator
    short = [k for k in names]
    print('\nirec per setting (columns = G* rule)')
    print('%-58s ' % 'setting' + ' '.join('%7s' % k[:7] for k in short))
    for lab, mi, _, _ in rows:
        print('%-58s ' % lab[:58] + ' '.join('%7.4f' % mi[k] for k in short))
    print('(ORACLE is averaged only over windows that have an absent class; reference only)')

    # ---- regret vs the best label-free rule of each setting: mean and worst over settings.
    # Not vs ORACLE: ORACLE is undefined on IID-like windows and is not always the best G*.
    rules = names[:-1]
    print('\nregret = best rule of the setting - irec(rule); sorted by WORST setting')
    print('%-11s %8s %8s %8s   %s' % ('rule', 'mean', 'worst', 'mean_l', 'worst setting'))
    summary = []
    for k in rules:
        reg = np.array([max(mi[r] for r in rules) - mi[k] for _, mi, _, _ in rows])
        lam_mean = np.nanmean([ml[k] for _, _, ml, _ in rows])
        w = int(np.nanargmax(reg))
        summary.append((np.nanmax(reg), np.nanmean(reg), lam_mean, k, rows[w][0]))
    for worst, mean, lm, k, wl in sorted(summary):
        print('%-11s %8.4f %8.4f %+8.3f   %s' % (k, mean, worst, lm, wl[:60]))


if __name__ == '__main__':
    main()
