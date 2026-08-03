#!/usr/bin/env python3
'''Further diagnostic figures for the 2D dipping-DZ profile inversions
(invert_twod_profiles.py results, results/profile_inversion/<outtag>/).

For each inverted profile, produces three figures under <outtag>/figs/diagnostics/:

    profile_NN_tradeoffs.png   parameter tradeoff ("corner") plot -- lower-triangle
                               scatter of every parameter pair + marginal histograms
                               on the diagonal, divergent NUTS transitions in red,
                               best (max-likelihood posterior) sample marked with a star.
    profile_NN_traces.png      per-chain posterior density + trace-vs-draw-index, one
                               row per parameter, annotated with rhat/ess (from the
                               stats already computed by invert_twod_profiles.py) --
                               makes visible *why* rhat is bad (chains apart) rather
                               than just the number.
    profile_NN_health.png      sampler-health check independent of rhat/ess: the NUTS
                               energy histogram (BFMI) and per-chain divergence rate.

Plus one summary figure, convergence_dashboard.png: rhat / ESS / divergence rate
per parameter, along strike, across every inverted profile.

See split_rhat / ess_bulk / convergence_stats in invert_twod_profiles.py for the
rhat/ESS definitions used here (same numbers, already stored per result under
`stats`); bfmi() below is the analogous per-chain sampler-health diagnostic for
`sample_stats.energy`, not computed at inversion time.

Usage:
    plot_inversion_diagnostics.py                  # all profiles in --outtag
    plot_inversion_diagnostics.py --profile 12      # just profile i=12
    plot_inversion_diagnostics.py --outtag twod01
'''
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import config
from result_io import load_result, param_labels

BFMI_POOR = 0.3   # Betancourt (2016): BFMI below this flags an under-exploring sampler


####    SAMPLER-HEALTH DIAGNOSTICS    ####
def bfmi(idata):
    '''Per-chain Bayesian Fraction of Missing Information: the ratio of the
    variance of the energy *transition* (successive-draw differences) to the
    marginal variance of the energy within the chain. Low BFMI means HMC's
    momentum resampling isn't reaching all the energy levels the posterior
    has -- a sign the sampler is struggling even when rhat/ess look fine,
    since rhat/ess are about the parameters, not the sampler's own dynamics.'''
    E = idata.sample_stats['energy'].values          # (chain, draw)
    dE = np.diff(E, axis=1)
    return (dE**2).mean(axis=1) / E.var(axis=1, ddof=1)


def divergence_rate(idata):
    return float(idata.sample_stats['diverging'].values.mean())


####    FIGURES    ####
def plot_tradeoffs(r, labels, path):
    '''Lower-triangle pairwise scatter + diagonal marginals: the main tool for
    seeing *which* parameters trade off against each other (e.g. dz_halfwidth
    vs modulus_ratio -- a wider, softer damage zone and a smaller modulus
    contrast can produce a similar profile shape, so the posterior often runs
    along a ridge in that plane rather than isolating a point).'''
    idata = r['idata']
    post = idata.posterior
    div = idata.sample_stats['diverging'].values.reshape(-1).astype(bool)
    samp = {l: post[l].values.reshape(-1) for l in labels}
    best = r['best']
    n = len(labels)

    fig, axes = plt.subplots(n, n, figsize=(2.3 * n, 2.3 * n), layout='constrained')
    axes = np.atleast_2d(axes)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            ax = axes[i, j]
            if i == j:
                ax.hist(samp[li], bins=40, color='steelblue', alpha=0.8)
                ax.axvline(best[li], color='crimson', lw=1.2)
                ax.set_yticks([])
            elif i > j:
                ax.scatter(samp[lj][~div], samp[li][~div], s=3, alpha=0.15,
                           color='0.4', rasterized=True, linewidths=0)
                if div.any():
                    ax.scatter(samp[lj][div], samp[li][div], s=8, color='red',
                               alpha=0.8, linewidths=0, label='divergent')
                ax.scatter(best[lj], best[li], marker='*', color='gold',
                           edgecolor='k', s=140, zorder=5, label='best sample')
                rho = np.corrcoef(samp[lj], samp[li])[0, 1]
                ax.text(0.05, 0.92, f'r={rho:+.2f}', transform=ax.transAxes,
                        fontsize=7, va='top',
                        color='crimson' if abs(rho) > 0.6 else 'black')
            else:
                ax.axis('off')
                continue
            if i == n - 1:
                ax.set_xlabel(lj, fontsize=8)
            if j == 0:
                ax.set_ylabel(li, fontsize=8)
            ax.tick_params(labelsize=6)

    n_div = int(div.sum())
    title = f'Profile {r["i"]:02d}: parameter tradeoffs'
    if n_div:
        title += f'  ({n_div}/{div.size} divergent draws in red)'
    if n_div:
        axes[1, 0].legend(fontsize=6, loc='upper right')
    fig.suptitle(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_traces(r, labels, path):
    '''Per-chain KDE + trace-vs-draw, one row per parameter. rhat/ess (already
    computed by invert_twod_profiles.py, stored in `stats`) are printed in each
    row's title -- seeing *which* chains disagree (separated KDE peaks, traces
    that don't overlap) is what a bare rhat number can't show you.'''
    post = r['idata'].posterior
    stats = r['stats']
    n = len(labels)
    fig, axes = plt.subplots(n, 2, figsize=(10, 1.9 * n), layout='constrained',
                             squeeze=False)
    cmap = plt.get_cmap('tab10')
    for k, l in enumerate(labels):
        x = post[l].values   # (chain, draw)
        ax_kde, ax_trace = axes[k]
        for c in range(x.shape[0]):
            color = cmap(c % 10)
            if np.ptp(x[c]) > 0:
                grid = np.linspace(x[c].min(), x[c].max(), 200)
                ax_kde.plot(grid, gaussian_kde(x[c])(grid), color=color, lw=0.9, alpha=0.85)
            ax_trace.plot(x[c], color=color, lw=0.4, alpha=0.7)
        ax_kde.set_ylabel(l, fontsize=8)
        ax_kde.set_yticks([])
        ax_trace.tick_params(labelsize=6)
        poor = l in stats.get('poor', [])
        ax_trace.set_title(f'rhat={stats[l]["rhat"]:.3f}  ess={stats[l]["ess"]:.0f}',
                           fontsize=8, color='crimson' if poor else '0.2')
    axes[0, 0].set_title('per-chain posterior density', fontsize=9)
    fig.suptitle(f'Profile {r["i"]:02d}: chain mixing (one color per chain)')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sampler_health(r, path):
    '''BFMI (is HMC's momentum resampling reaching the full energy range the
    posterior has?) and per-chain divergence rate (is NUTS refusing to cross
    some region of parameter space -- usually a sharp funnel/ridge in the
    posterior geometry, and the reason a handful of red points showed up in
    the tradeoff plot). Both catch sampler problems rhat/ess can miss.'''
    idata = r['idata']
    E = idata.sample_stats['energy'].values          # (chain, draw)
    div = idata.sample_stats['diverging'].values      # (chain, draw)
    b = bfmi(idata)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')

    E_c = E - E.mean()
    dE_c = np.diff(E, axis=1)
    dE_c = dE_c - dE_c.mean()
    ax1.hist(E_c.reshape(-1), bins=40, alpha=0.55, density=True,
             color='steelblue', label='marginal energy')
    ax1.hist(dE_c.reshape(-1), bins=40, alpha=0.55, density=True,
             color='darkorange', label='energy transition')
    ax1.set_xlabel('energy (centred)')
    ax1.legend(fontsize=8)
    bfmi_str = ', '.join(f'{v:.2f}' for v in b)
    flag = '  <-- LOW (sampler under-exploring)' if (b < BFMI_POOR).any() else ''
    ax1.set_title(f'BFMI per chain: {bfmi_str}{flag}', fontsize=8)

    rate = div.mean(axis=1) * 100
    ax2.bar(np.arange(len(rate)), rate, color='crimson')
    ax2.set_xlabel('chain')
    ax2.set_ylabel('% divergent draws')
    ax2.set_title(f'total: {int(div.sum())}/{div.size} divergences '
                 f'({100 * div.mean():.2f}%)', fontsize=8)

    fig.suptitle(f'Profile {r["i"]:02d}: sampler health')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_dashboard(results, labels, path):
    '''rhat / ESS / divergence rate per parameter, along strike, across every
    inverted profile -- the "which profiles can I trust" overview, analogous
    to summary_along_strike.png but for convergence rather than the estimated
    parameters themselves.'''
    fig, axes = plt.subplots(len(labels) + 1, 1, figsize=(9, 2.1 * (len(labels) + 1)),
                             sharex=True, layout='constrained')
    axes = np.atleast_1d(axes)

    for ax, prm in zip(axes[:-1], labels):
        ax2 = ax.twinx()
        for fid, marker, color in ((0, 'o', 'steelblue'), (1, 's', 'darkorange')):
            rs = sorted((r for r in results if r['fault_id'] == fid),
                        key=lambda r: r['x_along_fault'])
            if not rs:
                continue
            km = [r['x_along_fault'] / 1000. for r in rs]
            rhat = [r['stats'][prm]['rhat'] for r in rs]
            ess = [r['stats'][prm]['ess'] for r in rs]
            ax.plot(km, rhat, marker=marker, color=color, ms=5, lw=1,
                    label=f'fault {fid}')
            ax2.plot(km, ess, marker=marker, color=color, ms=4, lw=0.8,
                     ls=':', alpha=0.6)
        ax.axhline(1.05, color='crimson', ls='--', lw=0.7)
        ax.set_yscale('log')   # rhat spans ~1 to >100 across profiles; linear crushes the
                               # good-vs-bad distinction near 1.0 that actually matters
        ax.set_ylabel(f'{prm}\nR-hat (solid, log)', fontsize=8)
        ax2.set_ylabel('ESS (dotted)', fontsize=8)
        ax.grid(True, ls=':', alpha=0.4)

    ax = axes[-1]
    for fid, marker, color in ((0, 'o', 'steelblue'), (1, 's', 'darkorange')):
        rs = sorted((r for r in results if r['fault_id'] == fid),
                    key=lambda r: r['x_along_fault'])
        if not rs:
            continue
        km = [r['x_along_fault'] / 1000. for r in rs]
        rate = [100 * divergence_rate(r['idata']) for r in rs]
        ax.plot(km, rate, marker=marker, color=color, ms=5, lw=1, label=f'fault {fid}')
    ax.set_ylabel('% divergent', fontsize=8)
    ax.set_xlabel('Along-strike distance (km)')
    ax.legend(fontsize=8)
    ax.grid(True, ls=':', alpha=0.4)

    axes[0].set_title('Convergence diagnostics along strike (R-hat / ESS / divergence rate)')
    axes[0].legend(fontsize=8)
    fig.align_ylabels(axes)
    fig.savefig(path, dpi=180)
    plt.close(fig)


####    MAIN    ####
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outtag', default='twod01',
                    help='results/profile_inversion/<outtag> to read (default twod01)')
    ap.add_argument('--profile', type=int, default=None,
                    help='diagnose only the result with this index i')
    args = ap.parse_args()

    resdir = config.RESULTS_DIR / 'profile_inversion' / args.outtag
    figdir = resdir / 'figs' / 'diagnostics'
    figdir.mkdir(parents=True, exist_ok=True)

    paths = sorted(resdir.glob('profile_*.pickle'))
    if args.profile is not None:
        paths = [p for p in paths if int(p.stem.split('_')[1]) == args.profile]
        if not paths:
            raise SystemExit(f'no result pickle for profile {args.profile} in {resdir}')

    results = []
    for p in paths:
        try:
            r = load_result(p)
        except Exception as e:
            print(f'[diag] {p.name}: failed to load ({type(e).__name__}: {e}), skipping')
            continue
        labels = param_labels(r)
        plot_tradeoffs(r, labels, figdir / f'profile_{r["i"]:02d}_tradeoffs.png')
        plot_traces(r, labels, figdir / f'profile_{r["i"]:02d}_traces.png')
        plot_sampler_health(r, figdir / f'profile_{r["i"]:02d}_health.png')
        print(f'[diag] profile {r["i"]:02d}: tradeoffs + traces + health -> {figdir}')
        results.append(r)

    if len(results) > 1:
        labels = [l for l in param_labels(results[0]) if l != 'offset']
        plot_dashboard(results, labels, figdir / 'convergence_dashboard.png')
        print(f'[diag] dashboard ({len(results)} profiles) -> '
              f'{figdir / "convergence_dashboard.png"}')


if __name__ == '__main__':
    main()
