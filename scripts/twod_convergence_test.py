#!/usr/bin/env python3
'''Convergence study for the 2D profile inversion (invert_twod_profiles.py).

Runs a couple of representative profiles through increasing sampler settings and
reports, per run: wall time, split-Rhat and ESS per parameter, per-chain
dz_halfwidth means (multimodality check), and overlaid posteriors across
settings. Use this to pick draws/tune/chains for the production run.

Out: results/profile_inversion/twod01/convergence/ (pickles, figures, report.txt)

Run: ./.venv/bin/python scripts/twod_convergence_test.py
'''
import argparse
import os
import pickle
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from codes import HamiltonianInversion
import invert_twod_profiles as inv
import config

PROFILES = [15, 28]           # strong main-strand signal / weak foreshock signal
SETTINGS = [                  # (chains, draws, tune); last = production setting
    (4, 500, 500),
    (8, 1000, 1000),
    (8, 2000, 1500),
    (8, 4000, 2000),
]
TIMEOUT = 120                 # min per run

# production defaults for make_priors(); this script doesn't exercise --fix-mr/--free-dip
DEFAULT_ARGS = argparse.Namespace(fix_mr=None, free_dip=False, mr_lower=0.2)

RESDIR = config.RESULTS_DIR / 'profile_inversion' / 'twod01'
OUTDIR = RESDIR / 'convergence'
split_rhat, ess_bulk = inv.split_rhat, inv.ess_bulk


def run_one(rec, profile, fault, chains, draws, tune):
    x_off, dips, _ = inv.local_dip(fault, profile, rec['fault_id'])
    data, sign, _ = inv.prep_data(rec)
    model = inv.ProfileModel().build_dipping_patches(inv.VD, x_off)
    model.xs = np.asarray(rec['xs'], dtype=float)

    priors = inv.make_priors(DEFAULT_ARGS, dips)
    param_labels = [p.label for p in priors]
    model.param_labels = param_labels

    t0 = time.time()
    hi = HamiltonianInversion(model, priors, data, rec['cov_total'])
    hi = hi.run(draws=draws, tune=tune, chains=chains,
                cores=min(chains, os.cpu_count() or 1), timeout_minutes=TIMEOUT)
    wall = time.time() - t0

    stats = {}
    for l in param_labels:
        x = hi.result.posterior[l].values
        stats[l] = dict(rhat=split_rhat(x), ess=ess_bulk(x),
                        chain_means=x.mean(axis=1))
    return hi.result, stats, wall, model, data, param_labels


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    records = {r['i']: r for r in pickle.load(open(inv.RECORDS_PICKLE, 'rb'))}
    profiles = pickle.load(open(inv.EVALUATED_PICKLE, 'rb'))
    fault = pickle.load(open(config.FAULT_PICKLE, 'rb'))

    report = []
    def log(line=''):
        print(line, flush=True)
        report.append(line)

    for i in PROFILES:
        rec = records[i]
        log(f'\n{"="*80}\nPROFILE {i} (fault {rec["fault_id"]}, '
            f'{rec["x_along_fault"]/1000:.1f} km along strike)\n{"="*80}')
        runs = []
        param_labels = None
        for chains, draws, tune in SETTINGS:
            tag = f'{chains}ch_{draws}d_{tune}t'
            log(f'\n--- {tag} ---')
            idata, stats, wall, model, data, param_labels = run_one(
                records[i], profiles[i], fault, chains, draws, tune)
            best = inv.best_sample(idata, model, data, rec['cov_total'], param_labels)
            runs.append((tag, idata, stats, best))
            with open(OUTDIR / f'profile_{i:02d}_{tag}.pickle', 'wb') as f:
                pickle.dump(dict(i=i, setting=(chains, draws, tune), wall=wall,
                                 stats=stats, best=best, idata=idata), f)

            log(f'wall {wall/60:.1f} min '
                f'({chains*(draws+tune)/wall:.1f} draws/s total)')
            log(f'{"param":15s} {"rhat":>6s} {"ess":>7s}   chain means')
            for l in param_labels:
                s = stats[l]
                cm = ' '.join(f'{v:8.2f}' for v in s['chain_means'])
                flag = '  <-- POOR' if s['rhat'] > 1.05 or s['ess'] < 400 else ''
                log(f'{l:15s} {s["rhat"]:6.3f} {s["ess"]:7.0f}   {cm}{flag}')
            log(f'best sample: dz {best["dz_halfwidth"]:.0f} m, '
                f'mr {best["modulus_ratio"]:.2f}, '
                f'slips {[round(best[l], 2) for l in inv.SLIP_LABELS]}')

        # overlaid posteriors across settings
        fig, axes = plt.subplots(1, len(param_labels), figsize=(19, 3.2),
                                 layout='constrained')
        for k, l in enumerate(param_labels):
            ax = axes[k]
            for tag, idata, stats, _ in runs:
                x = idata.posterior[l].values.flatten()
                ax.hist(x, bins=40, density=True, histtype='step', lw=1.3,
                        label=f'{tag} (rhat {stats[l]["rhat"]:.2f})')
            ax.set_title(l, fontsize=10)
            ax.set_yticks([])
            ax.legend(fontsize=6)
        fig.suptitle(f'Profile {i}: posterior stability across sampler settings')
        fig.savefig(OUTDIR / f'profile_{i:02d}_posteriors.png', dpi=180)
        plt.close(fig)

    (OUTDIR / 'report.txt').write_text('\n'.join(report) + '\n')
    log(f'\n[done] report + figures -> {OUTDIR}')


if __name__ == '__main__':
    main()
