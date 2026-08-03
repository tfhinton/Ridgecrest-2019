#!/usr/bin/env python3
'''Hamiltonian inversion of the deep-removed 2D profiles (workflow step 9).

For each record from remove_deep_slip_contribution.py, invert the deep-corrected
fault-parallel residual with the 2D damage-zone forward model (TwoDDzForwardModel,
now with dipping-fault support -- validated in test_twod_dz_forward.py) for:
    shallow strike-slip on 3 depth patches (0-500 / 500-1500 / 1500-3000 m),
    damage-zone half-width, modulus ratio, and a datum offset.
--sheath swaps in TwoDDzSheathForwardModel: same parameters, but the damage
zone becomes a perpendicular-width sheath that dips with the fault (solved via
a boundary-integral / equivalent-density method) rather than a fixed vertical
column -- see that class's docstring and test_twod_dz_sheath_forward.py.

The local shallow dip is taken from the FaultTriangles mesh (config.FAULT_PICKLE):
area-weighted mean normal of the triangles within DIP_RADIUS of the profile, per
mesh layer (layer 0 = 0-1500 m, layer 1 = 1500-3000 m), projected into the
profile frame to give the horizontal offset of each patch interface.

Sign: the data are flipped (per profile) so the far-field step is positive,
matching positive slip in the forward model; the applied sign is stored in the
result pickle.

Results are pickled per profile as they finish (results/profile_inversion/twod01/),
so a run can be interrupted/resumed. Chains sample in parallel (one core each);
for the cluster, run one profile per job with --profile i and up the sampler
settings, e.g.  python invert_twod_profiles.py --profile 12 --draws 2000 --tune 1000 --chains 8

Usage:
    invert_twod_profiles.py               # all profiles
    invert_twod_profiles.py --profile 12  # just record i=12
    invert_twod_profiles.py --list        # print profiles + local dips, no inversion
'''
import argparse
import os
import pickle
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

from codes import (TwoDDzForwardModel, TwoDDzSheathForwardModel, UniformDist,
                    GaussianDist, HamiltonianInversion)
import config
from result_io import load_result

####    PARAMETERS    ####
VD = np.array([0., 500., 1500., 3000.])       # 2D patch depth interfaces (m)
SLIP_LABELS = [f'slip{i}' for i in range(len(VD) - 1)]

MESH_INTERFACES = [0., 1500., 3000.]          # top two mesh layer interfaces
DIP_RADIUS = 3000.                            # m, radius for the local dip estimate
DIP_PRIOR_SIGMA = 8.                          # deg, --free-dip Gaussian prior width

FAR_FIELD_DIST = 2000.                        # m, band for the sign check

# sampler defaults (CLI-overridable); generous -- sized for an unattended
# cluster run (one chain per core, 8 cores)
DRAWS, TUNE, CHAINS = 4000, 2000, 8
TIMEOUT_MINUTES = 360

RECORDS_PICKLE   = config.TMP_DIR / 'deep_removed_profiles.pickle'
EVALUATED_PICKLE = config.TMP_DIR / 'evaluated_profiles.pickle'


####    FORWARD MODEL    ####
class ProfileModelMixin:
    '''pred_func parameter order given by self.param_labels (set per-run in
    invert_record, since --fix-mr / --free-dip change which parameters are free).
    fixed_mr: constant modulus_ratio to use when 'modulus_ratio' is not a free param.
    vd / dip_signs: needed to rebuild patch geometry when dip0/dip1 ARE free params;
    otherwise the patches built once in invert_record are reused unchanged (cheap).
    Shared by ProfileModel (TwoDDzForwardModel: fixed vertical damage zone) and
    SheathProfileModel (TwoDDzSheathForwardModel: perpendicular-width sheath
    that dips with the fault, see --sheath) -- pred_func itself doesn't care
    which forward-model physics its base class provides.'''
    param_labels = None
    fixed_mr = None
    vd = None
    dip_signs = None

    def pred_func(self, p):
        d = dict(zip(self.param_labels, p))
        _self = self._copy()
        if 'dip0' in d:
            x_off = dip_to_offsets(self.vd, [d['dip0'], d['dip1']], self.dip_signs)
            _self = _self.build_dipping_patches(self.vd, x_off)
        _self.dz_half_width = d['dz_halfwidth']
        _self.modulus_ratio = self.fixed_mr if self.fixed_mr is not None else d['modulus_ratio']
        _self.slips = np.array([d[l] for l in SLIP_LABELS])
        return _self.run(_self.xs).sol + d['offset']


class ProfileModel(ProfileModelMixin, TwoDDzForwardModel):
    pass


class SheathProfileModel(ProfileModelMixin, TwoDDzSheathForwardModel):
    pass


def make_priors(args, dips_deg):
    '''Prior list, order matching ProfileModel.pred_func's expected param order.
    --fix-mr drops modulus_ratio from the free parameters entirely (see ProfileModel);
    --free-dip appends dip0/dip1 (Gaussian, centred on the mesh-derived local dip).'''
    priors = [UniformDist('dz_halfwidth', 0., 2500.)]
    if args.fix_mr is None:
        priors.append(UniformDist('modulus_ratio', args.mr_lower, 0.9))
    priors.append(UniformDist('offset', -3., 3.))
    priors += [UniformDist(l, 0., 10.) for l in SLIP_LABELS]
    if args.free_dip:
        priors += [GaussianDist(f'dip{k}', dips_deg[k], DIP_PRIOR_SIGMA)
                   for k in range(len(dips_deg))]
    return priors


def dispersed_initvals(priors, chains, rng):
    '''One dict of prior-drawn starting values per chain, for --dispersed-init.
    PyMC's default init ('jitter+adapt_diag') starts every chain near the same point
    with small jitter, so with a genuinely multimodal posterior (the dz/modulus_ratio
    trade-off seen on several profiles) which mode a chain finds is mostly luck. Drawing
    each chain's start from the full prior support instead gives many independent shots
    at finding every mode.'''
    out = []
    for _ in range(chains):
        vals = {}
        for p in priors:
            vals[p.label] = (rng.uniform(p.lower, p.upper) if isinstance(p, UniformDist)
                             else rng.normal(p.mu, p.sigma))
        out.append(vals)
    return out


####    GEOMETRY    ####
def dip_to_offsets(vd, dips_deg, signs):
    '''Piecewise-linear horizontal offset of the fault at each vd interface, given a dip
    MAGNITUDE (deg) and a lean sign per mesh layer -- the inverse of the slope->dip
    conversion in local_dip(), used by ProfileModel.pred_func when --free-dip lets dip
    vary. Clipped away from 0/90 deg to avoid the tan() singularity; with an 8 deg-sigma
    prior centred 60-85 deg from the mesh estimate this clip is never approached.'''
    slopes = [signs[k] / np.tan(np.radians(np.clip(dips_deg[k], 5., 89.9)))
              for k in range(len(dips_deg))]
    x_of = lambda z: (slopes[0] * min(z, MESH_INTERFACES[1])
                      + slopes[1] * max(z - MESH_INTERFACES[1], 0.))
    return np.array([x_of(z) for z in vd])


def local_dip(fault, profile, fault_id):
    '''Per-mesh-layer local dip near the profile -> horizontal offset of the fault
    at each VD interface, in the profile frame (+xs). Returns (x_offsets, dips_deg,
    signs) -- signs is the lean direction per mesh layer, needed by dip_to_offsets.'''
    c = np.asarray(profile.linestring.coords, dtype=float)
    u_hat = (c[-1] - c[0]) / np.hypot(*(c[-1] - c[0]))   # +xs direction (E, N)
    p0 = np.asarray(profile.fault_utm_refined, dtype=float)

    cen = fault.centroids
    near = ((fault.fault_ids == fault_id)
            & (np.hypot(cen[:, 0] - p0[0], cen[:, 1] - p0[1]) < DIP_RADIUS))

    # triangle normals (z-up), oriented consistently toward the +xs side; the
    # cross-product norm is 2x area, so summing raw normals area-weights them
    v = fault.vertices[fault.triangles]
    nrm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    flip = nrm[:, :2] @ u_hat < 0.
    nrm[flip] *= -1.

    slopes, dips, signs = [], [], []
    for lay in range(len(MESH_INTERFACES) - 1):
        sel = near & (fault.layers == lay)
        if sel.sum() == 0:                    # off the meshed extent: vertical
            slopes.append(0.); dips.append(90.); signs.append(1.)
            continue
        n = nrm[sel].sum(axis=0)
        n /= np.linalg.norm(n)
        nh, nz = n[:2], n[2]
        # downdip horizontal advance per metre depth = n_h * nz / |n_h|^2
        slope = float(nh @ u_hat * nz / (nh @ nh))
        slopes.append(slope)
        dips.append(float(np.degrees(np.arctan2(np.hypot(*nh), abs(nz)))))
        signs.append(1. if slope >= 0. else -1.)

    x_of = lambda z: (slopes[0] * min(z, MESH_INTERFACES[1])
                      + slopes[1] * max(z - MESH_INTERFACES[1], 0.))
    return np.array([x_of(z) for z in VD]), dips, signs


####    DATA    ####
def prep_data(rec):
    '''Sign-flip the deep-removed residual so its far-field step is positive
    (positive slip in the model = up-step). Returns (data, sign, step).'''
    xs, resid = rec['xs'], rec['resid']
    left, right = xs <= -FAR_FIELD_DIST, xs >= FAR_FIELD_DIST
    if left.sum() < 5 or right.sum() < 5:     # cropped profile: outermost bins
        left = np.arange(len(xs)) < 25
        right = np.arange(len(xs)) >= len(xs) - 25
    step = resid[right].mean() - resid[left].mean()
    sign = 1. if step >= 0. else -1.
    return sign * resid, sign, step


####    CONVERGENCE DIAGNOSTICS    ####
def split_rhat(x):
    '''Split-Rhat from (chain, draw) samples; arviz-version-proof.'''
    c, d = x.shape
    half = d // 2
    xs = x[:, :2*half].reshape(2*c, half)
    W = xs.var(axis=1, ddof=1).mean()
    B = half * xs.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((half - 1)/half * W + B/half) / W))


def ess_bulk(x):
    '''Crude bulk ESS: n_eff per chain from the autocorrelation (Geyer initial
    positive sequence), summed over chains.'''
    c, d = x.shape
    total = 0.
    for ch in range(c):
        v = x[ch] - x[ch].mean()
        acf = np.correlate(v, v, 'full')[d-1:] / (np.arange(d, 0, -1) * v.var() + 1e-30)
        s, t = 0., 1
        while t + 1 < d:
            pair = acf[t] + acf[t+1]
            if pair < 0:
                break
            s += pair
            t += 2
        total += d / (1. + 2.*s)
    return float(total)


def convergence_stats(idata, param_labels, tag=''):
    '''Per-parameter rhat / ESS / per-chain means; prints a table and returns the
    dict. Divergent per-chain means = the chains are stuck in different modes.'''
    stats, poor = {}, []
    print(f'[conv] {tag}: {"param":15s} {"rhat":>6s} {"ess":>7s}   chain means')
    for l in param_labels:
        x = idata.posterior[l].values
        stats[l] = dict(rhat=split_rhat(x), ess=ess_bulk(x),
                        chain_means=x.mean(axis=1))
        flag = ''
        if stats[l]['rhat'] > 1.05 or stats[l]['ess'] < 400:
            flag = '  <-- POOR'
            poor.append(l)
        cm = ' '.join(f'{v:8.2f}' for v in stats[l]['chain_means'])
        print(f'[conv] {tag}: {l:15s} {stats[l]["rhat"]:6.3f} '
              f'{stats[l]["ess"]:7.0f}   {cm}{flag}')
    stats['poor'] = poor
    return stats


####    HELPERS    ####
def kde_mode(samples):
    samples = np.asarray(samples, dtype=float)
    if np.ptp(samples) == 0.:
        return float(samples[0])
    try:
        grid = np.linspace(samples.min(), samples.max(), 512)
        return float(grid[np.argmax(gaussian_kde(samples)(grid))])
    except Exception:
        return float(np.median(samples))


def posterior_summary(idata, param_labels):
    out = {}
    for l in param_labels:
        s = idata.posterior[l].values.flatten()
        out[l] = {'map': kde_mode(s), 'med': float(np.median(s)),
                  'lo': float(np.percentile(s, 16)), 'hi': float(np.percentile(s, 84))}
    return out


def best_sample(idata, model, data, cov, param_labels, max_eval=4000):
    '''The maximum-likelihood posterior sample: a coherent parameter vector even
    when the posterior is multimodal (marginal medians are not).'''
    from scipy.linalg import cho_factor, cho_solve
    samp = np.column_stack([idata.posterior[l].values.reshape(-1)
                            for l in param_labels])
    if len(samp) > max_eval:
        samp = samp[np.linspace(0, len(samp) - 1, max_eval).astype(int)]
    cf = cho_factor(cov)
    chi2 = np.array([r @ cho_solve(cf, r) for r in
                     (model.pred_func(p) - data for p in samp)])
    return dict(zip(param_labels, samp[np.argmin(chi2)]))


####    FIGURES    ####
def plot_chain_modes(idata, param_labels, path):
    '''Per-chain posterior means for dz_halfwidth vs. modulus_ratio (or vs. offset if
    --fix-mr held modulus_ratio fixed) -- one point per chain. Meant for --dispersed-init
    runs with many chains: a single rhat number says "not converged" but this scatter
    shows HOW -- e.g. two clusters of chains stuck in different modes -- and roughly how
    much of the chain population (a proxy for posterior mass, modulo mixing-time bias)
    each mode captured.'''
    post = idata.posterior
    ylab = 'modulus_ratio' if 'modulus_ratio' in param_labels else 'offset'
    x = post['dz_halfwidth'].values.mean(axis=1)
    y = post[ylab].values.mean(axis=1)
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    ax.scatter(x, y, s=25, alpha=0.6, color='steelblue', edgecolor='k', linewidth=0.3)
    ax.set_xlabel('dz_halfwidth (m, per-chain mean)')
    ax.set_ylabel(f'{ylab} (per-chain mean)')
    ax.set_title(f'{len(x)} chains, dispersed init: mode occupancy')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_inversion(rec, model, data, idata, summ, best, param_labels, path, title):
    post = idata.posterior
    xs = rec['xs']

    fig = plt.figure(figsize=(13, 8), layout='constrained')
    gs = gridspec.GridSpec(2, len(param_labels), height_ratios=[2, 1], figure=fig)
    fig.suptitle(title)

    # slip vs depth (best sample + marginal 16-84)
    ax = fig.add_subplot(gs[0, 0])
    for i, l in enumerate(SLIP_LABELS):
        zs = [VD[i], VD[i+1]]
        ax.fill_betweenx(zs, [summ[l]['lo']]*2, [summ[l]['hi']]*2,
                         color='steelblue', alpha=0.3)
        ax.plot([best[l]]*2, zs, color='navy')
    ax.axvline(0, color='lightgray', ls='--')
    ax.invert_yaxis()
    ax.set_xlabel('Slip (m, data sense)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Shallow slip (best sample)', fontsize=10)

    # data + fit; posterior draws as thin lines
    ax = fig.add_subplot(gs[0, 1:])
    ax.axvspan(-best['dz_halfwidth'], best['dz_halfwidth'], color='wheat',
               alpha=0.4, label=f'damage zone ({best["dz_halfwidth"]:.0f} m)')
    samp = {l: post[l].values.reshape(-1) for l in param_labels}
    for j in np.linspace(0, len(samp[param_labels[0]]) - 1, 30).astype(int):
        p = [samp[l][j] for l in param_labels]
        ax.plot(xs, model.pred_func(np.array(p)), color='crimson', lw=0.3, alpha=0.25)
    sig = np.sqrt(np.diag(rec['cov_total']))
    ax.errorbar(xs, data, yerr=sig, fmt='.', ms=3, color='0.45', ecolor='0.8',
                elinewidth=0.6, label='data (deep-removed residual)', zorder=2)
    ax.plot(xs, model.pred_func(np.array([best[l] for l in param_labels])),
            color='crimson', lw=1.8, label='model (best posterior sample)', zorder=3)
    ax.axvline(0, color='lightgray', ls='--')
    ax.set_xlabel('Distance from fault (m)')
    ax.set_ylabel('Fault-parallel displacement (m)')
    ax.legend(fontsize=8)

    # posterior histograms
    for k, l in enumerate(param_labels):
        ax = fig.add_subplot(gs[1, k])
        s = post[l].values.flatten()
        ax.hist(s, bins=30, color='steelblue', alpha=0.8)
        ax.axvline(best[l], color='crimson', lw=1.2)
        ax.axvline(summ[l]['lo'], color='k', ls=':', lw=0.8)
        ax.axvline(summ[l]['hi'], color='k', ls=':', lw=0.8)
        ax.set_title(l, fontsize=9)
        ax.set_yticks([])

    fig.savefig(path, dpi=180)
    plt.close(fig)


def summary_plot(results, path):
    # whatever params this outtag's runs actually inferred (varies with --fix-mr /
    # --free-dip), skipping the 'offset' nuisance datum
    params = [l for l in results[0]['param_labels'] if l != 'offset']
    fig, axes = plt.subplots(len(params), 1, figsize=(9, 2.2 * len(params)),
                             sharex=True, layout='constrained')
    for ax, prm in zip(np.atleast_1d(axes), params):
        for fid, (marker, color) in ((0, ('o', 'steelblue')), (1, ('s', 'darkorange'))):
            rs = [r for r in results if r['fault_id'] == fid]
            if not rs:
                continue
            km = np.array([r['x_along_fault'] / 1000. for r in rs])
            m = np.array([r['summary'][prm]['map'] for r in rs])
            lo = np.array([r['summary'][prm]['lo'] for r in rs])
            hi = np.array([r['summary'][prm]['hi'] for r in rs])
            errs = np.clip(np.vstack([m - lo, hi - m]), 0., None)
            ax.errorbar(km, m, yerr=errs, fmt=marker, color=color, ecolor='black',
                        elinewidth=1., capsize=2.5, ms=5,
                        label=f'fault {fid}', zorder=5)
        ax.set_ylabel(prm)
        ax.grid(True, ls=':', alpha=0.6)
    np.atleast_1d(axes)[0].set_title('2D inversion along strike (mode, 16-84th pct)')
    np.atleast_1d(axes)[0].legend(fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel('Along-strike distance (km)')
    fig.savefig(path, dpi=200)
    plt.close(fig)


####    MAIN    ####
def invert_record(rec, profile, fault, args, resdir, figdir):
    i = rec['i']
    tag = f'profile {i:02d} (f{rec["fault_id"]}, {rec["x_along_fault"]/1000:.1f} km)'

    x_off, dips, signs = local_dip(fault, profile, rec['fault_id'])
    data, sign, step = prep_data(rec)
    print(f'[inv] {tag}: dips {dips[0]:.0f}/{dips[1]:.0f} deg, interface offsets '
          f'{np.round(x_off).astype(int)} m, step {step:+.2f} m -> sign {sign:+.0f}')

    xs = np.asarray(rec['xs'], dtype=float)
    model_cls = SheathProfileModel if args.sheath else ProfileModel
    model = model_cls().build_dipping_patches(VD, x_off)
    model.xs = xs
    model.fixed_mr = args.fix_mr
    model.vd = VD
    model.dip_signs = signs

    priors = make_priors(args, dips)
    param_labels = [p.label for p in priors]
    model.param_labels = param_labels

    inv = HamiltonianInversion(model, priors, data, rec['cov_total'])
    init_mode, initvals = 'auto', None
    if args.dispersed_init:
        rng = np.random.default_rng(args.seed + i)
        initvals = dispersed_initvals(priors, args.chains, rng)
        init_mode = 'adapt_diag'   # skip PyMC's own jitter -- we've already dispersed
    inv = inv.run(draws=args.draws, tune=args.tune, chains=args.chains,
                  cores=min(args.chains, os.cpu_count() or 1),
                  timeout_minutes=args.timeout, target_accept=args.target_accept,
                  init=init_mode, initvals=initvals)

    summ = posterior_summary(inv.result, param_labels)
    best = best_sample(inv.result, model, data, rec['cov_total'], param_labels)
    stats = convergence_stats(inv.result, param_labels, tag=f'profile {i:02d}')
    out = dict(i=i, fault_id=rec['fault_id'], x_along_fault=rec['x_along_fault'],
               strike=rec['strike'], sign=sign, vd=VD, x_offsets=x_off,
               dips_deg=dips, param_labels=param_labels, xs=xs, data=data,
               sigma=np.sqrt(np.diag(rec['cov_total'])),
               summary=summ, best=best, stats=stats, idata=inv.result,
               sampler=dict(draws=args.draws, tune=args.tune, chains=args.chains,
                            target_accept=args.target_accept, fix_mr=args.fix_mr,
                            mr_lower=args.mr_lower, free_dip=args.free_dip,
                            dispersed_init=args.dispersed_init))
    with open(resdir / f'profile_{i:02d}.pickle', 'wb') as f:
        pickle.dump(out, f)

    plot_inversion(rec, model, data, inv.result, summ, best, param_labels,
                   figdir / f'profile_{i:02d}.png',
                   f'Profile {i} (fault {rec["fault_id"]}, '
                   f'{rec["x_along_fault"]/1000:.1f} km along strike, '
                   f'dip {dips[0]:.0f}/{dips[1]:.0f} deg, sign {sign:+.0f})')
    if args.chains > 16:
        plot_chain_modes(inv.result, param_labels,
                         figdir / f'profile_{i:02d}_chainmodes.png')
    mr_str = f'{best["modulus_ratio"]:.2f}' if 'modulus_ratio' in best else f'fixed {args.fix_mr:.2f}'
    print(f'[inv] {tag}: saved pickle + figure  '
          f'(best dz {best["dz_halfwidth"]:.0f} m, mr {mr_str}, '
          f'surface slip {best["slip0"]:.2f} m)')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', type=int, default=None,
                    help='invert only the record with this index i')
    ap.add_argument('--draws', type=int, default=DRAWS)
    ap.add_argument('--tune', type=int, default=TUNE)
    ap.add_argument('--chains', type=int, default=CHAINS)
    ap.add_argument('--timeout', type=float, default=TIMEOUT_MINUTES)
    ap.add_argument('--target-accept', type=float, default=0.9,
                    help='NUTS target_accept (raise towards e.g. 0.95-0.99 for '
                         'tricky/multimodal posteriors, at the cost of speed)')
    ap.add_argument('--mr-lower', type=float, default=0.2,
                    help='lower bound of the modulus_ratio prior (ignored with --fix-mr)')
    ap.add_argument('--fix-mr', type=float, default=None,
                    help='hold modulus_ratio fixed at this value instead of inferring it')
    ap.add_argument('--free-dip', action='store_true',
                    help=f'let dip vary per mesh layer as a free parameter (Gaussian '
                         f'prior, mesh estimate +/- {DIP_PRIOR_SIGMA:g} deg) instead of '
                         f'fixing it from the mesh estimate')
    ap.add_argument('--sheath', action='store_true',
                    help='use TwoDDzSheathForwardModel instead of TwoDDzForwardModel: '
                         'the damage zone becomes a perpendicular-width sheath that dips '
                         'with the fault (boundary-integral solve), rather than a fixed '
                         'vertical column. dz_halfwidth/modulus_ratio/priors are '
                         'unchanged -- see TwoDDzSheathForwardModel and '
                         'test_twod_dz_sheath_forward.py for the method and its validation')
    ap.add_argument('--dispersed-init', action='store_true',
                    help='start each chain from an independent prior draw instead of '
                         "PyMC's small-jitter default; combine with a large --chains "
                         'to probe multimodality (see dispersed_initvals docstring)')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed for --dispersed-init')
    ap.add_argument('--outtag', default='twod01',
                    help='output subdir under results/profile_inversion/ (default '
                         'twod01; use a distinct tag for experiment re-runs so they '
                         "don't overwrite the production results)")
    ap.add_argument('--vd', type=str, default=None,
                    help='comma-separated depth interfaces (m) for the slip patches, '
                         'e.g. "0,100,350,800,1700,3000" for 5 patches (default: '
                         '0,500,1500,3000, i.e. 3 patches)')
    ap.add_argument('--list', action='store_true',
                    help='print the profiles + local dips and exit')
    ap.add_argument('--skip-done', action='store_true',
                    help='skip records whose result pickle already exists (resume)')
    args = ap.parse_args()

    if args.vd is not None:
        # VD/SLIP_LABELS are read as module globals throughout (local_dip,
        # make_priors, ProfileModel.pred_func, plot_inversion's slip panel) --
        # reassigning them here before anything else runs is the minimal way to
        # thread a custom discretisation through without passing vd everywhere
        global VD, SLIP_LABELS
        VD = np.array([float(x) for x in args.vd.split(',')])
        SLIP_LABELS = [f'slip{i}' for i in range(len(VD) - 1)]
        print(f'[load] custom discretisation: {len(SLIP_LABELS)} slip patches, '
              f'VD={VD.tolist()}')

    resdir = config.RESULTS_DIR / 'profile_inversion' / args.outtag
    figdir = resdir / 'figs'
    resdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(exist_ok=True)

    records = pickle.load(open(RECORDS_PICKLE, 'rb'))
    profiles = pickle.load(open(EVALUATED_PICKLE, 'rb'))
    fault = pickle.load(open(config.FAULT_PICKLE, 'rb'))
    print(f'[load] {len(records)} records, fault mesh {fault.n_patches} patches')

    if args.list:
        for rec in records:
            x_off, dips, _ = local_dip(fault, profiles[rec['i']], rec['fault_id'])
            _, sign, step = prep_data(rec)
            print(f'  {rec["i"]:3d}  f{rec["fault_id"]}  '
                  f'{rec["x_along_fault"]/1000:6.1f} km  '
                  f'dips {dips[0]:5.1f}/{dips[1]:5.1f} deg  '
                  f'offsets {np.round(x_off).astype(int)} m  '
                  f'step {step:+.2f} m (sign {sign:+.0f})')
        return

    if args.profile is not None:
        records = [r for r in records if r['i'] == args.profile]
        if not records:
            raise SystemExit(f'no record with i={args.profile}')

    for n, rec in enumerate(records, 1):
        if args.skip_done and (resdir / f'profile_{rec["i"]:02d}.pickle').exists():
            print(f'[inv] profile {rec["i"]:02d} already done, skipping')
            continue
        print(f'\n=== {n}/{len(records)} ===')
        t0 = time.time()
        try:
            invert_record(rec, profiles[rec['i']], fault, args, resdir, figdir)
            print(f'[inv] done in {(time.time()-t0)/60:.1f} min')
        except (Exception, KeyboardInterrupt) as e:
            print(f'[inv] profile {rec["i"]} FAILED: {type(e).__name__}: {e}')
            if isinstance(e, KeyboardInterrupt):
                break

    # along-strike summary from everything inverted so far
    results = []
    for f in sorted(resdir.glob('profile_*.pickle')):
        results.append(load_result(f))
    if len(results) > 1:
        summary_plot(results, figdir / 'summary_along_strike.png')
        print(f'\n[done] {len(results)} results; summary -> '
              f'{figdir / "summary_along_strike.png"}')

    # convergence recap
    poor = [(r['i'], r['stats']['poor']) for r in results
            if r.get('stats', {}).get('poor')]
    if poor:
        print('\n[conv] profiles with poorly converged parameters '
              '(rhat > 1.05 or ess < 400):')
        for i, params in poor:
            print(f'[conv]   profile {i:02d}: {", ".join(params)}')
    elif results:
        print('\n[conv] all inverted profiles converged cleanly.')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main()
