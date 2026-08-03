#!/usr/bin/env python3
'''Remove the deep-slip contribution from the evaluated profiles.

Propagates the far-field AlTar posterior through to the near-field profiles:
draws N_POST posterior slip models, keeps only the DEEP patches (centroid
depth > DEEP_DEPTH_M; with layer interfaces at 0/1500/3000/5500 m this selects
triangles entirely below 3000 m -- slip in the 1500-3000 m layer is left in the
data, to be re-solved by the shallow 2D inversion), forward-predicts the
fault-parallel displacement of each sample at the binned profile points (one
batched TDE Green's-function call), and subtracts the ensemble MEAN prediction.

The residual covariance combines the two uncertainty sources:
    cov_total = cov_model (ensemble covariance of the deep predictions,
                           correlated along the profile)
              + sigma_noise^2 * I (data noise, estimated from the detrended
                                   far-field band of the residual)

Everything stays in the PHYSICAL frame of profile.displacements[0] (projection
onto +strike); the inversion script applies its own PARALLEL_SIGN.

Targets the archived tri01 run (330-patch mesh), so the matching fault is
reconstructed from the *_remesh.npz files -- for a fresh run on the current
mesh, load config.FAULT_PICKLE instead (see plot_altar_output.py).

In:  results/working/tmp/evaluated_profiles.pickle  (evaluate_profiles.py)
Out: results/working/tmp/deep_removed_profiles.pickle  (list of dicts)
     + per-profile diagnostic figures and a model-vs-noise summary figure.
'''
import pickle
import time
import warnings

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from codes import FaultTriangles, AltarOutput
import config

RUN    = config.ALTAR_DIR / 'outputs_16000_12000'
INPUTS = config.ALTAR_DIR / 'inputs'

DEEP_DEPTH_M = 2500.   # patches with centroid depth > this are removed
N_POST       = 1000    # posterior samples for the prediction ensemble
SEED         = 0

# binning before prediction/inversion (matches the old inversion script)
N_BINS          = 200
N_NEAR_BINS     = 100
NEAR_FAULT_DIST = 400.   # m

FAR_FIELD_DIST = 2000.   # m; |xs| beyond this = far-field band for the noise std

EVALUATED_PICKLE = config.TMP_DIR / 'evaluated_profiles.pickle'
OUT_PICKLE       = config.TMP_DIR / 'deep_removed_profiles.pickle'
FIG_DIR          = config.TMP_DIR / 'deep_removal'
FIG_DIR.mkdir(exist_ok=True)


####    LOADING    ####
def load_altar():
    '''The 330-patch fault + AlTar posterior of the archived tri01 run.'''
    def _remesh(path):
        z = np.load(path, allow_pickle=True)
        return FaultTriangles(z['vertices'], z['triangles'], layers=z['layers'],
                              name=str(z['name']))
    fault = FaultTriangles.merge([
        _remesh(config.FAULT_DIR / 'mainshock_fault_remesh.npz'),
        _remesh(config.FAULT_DIR / 'foreshock_fault_remesh.npz')])

    with h5py.File(INPUTS / 'gf.h5', 'r') as fh:
        G = fh['gf'][:]
    with h5py.File(INPUTS / 'data.h5', 'r') as fh:
        d = fh['data'][:]
        dslices = {k: slice(*fh['datasets'][k][:]) for k in fh['datasets']}
    ao = AltarOutput(RUN, fault, G=G, d=d, dataset_slices=dslices)
    ao.check_ordering()
    return fault, ao


def deep_slip_samples(fault, ao):
    '''(ss, ds) posterior sample matrices + (ss_mode, ds_mode) KDE-mode vectors,
    with shallow patches zeroed.'''
    deep = fault.depths > DEEP_DEPTH_M
    rng = np.random.default_rng(SEED)
    idx = rng.choice(ao.summ['ss'].shape[0], size=N_POST, replace=False)
    ss, ds = ao.summ['ss'][idx].copy(), ao.summ['ds'][idx].copy()
    ss[:, ~deep] = 0.
    ds[:, ~deep] = 0.
    ss_mode, ds_mode = ao.summ['ss_mode'].copy(), ao.summ['ds_mode'].copy()
    ss_mode[~deep] = 0.
    ds_mode[~deep] = 0.
    print(f'[deep] {int(deep.sum())}/{fault.n_patches} patches deeper than '
          f'{DEEP_DEPTH_M:.0f} m; {N_POST} posterior samples '
          f'(|SS| mode up to {np.abs(ss_mode).max():.2f} m)')
    return ss, ds, ss_mode, ds_mode


####    PROFILE GEOMETRY    ####
def bin_profile(p):
    '''Bin-average one evaluated profile -> (xs, parallel, normal), NaNs dropped.'''
    b = p.bin_average(n_bins=N_BINS, n_near_fault_bins=N_NEAR_BINS,
                      near_fault_dist=NEAR_FAULT_DIST)
    ok = np.isfinite(b.displacements[0])
    return b.xs[ok], b.displacements[0, ok], b.displacements[1, ok]


def profile_points(p, xs):
    '''UTM sample points + strike unit vector. xs = 0 sits on the strain-relocated
    fault (fault_utm_refined); the trace linestring runs -plen -> +plen.'''
    c = np.asarray(p.linestring.coords, dtype=float)
    u = (c[-1] - c[0]) / np.hypot(*(c[-1] - c[0]))
    pts = np.asarray(p.fault_utm_refined)[None, :] + np.asarray(xs)[:, None] * u
    theta = np.radians(p.strike)
    s_hat = np.array([np.sin(theta), np.cos(theta)])
    return pts, s_hat


####    DEEP-FIELD PREDICTION    ####
def deep_predictions(fault, ss, ds, ss_mode, ds_mode, prof_pts):
    '''Fault-parallel deep-slip predictions at each profile's points, for the
    whole posterior ensemble. One batched GF call; returns per profile
    (preds (N_POST, n), pred_mode (n,)).'''
    all_pts = np.vstack([pts for pts, _ in prof_pts])
    print(f'[deep] TDE Green\'s functions at {len(all_pts)} points x '
          f'{fault.n_patches} patches ...')
    t0 = time.time()
    fault = fault.compute_greens_functions(all_pts.T)   # (2, T, 3, n_pts)
    print(f'[deep] done in {time.time() - t0:.0f} s')

    out, k = [], 0
    for pts, s_hat in prof_pts:
        n = len(pts)
        # parallel-component GF matrices (T, n) for this profile's strike
        gss = np.tensordot(s_hat, fault.gfs[0][:, :2, k:k + n], axes=(0, 1))
        gds = np.tensordot(s_hat, fault.gfs[1][:, :2, k:k + n], axes=(0, 1))
        out.append((ss @ gss + ds @ gds, ss_mode @ gss + ds_mode @ gds))
        k += n
    return out


def noise_std(xs, resid):
    '''Data noise std from the detrended far-field band(s) of the residual.'''
    parts = []
    for side in (xs <= -FAR_FIELD_DIST, xs >= FAR_FIELD_DIST):
        r = resid[side]
        if r.size >= 10:
            t = xs[side]
            parts.append(r - np.polyval(np.polyfit(t, r, 1), t))
    if not parts:
        return float(np.std(resid))
    return float(np.std(np.concatenate(parts)))


####    FIGURES    ####
def plot_profile(rec, preds, path):
    xs = rec['xs']
    fig = plt.figure(figsize=(12, 8), layout='constrained')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.6, 1], figure=fig)
    fig.suptitle(f'Profile {rec["i"]}  (fault {rec["fault_id"]}, '
                 f'{rec["x_along_fault"] / 1000:.1f} km along strike)')

    # data, prediction ensemble, residual
    ax = fig.add_subplot(gs[0, :])
    ax.plot(xs, preds[:60].T, color='seagreen', lw=0.4, alpha=0.25)
    ax.plot(xs, rec['data_par'], color='0.6', lw=1.2, label='data (fault-parallel)')
    ax.plot(xs, rec['pred_mean'], color='seagreen', lw=1.6,
            label='deep model (posterior mean)')
    ax.plot(xs, rec['pred_mode'], color='darkorange', lw=1.2, ls='--',
            label='deep model (KDE mode)')
    ax.plot(xs, rec['resid'], color='crimson', lw=1.2, label='residual (data - mean)')
    ax.axvline(0, color='lightgray', ls='--')
    ax.set_xlabel('Distance from fault (m)')
    ax.set_ylabel('Displacement (m)')
    ax.legend(fontsize=8)

    # model uncertainty vs data noise along the profile
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(xs, rec['sigma_model'], color='seagreen',
            label=r'$\sigma_{model}$ (deep-slip posterior)')
    ax.axhline(rec['sigma_noise'], color='0.4', ls='--',
               label=rf'$\sigma_{{noise}}$ = {rec["sigma_noise"]:.3f} m')
    ax.axvline(0, color='lightgray', ls='--')
    ax.set_xlabel('Distance from fault (m)')
    ax.set_ylabel('sigma (m)')
    ax.legend(fontsize=8)

    # correlation structure of the total covariance
    ax = fig.add_subplot(gs[1, 1])
    sd = np.sqrt(np.diag(rec['cov_total']))
    im = ax.imshow(rec['cov_total'] / np.outer(sd, sd), cmap='RdBu_r',
                   vmin=-1, vmax=1)
    ax.set_title('corr(cov_total)', fontsize=9)
    ax.set_xlabel('bin index')
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_figure(records, path):
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for fid, marker in ((0, 'o'), (1, 's')):
        rs = [r for r in records if r['fault_id'] == fid]
        if not rs:
            continue
        km = [r['x_along_fault'] / 1000 for r in rs]
        ax.plot(km, [r['sigma_noise'] for r in rs], marker, color='0.4',
                label=f'fault {fid}: noise')
        ax.plot(km, [np.median(r['sigma_model']) for r in rs], marker,
                color='seagreen', label=f'fault {fid}: model (median)')
        ax.plot(km, [r['sigma_model'].max() for r in rs], marker,
                mfc='none', color='seagreen', label=f'fault {fid}: model (max)')
    ax.set_xlabel('Along-strike distance (km)')
    ax.set_ylabel('sigma (m)')
    ax.set_title('Deep-slip model uncertainty vs data noise, per profile')
    ax.legend(fontsize=8)
    ax.grid(True, ls=':', alpha=0.6)
    fig.savefig(path, dpi=200)
    plt.close(fig)


####    MAIN    ####
def main():
    profiles = pickle.load(open(EVALUATED_PICKLE, 'rb'))
    print(f'[load] {len(profiles)} evaluated profiles')
    fault, ao = load_altar()
    ss, ds, ss_mode, ds_mode = deep_slip_samples(fault, ao)

    # bin everything first so the GFs go in one batch
    binned, geoms = [], []
    for i, p in enumerate(profiles):
        xs, par, nrm = bin_profile(p)
        binned.append((i, p, xs, par, nrm))
        geoms.append(profile_points(p, xs))

    ensembles = deep_predictions(fault, ss, ds, ss_mode, ds_mode, geoms)

    records = []
    for (i, p, xs, par, nrm), (preds, pred_mode) in zip(binned, ensembles):
        pred_mean = preds.mean(axis=0)
        cov_model = np.cov(preds, rowvar=False)
        resid = par - pred_mean
        sig_n = noise_std(xs, resid)
        cov_total = cov_model + sig_n ** 2 * np.eye(len(xs))

        rec = dict(i=i, fault_id=p.fault_id, x_along_fault=p.x_along_fault,
                   strike=p.strike, xs=xs, data_par=par, data_nrm=nrm,
                   pred_mean=pred_mean, pred_mode=pred_mode, resid=resid,
                   sigma_model=np.sqrt(np.diag(cov_model)), sigma_noise=sig_n,
                   cov_model=cov_model, cov_total=cov_total)
        records.append(rec)
        plot_profile(rec, preds, FIG_DIR / f'profile_{i:02d}.png')

        # far-field step: the deep model should have the same sense as the data
        left, right = xs <= -FAR_FIELD_DIST, xs >= FAR_FIELD_DIST
        if left.sum() < 5 or right.sum() < 5:   # cropped profile: outermost bins
            left = np.arange(len(xs)) < 25
            right = np.arange(len(xs)) >= len(xs) - 25
        step_d = par[right].mean() - par[left].mean()
        step_m = pred_mean[right].mean() - pred_mean[left].mean()
        # only meaningful when the deep model actually predicts a step
        flag = '  <-- OPPOSITE SIGN' if step_d * step_m < 0 and abs(step_m) > 0.05 else ''
        print(f'[prof {i:02d}] f{p.fault_id} {p.x_along_fault / 1000:5.1f} km: '
              f'step data {step_d:+.2f} / deep model {step_m:+.2f} m | '
              f'sigma noise {sig_n:.3f}, model med {np.median(rec["sigma_model"]):.3f} '
              f'max {rec["sigma_model"].max():.3f} m{flag}')

    med_ratio = np.median([np.median(r['sigma_model']) / r['sigma_noise']
                           for r in records])
    print(f'\n[summary] median sigma_model/sigma_noise = {med_ratio:.2f} '
          f'(model uncertainty {"dominates" if med_ratio > 1 else "is below data noise"})')

    summary_figure(records, FIG_DIR / 'summary_model_vs_noise.png')
    pickle.dump(records, open(OUT_PICKLE, 'wb'))
    print(f'[done] {len(records)} records -> {OUT_PICKLE}\n'
          f'       figures -> {FIG_DIR}')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main()
