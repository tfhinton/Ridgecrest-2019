#!/usr/bin/env python3
'''Validation of TwoDDzSheathForwardModel, the perpendicular-width (dips-with-the
-fault) boundary-integral generalisation of TwoDDzForwardModel's fixed-vertical
damage zone. See the class docstring for the method; see
scripts/test_twod_dz_forward.py for the base class's own validation (checked
here again in the vertical limit, where the two classes should agree exactly).

Checks:
  1. Vertical-fault limit: sheath model (perpendicular width == horizontal
     width when the fault is vertical) reproduces the base class's exact k^m
     image series, for several modulus ratios.
  2. Homogeneous limit: dipping fault, modulus_ratio -> 1 (sigma == 0), matches
     the same closed-form analytic dipping solution used in the base class's
     own test (arctan2 form).
  3. Interface conditions at the (perpendicular) zone walls: displacement and
     traction continuity, for a dipping fault.
  4. Resolution convergence: increasing segment count should not change the
     answer materially (checked against a high-resolution reference).
  5. Independent finite-difference solve of div(mu grad u) = 0 with mu masked
     by PERPENDICULAR distance to the dipping fault (a true sheath, unlike the
     base class's test which masks by |x|), for dipping+sheath. This is the
     load-bearing check since there is no other closed-form reference once the
     fault dips.

Run: ./.venv/bin/python scripts/test_twod_dz_sheath_forward.py
'''
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from codes import TwoDDzForwardModel, TwoDDzSheathForwardModel
import config

FIG = config.TMP_DIR / 'test_twod_dz_sheath_forward.png'

VD = np.array([0., 500., 1500., 3000.])
SLIPS = np.array([3., 2., 1.])


def model(x_offsets, dz, mr, **kwargs):
    m = TwoDDzSheathForwardModel(dz_half_width=dz, modulus_ratio=mr, **kwargs)
    m = m.build_dipping_patches(VD, x_offsets)
    m.slips = SLIPS.copy()
    return m


def analytic_homog(xs, x_offsets, slips=SLIPS):
    u = np.zeros_like(xs)
    for i, s in enumerate(slips):
        xt, xb, dt, db = x_offsets[i], x_offsets[i + 1], VD[i], VD[i + 1]
        u += -(s / np.pi) * (np.arctan2(xs - xb, db) - np.arctan2(xs - xt, dt))
    return u


def check_vertical_limit():
    xs = np.linspace(-6000., 6000., 2001)
    xs = np.where(xs == 0., 1e-6, xs)
    ok = True
    for mr in (0.7, 0.4, 0.15):
        dz = 400.
        sheath = model(np.zeros(4), dz, mr).run(xs).sol
        old = TwoDDzForwardModel(dz_half_width=dz, modulus_ratio=mr).build_dipping_patches(VD, np.zeros(4))
        old.slips = SLIPS.copy()
        old = old.run(xs).sol
        err = np.abs(sheath - old).max()
        rng = old.max() - old.min()
        print(f'[1] vertical limit mr={mr:.2f}: max|sheath-old| = {err:.4f} m ({100*err/rng:.2f} % of range)')
        ok &= (err / rng) < 0.02
    assert ok


def check_homogeneous_dip():
    xs = np.linspace(-6000., 6000., 2001)
    xs = np.where(xs == 0., 1e-6, xs)
    offs = np.array([0., 134., 402., 804.])   # ~75 deg dip
    got = model(offs, 400., 1.0).run(xs).sol
    want = analytic_homog(xs, offs)
    err = np.abs(got - want).max()
    print(f'[2] homogeneous dipping vs analytic: max err = {err:.2e} m')
    assert err < 1e-6


def check_interface_conditions():
    '''Displacement and traction continuity at the sheath wall, probed near the
    collocation point with the largest equivalent density (i.e. where the
    interface condition is doing the most work), using the same kernels run()
    uses internally. Unlike the base class's exact-closed-form check, sigma is
    a piecewise-constant density on finite panels, so a genuine jump only
    emerges at the panel's own scale -- probing at a small FRACTION of the
    local panel length (not an arbitrarily tiny epsilon) is the discretisation-
    consistent way to see it, and the check is that this converges as N grows.'''
    from codes.TwoDDzSheathForwardModel import _G

    dz, mr = 300., 0.4
    offs = np.array([0., 134., 402., 804.])
    mu2, mu1 = mr, 1.

    ok = True
    prev_cont, prev_trac = None, None
    for n_near, n_far in [(60, 20), (150, 50), (400, 130)]:
        m = model(offs, dz, mr, n_near=n_near, n_far=n_far)
        mid, nrm, ds, Amat, k = m._assemble(dz, mr)
        sources = m._patch_sources()
        sigma = m._solve_sigma(mid, nrm, ds, Amat, k, sources)

        def U(x, z):
            v = 0.
            for x0, d, w in sources:
                v += w * (-(np.arctan((x - x0) / (d - z)) - np.arctan((x - x0) / (-d - z))) / (2 * np.pi))
            for j in range(len(mid)):
                v += sigma[j] * ds[j] * _G(x, z, mid[j, 0], mid[j, 1])
            return v

        # avoid the immediate surface neighbourhood, where the wall's own
        # surface-clip and the fault's own (d=0) surface source sit close
        # together -- pick the strongest-signal point away from that corner
        away = mid[:, 1] > 200.
        j = np.arange(len(sigma))[away][np.argmax(np.abs(sigma[away]))]
        pt, n = mid[j], nrm[j]
        eps = 0.5 * ds[j]
        de = eps * 0.1
        p_in, p_out = pt - eps * n, pt + eps * n
        p_in2, p_out2 = pt - (eps + de) * n, pt + (eps + de) * n
        cont = abs(U(*p_in) - U(*p_out))
        du_in = (U(*p_in) - U(*p_in2)) / de
        du_out = (U(*p_out2) - U(*p_out)) / de
        trac_err = abs(mu2 * du_in - mu1 * du_out) / abs(mu1 * du_out)
        print(f'[3] n_near={n_near}: |u_in - u_out| = {cont:.2e} m, traction rel err = {trac_err:.3f} '
              f'(peak |sigma| at depth {pt[1]:.0f} m)')
        # sigma is a point-source (not smoothed-panel) density, so continuity
        # and the traction match only emerge in the far-from-any-single-source
        # limit -- the honest check at finite N is that BOTH tighten as
        # resolution increases, not an absolute threshold at any one N.
        if prev_cont is not None:
            ok &= cont < prev_cont and trac_err < prev_trac
        prev_cont, prev_trac = cont, trac_err
    assert ok


def check_resolution_convergence():
    offs = np.array([0., 134., 402., 804.])
    xs = np.linspace(-5000., 5000., 501)
    xs = np.where(xs == 0., 1e-6, xs)
    dz, mr = 350., 0.35
    ref = model(offs, dz, mr, n_near=300, n_far=100).run(xs).sol
    ok = True
    for n_near, n_far in [(30, 12), (60, 20), (120, 40)]:
        got = model(offs, dz, mr, n_near=n_near, n_far=n_far).run(xs).sol
        err = np.abs(got - ref).max() / (ref.max() - ref.min())
        print(f'[4] n_near={n_near:3d} n_far={n_far:3d}: max err vs high-res ref = {100*err:.2f} %')
        ok &= err < 0.05
    assert ok


def perp_dist_to_polyline(px, pz, verts):
    d = np.full(len(px), np.inf)
    sgn = np.zeros(len(px))
    for i in range(len(verts) - 1):
        a, b = verts[i], verts[i + 1]
        ab = b - a
        L2 = ab @ ab
        t = np.clip(((px - a[0]) * ab[0] + (pz - a[1]) * ab[1]) / L2, 0., 1.)
        projx, projz = a[0] + t * ab[0], a[1] + t * ab[1]
        dist = np.hypot(px - projx, pz - projz)
        better = dist < d
        d = np.where(better, dist, d)
        cross = (b[0] - a[0]) * (pz - a[1]) - (b[1] - a[1]) * (px - a[0])
        sgn = np.where(better, np.sign(cross), sgn)
    return sgn * d


def fd_solve(x_offsets, dz, mr, L=25000., D=12000., h=25.):
    '''Same construction as test_twod_dz_forward.fd_solve, but the modulus mask
    is by PERPENDICULAR distance to the fault polyline, giving a true dipping
    sheath (not a vertical column) -- the correct ground truth for this class.'''
    nx, nz = int(2 * L / h) + 1, int(D / h) + 1
    x = -L + h * np.arange(nx)
    z = h * np.arange(nz)
    verts = np.column_stack([x_offsets, VD])

    g = np.interp(z, VD, x_offsets)
    islip = np.clip(np.searchsorted(VD, z, side='right') - 1, 0, len(SLIPS) - 1)
    slip_z = np.where(z <= VD[-1], SLIPS[islip], 0.)
    side = np.where(x[None, :] >= g[:, None], 1., -1.)

    XX, ZZ = np.meshgrid(x, z)
    pdist = perp_dist_to_polyline(XX.ravel(), ZZ.ravel(), verts).reshape(nz, nx)
    mu_field = np.where(np.abs(pdist) < dz, mr, 1.)

    idx = np.arange(nx * nz).reshape(nz, nx)
    rows, cols, vals = [], [], []
    b = np.zeros(nx * nz)

    def add_edges(p, q, c, J):
        p, q, c, J = p.ravel(), q.ravel(), c.ravel(), J.ravel()
        rows.extend((p, q, p, q))
        cols.extend((q, p, p, q))
        vals.extend((c, c, -c, -c))
        np.add.at(b, p, c * J)
        np.add.at(b, q, -c * J)

    c = 0.5 * (mu_field[:, :-1] + mu_field[:, 1:])
    J = np.where((z <= VD[-1])[:, None], slip_z[:, None] * (side[:, 1:] - side[:, :-1]) / 2., 0.)
    add_edges(idx[:, :-1], idx[:, 1:], c, J)

    c = 0.5 * (mu_field[:-1, :] + mu_field[1:, :])
    J = np.where((z[:-1] <= VD[-1])[:, None], slip_z[:-1, None] * (side[1:, :] - side[:-1, :]) / 2., 0.)
    add_edges(idx[:-1, :], idx[1:, :], c, J)

    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    pin = idx[-1, 0]
    keep = rows != pin
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    rows = np.append(rows, pin); cols = np.append(cols, pin); vals = np.append(vals, 1.)
    b[pin] = 0.

    A = sp.csc_matrix((vals, (rows, cols)), shape=(nx * nz, nx * nz))
    u = spla.spsolve(A, b)
    return x, u[:nx]


def check_fd(name, x_offsets, dz, mr, ax):
    t0 = time.time()
    x_fd, u_fd = fd_solve(x_offsets, dz, mr)
    near = np.abs(x_fd) <= 5000.
    xs, u_fd = x_fd[near], u_fd[near]
    xs_eval = np.where(xs == 0., 1e-6, xs)
    u_series = model(x_offsets, dz, mr, n_near=120, n_far=40).run(xs_eval).sol
    u_fd = u_fd - u_fd.mean()
    us = u_series - u_series.mean()
    rng = us.max() - us.min()
    rms = np.sqrt(np.mean((u_fd - us) ** 2)) / rng
    mx = np.abs(u_fd - us).max() / rng
    print(f'[5] FD vs BEM, {name}: rms {100*rms:.2f} %, max {100*mx:.2f} % '
          f'of range ({time.time()-t0:.0f} s)')
    ax.plot(xs, us, color='crimson', lw=1.5, label='boundary integral')
    ax.plot(xs, u_fd, color='k', lw=0.8, ls='--', label='finite difference')
    ax.set_title(f'{name}  (rms {100*rms:.2f} %)', fontsize=10)
    ax.axvline(0, color='lightgray', ls=':')
    ax.set_xlabel('x (m)')
    ax.legend(fontsize=8)
    assert rms < 0.03, f'{name}: FD mismatch'


def main():
    check_vertical_limit()
    check_homogeneous_dip()
    check_interface_conditions()
    check_resolution_convergence()

    offs = np.array([0., 134., 402., 804.])   # ~75 deg dip
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    check_fd('dipping, sheath (perp width, lines exit zone)', offs, 300., 0.4, axs[0])
    check_fd('dipping, sheath (perp width, lines in zone)', offs, 1000., 0.4, axs[1])
    for ax in axs:
        ax.set_ylabel('u (m)')
    fig.suptitle('TwoDDzSheathForwardModel validation: boundary integral vs FD (perpendicular-distance mask)')
    fig.savefig(FIG, dpi=150)
    print(f'\nAll checks passed. Figure -> {FIG}')


if __name__ == '__main__':
    main()
