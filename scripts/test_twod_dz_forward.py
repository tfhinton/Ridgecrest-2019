#!/usr/bin/env python3
'''Validation of the dipping-fault extension of TwoDDzForwardModel.

Checks:
  1. Vertical regression: dipping code with zero offsets == original
     compute_two_d_dz expressions.
  2. Homogeneous limit: dipping fault, modulus_ratio -> 1, vs the analytic
     antiplane half-space solution u = -(s/pi)[atan2(x-x_b, d_b) - atan2(x-x_t, d_t)].
  3. Interface conditions at the zone walls x = +-h: displacement continuity and
     traction continuity (mu2 u'(in) == mu1 u'(out)), with dislocation lines both
     inside and outside the zone.
  4. Source-position continuity as a line crosses the zone wall (x0 = h -+ eps).
  5. Independent finite-difference solution of div(mu grad u) = 0 with the slip
     jump imposed across the (dipping) fault cut, for vertical+DZ, dipping
     homogeneous, and dipping+DZ cases. Writes a comparison figure.

Run: ./.venv/bin/python scripts/test_twod_dz_forward.py
'''
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from codes import TwoDDzForwardModel
from codes.TwoDDzForwardModel import compute_two_d_dz, dislocation_line_dz
import config

FIG = config.TMP_DIR / 'test_twod_dz_forward.png'

VD = np.array([0., 500., 1500., 3000.])
SLIPS = np.array([3., 2., 1.])


def model(x_offsets, dz, mr):
    m = TwoDDzForwardModel(dz_half_width=dz, modulus_ratio=mr)
    m = m.build_dipping_patches(VD, x_offsets)
    m.slips = SLIPS.copy()
    return m


def analytic_homog(xs, x_offsets, slips=SLIPS):
    '''Homogeneous half-space antiplane solution for the dipping segments.'''
    u = np.zeros_like(xs)
    for i, s in enumerate(slips):
        xt, xb, dt, db = x_offsets[i], x_offsets[i+1], VD[i], VD[i+1]
        u += -(s/np.pi) * (np.arctan2(xs - xb, db) - np.arctan2(xs - xt, dt))
    return u


####    1-4: series checks    ####
def check_vertical_regression():
    xs = np.linspace(-6000., 6000., 4000)
    dz, mr = 400., 0.4
    new = model(np.zeros(4), dz, mr).run(xs).sol
    m_max = int(np.ceil(np.log(1e-6) / np.log((1-mr)/(1+mr))))
    old = np.zeros_like(xs)
    for i, s in enumerate(SLIPS):
        old += compute_two_d_dz(xs, VD[i+1], s, dz, mr, m_max)
        if VD[i] > 0.:
            old -= compute_two_d_dz(xs, VD[i], s, dz, mr, m_max)
    err = np.abs(new - old).max()
    print(f'[1] vertical regression: max |new - old| = {err:.2e}')
    assert err < 1e-6   # the two truncated series differ by one tail term


def check_homogeneous_dip():
    xs = np.linspace(-6000., 6000., 4000)
    offs = np.array([0., 134., 402., 804.])   # ~75 deg dip
    got = model(offs, 400., 1.0).run(xs).sol
    want = analytic_homog(xs, offs)
    err = np.abs(got - want).max()
    print(f'[2] homogeneous dipping vs analytic: max err = {err:.2e} m')
    assert err < 1e-9


def check_interface_conditions():
    dz, mr = 300., 0.4
    offs = np.array([0., 134., 402., 804.])   # lines at 402, 804 are OUTSIDE the zone
    mu2, mu1 = mr, 1.
    m = model(offs, dz, mr)
    f = lambda x: m.run(np.array([x])).sol[0]
    eps, de = 1e-3, 1e-4
    ok = True
    for wall in (dz, -dz):
        s = np.sign(wall)
        u_in, u_out = f(wall - s*eps), f(wall + s*eps)
        du_in = (f(wall - s*eps) - f(wall - s*(eps+de))) / (s*de)
        du_out = (f(wall + s*(eps+de)) - f(wall + s*eps)) / (s*de)
        cont = abs(u_in - u_out)
        trac = abs(mu2*du_in - mu1*du_out) / max(abs(mu2*du_in), 1e-12)
        print(f'[3] wall x={wall:+.0f}: |u_in - u_out| = {cont:.2e} m, '
              f'traction rel err = {trac:.2e}')
        ok &= cont < 1e-5 and trac < 1e-3   # limited by the 1e-6 series tolerance
    assert ok


def check_source_crossing():
    xs = np.linspace(-6000., 6000., 1200)
    dz, mr = 300., 0.4
    k = (1-mr)/(1+mr)
    m_max = int(np.ceil(np.log(1e-6)/np.log(k)))
    a = dislocation_line_dz(xs, dz - 0.01, 1000., dz, k, m_max)
    b = dislocation_line_dz(xs, dz + 0.01, 1000., dz, k, m_max)
    err = np.abs(a - b).max()
    print(f'[4] source crossing the wall (x0 = h -+ 0.01): max diff = {err:.2e}')
    assert err < 1e-3


####    5: finite-difference cross-check    ####
def fd_solve(x_offsets, dz, mr, L=25000., D=12000., h=25.):
    '''Solve div(mu grad u) = 0 on x in [-L, L], z in [0, D], with slip SLIPS
    imposed as a displacement jump across the piecewise-linear fault x = g(z).
    Free surface at z = 0 and zero-flux far boundaries (one pinned node), so the
    solution floats on a constant. Returns (x_nodes, u_surface).'''
    nx, nz = int(2*L/h) + 1, int(D/h) + 1
    x = -L + h*np.arange(nx)
    z = h*np.arange(nz)

    g = np.interp(z, VD, x_offsets)                     # fault x at each row
    islip = np.clip(np.searchsorted(VD, z, side='right') - 1, 0, len(SLIPS)-1)
    slip_z = np.where(z <= VD[-1], SLIPS[islip], 0.)    # slip at each row depth
    side = np.where(x[None, :] >= g[:, None], 1., -1.)  # (nz, nx)
    mu_of = lambda xx: np.where(np.abs(xx) < dz, mr, 1.)

    idx = np.arange(nx*nz).reshape(nz, nx)
    rows, cols, vals = [], [], []
    b = np.zeros(nx*nz)

    def add_edges(p, q, c, J):
        '''Edges p -> q with conductance c and slip jump J (= u_q - u_p across the
        cut). Adds the symmetric Laplacian entries and the jump terms to b.'''
        p, q, c, J = p.ravel(), q.ravel(), c.ravel(), J.ravel()
        rows.extend((p, q, p, q))
        cols.extend((q, p, p, q))
        vals.extend((c, c, -c, -c))
        np.add.at(b, p, c*J)
        np.add.at(b, q, -c*J)

    # horizontal edges (i, j) -> (i+1, j)
    c = np.broadcast_to(mu_of((x[:-1] + x[1:])/2.)[None, :], (nz, nx-1))
    J = np.where((z <= VD[-1])[:, None], slip_z[:, None]*(side[:, 1:] - side[:, :-1])/2., 0.)
    add_edges(idx[:, :-1], idx[:, 1:], c, J)

    # vertical edges (i, j) -> (i, j+1); a crossing means g passes x_i between rows
    c = np.broadcast_to(mu_of(x)[None, :], (nz-1, nx))
    J = np.where((z[:-1] <= VD[-1])[:, None], slip_z[:-1, None]*(side[1:, :] - side[:-1, :])/2., 0.)
    add_edges(idx[:-1, :], idx[1:, :], c, J)

    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    pin = idx[-1, 0]                                    # fix the floating constant
    keep = rows != pin
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    rows = np.append(rows, pin); cols = np.append(cols, pin); vals = np.append(vals, 1.)
    b[pin] = 0.

    A = sp.csc_matrix((vals, (rows, cols)), shape=(nx*nz, nx*nz))
    u = spla.spsolve(A, b)
    return x, u[:nx]


def check_fd(name, x_offsets, dz, mr, ax):
    t0 = time.time()
    x_fd, u_fd = fd_solve(x_offsets, dz, mr)
    near = np.abs(x_fd) <= 5000.
    xs = x_fd[near]
    u_fd = u_fd[near]
    # series at the node positions; x=0 evaluated as the +side limit to match the
    # FD side convention at the trace node
    u_series = model(x_offsets, dz, mr).run(np.where(xs == 0., 1e-6, xs)).sol
    u_fd = u_fd - u_fd.mean()
    us = u_series - u_series.mean()
    rng = us.max() - us.min()
    rms = np.sqrt(np.mean((u_fd - us)**2)) / rng
    mx = np.abs(u_fd - us).max() / rng
    print(f'[5] FD vs series, {name}: rms {100*rms:.2f} %, max {100*mx:.2f} % '
          f'of range ({time.time()-t0:.0f} s)')
    ax.plot(xs, us, color='crimson', lw=1.5, label='image series')
    ax.plot(xs, u_fd, color='k', lw=0.8, ls='--', label='finite difference')
    ax.set_title(f'{name}  (rms {100*rms:.2f} %)', fontsize=10)
    ax.axvline(0, color='lightgray', ls=':')
    ax.set_xlabel('x (m)')
    ax.legend(fontsize=8)
    assert rms < 0.02, f'{name}: FD mismatch'


def main():
    check_vertical_regression()
    check_homogeneous_dip()
    check_interface_conditions()
    check_source_crossing()

    offs = np.array([0., 134., 402., 804.])   # ~75 deg dip
    fig, axs = plt.subplots(1, 4, figsize=(18, 4), layout='constrained')
    check_fd('vertical + DZ', np.zeros(4), 400., 0.4, axs[0])
    check_fd('dipping, homogeneous', offs, 400., 1.0, axs[1])
    check_fd('dipping + DZ (lines exit zone)', offs, 300., 0.4, axs[2])
    check_fd('dipping + DZ (lines in zone)', offs, 1000., 0.4, axs[3])
    for ax in axs:
        ax.set_ylabel('u (m)')
    fig.suptitle('TwoDDzForwardModel dipping validation: image series vs FD')
    fig.savefig(FIG, dpi=150)
    print(f'\nAll checks passed. Figure -> {FIG}')


if __name__ == '__main__':
    main()
