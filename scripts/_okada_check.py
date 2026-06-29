"""
Independent Okada (1985) reference transcribed DIRECTLY from the provided document
(Segall 3.6.4 eqs 3.105-3.110 / Okada 1985 eqs 25-30), used to verify the dipping
convention in codes.Fault3d.compute_okada / compute_greens_functions.

Natural Okada frame (textbook):
  x along strike, y perpendicular, z up, medium z<=0.
  Reference corner = fault corner at xi'=0 (strike end) and eta'=0 (the DEEP edge,
  at depth d).  Surface projection of that corner is the origin (0,0).
  Fault footprint: xi' in [0,L] (+x), eta' in [0,W] up-dip -> depth d - eta' sin d,
  perpendicular y = eta' cos d (so the shallow edge is at y = W cos d, depth d-W sin d).
  p = y cos d + dpt sin d ;  q = y sin d - dpt cos d.
  Chinnery: f(X,p) - f(X,p-W) - f(X-L,p) + f(X-L,p-W).
Lame: lambda = mu  ->  mu/(lambda+mu) = 1/2  (nu = 0.25).
"""
import numpy as np

K = 0.5  # mu/(lambda+mu)


def okada_disp(X, Y, dpt, delta, L, W, U1, U2, eps=1e-10):
    sd, cd = np.sin(delta), np.cos(delta)
    p = Y * cd + dpt * sd
    q = Y * sd - dpt * cd

    def terms(xi, eta):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        yt = eta * cd + q * sd
        dt = eta * sd - q * cd
        Xr = np.sqrt(xi**2 + q**2)
        return R, yt, dt, Xr

    def atanq(xi, eta, R):
        sden = np.where(np.abs(q) < eps, 1.0, q * R)
        return np.where(np.abs(q) < eps, 0.0, np.arctan(xi * eta / sden))

    def I5(xi, eta):
        R, yt, dt, Xr = terms(xi, eta)
        if abs(cd) < eps:
            return -K * xi * sd / (R + dt)
        num = eta * (Xr + q * cd) + Xr * (R + Xr) * sd
        den = xi * (R + Xr) * cd
        sden = np.where(np.abs(xi) < eps, 1.0, den)
        val = K * (2.0 / cd) * np.arctan(num / sden)
        return np.where(np.abs(xi) < eps, 0.0, val)

    def I4(xi, eta):
        R, yt, dt, Xr = terms(xi, eta)
        if abs(cd) < eps:
            return -K * q / (R + dt)
        return K * (1.0 / cd) * (np.log(R + dt) - sd * np.log(R + eta))

    def I3(xi, eta):
        R, yt, dt, Xr = terms(xi, eta)
        if abs(cd) < eps:
            return (K / 2.0) * (eta / (R + dt) + yt * q / (R + dt)**2 - np.log(R + eta))
        return K * (yt / (cd * (R + dt)) - np.log(R + eta)) + (sd / cd) * I4(xi, eta)

    def I1(xi, eta):
        R, yt, dt, Xr = terms(xi, eta)
        if abs(cd) < eps:
            return -(K / 2.0) * xi * q / (R + dt)**2
        return K * (-xi / (cd * (R + dt))) - (sd / cd) * I5(xi, eta)

    def I2(xi, eta):
        R, yt, dt, Xr = terms(xi, eta)
        return K * (-np.log(R + eta)) - I3(xi, eta)

    def I5v(xi, eta):  # cos d = 0 branch for I5
        R, yt, dt, Xr = terms(xi, eta)
        return -K * xi * sd / (R + dt)

    def _I5(xi, eta):
        return I5v(xi, eta) if abs(cd) < eps else I5(xi, eta)

    # strike-slip integrands
    def ux_ss(xi, eta):
        R, yt, dt, Xr = terms(xi, eta)
        return xi * q / (R * (R + eta)) + atanq(xi, eta, R) + I1(xi, eta) * sd

    def uy_ss(xi, eta):
        R, yt, dt, Xr = terms(xi, eta)
        return yt * q / (R * (R + eta)) + q * cd / (R + eta) + I2(xi, eta) * sd

    def uz_ss(xi, eta):
        R, yt, dt, Xr = terms(xi, eta)
        return dt * q / (R * (R + eta)) + q * sd / (R + eta) + I4(xi, eta) * sd

    # dip-slip integrands
    def ux_ds(xi, eta):
        R, yt, dt, Xr = terms(xi, eta)
        return q / R - I3(xi, eta) * sd * cd

    def uy_ds(xi, eta):
        R, yt, dt, Xr = terms(xi, eta)
        return yt * q / (R * (R + xi)) + cd * atanq(xi, eta, R) - I1(xi, eta) * sd * cd

    def uz_ds(xi, eta):
        R, yt, dt, Xr = terms(xi, eta)
        return dt * q / (R * (R + xi)) + sd * atanq(xi, eta, R) - _I5(xi, eta) * sd * cd

    def chin(f):
        return f(X, p) - f(X, p - W) - f(X - L, p) + f(X - L, p - W)

    ux = -U1 / (2 * np.pi) * chin(ux_ss) - U2 / (2 * np.pi) * chin(ux_ds)
    uy = -U1 / (2 * np.pi) * chin(uy_ss) - U2 / (2 * np.pi) * chin(uy_ds)
    uz = -U1 / (2 * np.pi) * chin(uz_ss) - U2 / (2 * np.pi) * chin(uz_ds)
    return ux, uy, uz


def patch_disp_ref(patch, e, n, U1, U2):
    """Displacement (E, N, Up) for a Fault3d Patch via the independent reference."""
    x0, y0, z1 = patch.x0, patch.y0, patch.z1
    L = patch.get_along_strike_length()
    delta = patch.get_dip()
    W = patch.get_dd_width()
    sx = (patch.x1 - x0) / L
    sy = (patch.y1 - y0) / L
    ddx, ddy = -sy, sx          # +ys down-dip horizontal unit (strike rotated +90 CCW)
    d_deep = patch.z0
    # Okada natural frame: origin = deep-edge xi'=0 corner = A + W cos d along down-dip;
    # +Y = up-dip = -down-dip.
    refx = x0 + ddx * W * np.cos(delta)
    refy = y0 + ddy * W * np.cos(delta)
    de, dn = e - refx, n - refy
    X = de * sx + dn * sy
    Y = -(de * ddx + dn * ddy)              # up-dip positive
    uX, uY, uZ = okada_disp(X, Y, d_deep, delta, L, W, U1, U2)
    uE = uX * sx + uY * (-ddx)             # uY is along +Y = -down-dip
    uN = uX * sy + uY * (-ddy)
    return uE, uN, uZ


# -------------------- checks --------------------
if __name__ == "__main__":
    from codes import Fault3d, Patch, Cell, TwoDHomogeneousForwardModel

    # (A) independent ref: vertical SS vs 2D screw dislocation.
    L = 4e5
    Y = np.linspace(300., 1e4, 400)
    X = np.full_like(Y, L / 2)            # mid-strike
    uX, uY, uZ = okada_disp(X, Y, 1e4, np.pi/2, L, 1e4, 1.0, 0.0)
    two = TwoDHomogeneousForwardModel().run(Y)
    err = min(np.max(np.abs(uX - two.sol)), np.max(np.abs(uX + two.sol)))
    print(f"(A) independent ref vertical SS vs 2D: rel {err/np.max(np.abs(two.sol)):.3%}")

    # (B) current compute_greens_functions vs independent ref, several dips.
    def one_patch(dip_deg):
        p = Patch(-1e5, 0., 1e4, 1e5, 0., 0., dip=np.deg2rad(dip_deg), slip_sign=1)
        f = Fault3d()
        c = Cell("t", 1, 0, 2e5, 0., 1e4); c.patches = [p]
        f.cells = {(0, 0): c}; f.slips = np.zeros(1)
        return f, p

    e = np.linspace(-2e4, 2e4, 41)
    n = np.linspace(-2e4, 2e4, 41) + 137.0   # offset grid off the fault/trace lines
    E, N = np.meshgrid(e, n)
    pts = np.vstack((E.ravel(), N.ravel()))
    for dip_deg in (90., 70., 45.):
        f, p = one_patch(dip_deg)
        f.compute_greens_functions(pts)
        for slip_idx, (U1, U2, lbl) in enumerate([(1., 0., "SS"), (0., 1., "DS")]):
            code = f.gfs[slip_idx, 0]                       # (3, n) E,N,Up
            refE, refN, refUp = patch_disp_ref(p, pts[0], pts[1], U1, U2)
            ref = np.vstack((refE, refN, refUp))
            amp = np.max(np.abs(ref))
            d = np.max(np.abs(code - ref))
            print(f"(B) dip={dip_deg:4.0f} {lbl}: max|code-ref|={d:.3e}  amp={amp:.3e}  rel={d/amp:.3%}")
