"""
Independent Okada (1985) surface-displacement reference, transcribed from the
documented okada85 convention (Beauducel). Used ONLY to validate the dipping
geometry/convention in codes.Fault3d. Not part of the production package.

Convention (okada85):
  e, n     : observation coords (East, North), relative to the FAULT CENTROID
             surface projection taken as (0, 0).
  depth    : depth of the fault CENTROID (> 0, positive down).
  strike   : trace azimuth (deg from North), defined so the fault dips to the
             RIGHT of the strike vector.
  dip      : 0..90 deg.
  L, W     : along-strike length, down-dip width (> 0).
  rake     : slip direction of the hanging wall, measured from strike.
             0 = left-lateral, 180 = right-lateral, 90 = reverse (dip-slip up).
  slip     : magnitude of slip in the rake direction.
Returns surface uE, uN, uUp.
"""
import numpy as np


def okada85(e, n, depth, strike, dip, L, W, rake, slip, nu=0.25):
    strike = np.deg2rad(strike)
    dip = np.deg2rad(dip)
    rake = np.deg2rad(rake)
    U1 = np.cos(rake) * slip   # strike-slip component
    U2 = np.sin(rake) * slip   # dip-slip component

    e = np.asarray(e, dtype=float)
    n = np.asarray(n, dtype=float)

    # centroid -> Okada corner reference (bottom-edge corner at depth d)
    d = depth + np.sin(dip) * W / 2.0
    ec = e + np.cos(strike) * np.cos(dip) * W / 2.0
    nc = n - np.sin(strike) * np.cos(dip) * W / 2.0
    x = np.cos(strike) * nc + np.sin(strike) * ec + L / 2.0
    y = np.sin(strike) * nc - np.cos(strike) * ec + np.cos(dip) * W

    p = y * np.cos(dip) + d * np.sin(dip)
    q = y * np.sin(dip) - d * np.cos(dip)

    def chin(f):
        return (f(x, p, q) - f(x, p - W, q)
                - f(x - L, p, q) + f(x - L, p - W, q))

    sd, cd = np.sin(dip), np.cos(dip)
    eps = 1e-12

    def terms(xi, eta, q):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * sd - q * cd          # d-tilde
        yb = eta * cd + q * sd          # y-tilde
        X = np.sqrt(xi**2 + q**2)
        return R, db, yb, X

    def I5(xi, eta, q):
        R, db, yb, X = terms(xi, eta, q)
        if abs(cd) < eps:
            return -(1 - 2 * nu) * xi * sd / (R + db)
        denom = xi * (R + X) * cd
        num = eta * (X + q * cd) + X * (R + X) * sd
        val = (1 - 2 * nu) * (2.0 / cd) * np.arctan(
            num / np.where(np.abs(denom) < eps, np.inf, denom))
        return np.where(np.abs(xi) < eps, 0.0, val)

    def I4(xi, eta, q):
        R, db, yb, X = terms(xi, eta, q)
        if abs(cd) < eps:
            return -(1 - 2 * nu) * q / (R + db)
        return (1 - 2 * nu) / cd * (np.log(R + db) - sd * np.log(R + eta))

    def I3(xi, eta, q):
        R, db, yb, X = terms(xi, eta, q)
        if abs(cd) < eps:
            return 0.5 * (1 - 2 * nu) * (eta / (R + db) + yb * q / (R + db)**2 - np.log(R + eta))
        return (1 - 2 * nu) * (yb / (cd * (R + db)) - np.log(R + eta)) + sd / cd * I4(xi, eta, q)

    def I1(xi, eta, q):
        R, db, yb, X = terms(xi, eta, q)
        if abs(cd) < eps:
            return -0.5 * (1 - 2 * nu) * xi * q / (R + db)**2
        return (1 - 2 * nu) * (-xi / (cd * (R + db))) - sd / cd * I5(xi, eta, q)

    def I2(xi, eta, q):
        R, db, yb, X = terms(xi, eta, q)
        return (1 - 2 * nu) * (-np.log(R + eta)) - I3(xi, eta, q)

    def atan_q(xi, eta, q, R):
        return np.where(np.abs(q) < eps, 0.0,
                        np.arctan(xi * eta / np.where(np.abs(q) < eps, np.inf, q * R)))

    # strike-slip
    def ux_ss(xi, eta, q):
        R, db, yb, X = terms(xi, eta, q)
        return xi * q / (R * (R + eta)) + atan_q(xi, eta, q, R) + I1(xi, eta, q) * sd

    def uy_ss(xi, eta, q):
        R, db, yb, X = terms(xi, eta, q)
        return yb * q / (R * (R + eta)) + q * cd / (R + eta) + I2(xi, eta, q) * sd

    def uz_ss(xi, eta, q):
        R, db, yb, X = terms(xi, eta, q)
        return db * q / (R * (R + eta)) + q * sd / (R + eta) + I4(xi, eta, q) * sd

    # dip-slip
    def ux_ds(xi, eta, q):
        R, db, yb, X = terms(xi, eta, q)
        return q / R - I3(xi, eta, q) * sd * cd

    def uy_ds(xi, eta, q):
        R, db, yb, X = terms(xi, eta, q)
        return yb * q / (R * (R + xi)) + cd * atan_q(xi, eta, q, R) - I1(xi, eta, q) * sd * cd

    def uz_ds(xi, eta, q):
        R, db, yb, X = terms(xi, eta, q)
        return db * q / (R * (R + xi)) + sd * atan_q(xi, eta, q, R) - I5(xi, eta, q) * sd * cd

    ux = -U1 / (2 * np.pi) * chin(ux_ss) - U2 / (2 * np.pi) * chin(ux_ds)
    uy = -U1 / (2 * np.pi) * chin(uy_ss) - U2 / (2 * np.pi) * chin(uy_ds)
    uz = -U1 / (2 * np.pi) * chin(uz_ss) - U2 / (2 * np.pi) * chin(uz_ds)

    # fault frame -> geographic (East, North)
    uE = np.sin(strike) * ux - np.cos(strike) * uy
    uN = np.cos(strike) * ux + np.sin(strike) * uy
    return uE, uN, uz
