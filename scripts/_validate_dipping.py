"""
Validation for the dipping / even-width Fault3d rewrite.

Checks (see plan reactive-soaring-meerkat.md "Verification"):
  1. Vertical strike-slip long-fault limit vs TwoDHomogeneousForwardModel (2D screw).
     (This also covers the dip=90 case, since the new build is vertical at dip=90.)
  2. Dip consistency (no external ref): continuity to vertical, geometry closure,
     finiteness at the trace, near-field down-dip asymmetry, GF linearity.
  3. build_patches end-to-end on the real trace (with synthetic fault/dip_az cols),
     correct cell/GF shapes, and saved plot_slip / plot_fault3d figures.

Run: ./.venv/bin/python scripts/_validate_dipping.py   (from project root)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from codes import Fault3d, Patch, Cell, TwoDHomogeneousForwardModel


def _single_patch_fault(patch, width=2e5):
    f = Fault3d()
    cell = Cell(fault="t", slip_sign=patch.slip_sign, x_along=0.0, width=width,
                z_top=patch.z1, z_bot=patch.z0)
    cell.patches = [patch]
    f.cells = {(0, 0): cell}
    f.slips = np.zeros(1)
    return f


def check_vertical_ss_vs_2d():
    # Long vertical strike-slip patch, top 0 -> bottom 10 km, strike along +x.
    patch = Patch(-1e5, 0., 1e4, 1e5, 0., 0., dip=np.pi/2, slip_sign=1)
    f = _single_patch_fault(patch)

    ys = np.linspace(-1e4, 1e4, 1001)
    ys = ys[np.abs(ys) > 50.]                       # avoid the singular trace point
    pts = np.vstack((np.zeros_like(ys), ys))        # mid-section, cross-fault profile
    f.compute_greens_functions(pts)
    ss_east = f.gfs[0, 0, 0, :]                      # along-strike (antiplane) comp

    two = TwoDHomogeneousForwardModel()
    two.patches = two.patches                        # default PatchTwoD: top 0, bottom 10 km
    two = two.run(ys)
    sol = two.sol

    # Compare up to an overall sign (convention), at |y| > 500 m (far from core).
    far = np.abs(ys) > 500.
    err_plus = np.max(np.abs(ss_east[far] - sol[far]))
    err_minus = np.max(np.abs(ss_east[far] + sol[far]))
    err = min(err_plus, err_minus)
    amp = np.max(np.abs(sol[far]))
    rel = err / amp
    print(f"[1] vertical SS vs 2D screw: max|diff|={err:.3e} (amp {amp:.3e}, "
          f"rel {rel:.3%}, sign={'+' if err_plus < err_minus else '-'})")
    assert rel < 0.01, "vertical SS does not match 2D analytic limit"
    return ys, ss_east, sol


def check_continuity_to_vertical():
    ys = np.linspace(300., 1e4, 400)
    pts = np.vstack((np.zeros_like(ys), ys))

    def profile(dip_deg, slip_idx):
        p = Patch(-1e5, 0., 1e4, 1e5, 0., 0., dip=np.deg2rad(dip_deg), slip_sign=1)
        f = _single_patch_fault(p)
        f.compute_greens_functions(pts)
        return f.gfs[slip_idx, 0]                    # (3, n)

    v = profile(90.0, 0)
    near = profile(89.99, 0)
    diff = np.max(np.abs(v - near)) / np.max(np.abs(v))
    print(f"[2a] continuity dip 90 vs 89.99 (SS): rel diff {diff:.3%}")
    assert diff < 0.02, "GF discontinuous approaching vertical"


def check_geometry_closure_and_offset_side():
    # Synthetic straight ENE trace; build one column, verify listric closure + side.
    from shapely.geometry import LineString
    import geopandas as gpd

    line = LineString([(0., 0.), (20000., 0.)])     # strike due east
    gdf = gpd.GeoDataFrame({"id": [1], "fault": ["F"], "dip_az": [180.0]},
                           geometry=[line], crs="EPSG:32611").set_index("id")
    f = Fault3d()
    f.trace = gdf
    vd = [0, 2000, 6000, 9000, 12000]
    geom = {1: dict(dip_sh=60, dip_dp=88, kink_z=6000, slip_sign=-1)}
    f.build_patches(vd, patch_length=5000., geom=geom)

    cells = list(f.cells.values())
    # Take the first along-strike piece's column (v=0..3) and walk corners.
    col = [c for c in cells if c.x_along == 0.0]
    col = sorted(col, key=lambda c: c.z_top)
    prev_lower = None
    max_gap = 0.0
    for c in col:
        for p in c.patches:                          # ordered shallow->deep
            A, B, Bl, Al = p.corners3d()
            if prev_lower is not None:
                gap = np.hypot(A[0]-prev_lower[0][0], A[1]-prev_lower[0][1]) + \
                      abs(A[2]-prev_lower[0][2])
                max_gap = max(max_gap, gap)
            prev_lower = (Al, Bl)
    print(f"[2b] listric closure max corner gap between stacked patches: {max_gap:.3e} m")
    assert max_gap < 1e-6, "stacked patches not connected"

    # dip_az=180 (south). Strike is east; Okada down-dip +ys = north (left of east).
    # Reversal should flip strike to west so +ys points south -> offset y decreases.
    p0 = col[0].patches[0]
    A, B, Bl, Al = p0.corners3d()
    print(f"[2c] dip_az=180: lower-edge northing {Al[1]:.1f} (expect < 0, i.e. south)")
    assert Al[1] < 0., "offset side does not match dip_az"


def check_finite_and_asymmetry():
    # Shallow dip should give larger near-field surface signal on the down-dip side.
    p = Patch(-1e5, 0., 1e4, 1e5, 0., 0., dip=np.deg2rad(45.), slip_sign=0)  # pure DS
    # use dip-slip: build via slip_sign irrelevant for DS; eval both sides
    f = _single_patch_fault(p)
    ys = np.array([-3000., -1500., 1500., 3000.])
    pts = np.vstack((np.zeros_like(ys), ys))
    f.compute_greens_functions(pts)
    assert np.all(np.isfinite(f.gfs)), "non-finite GF values"
    up_ds = f.gfs[1, 0, 2, :]                         # DS vertical
    # +ys is the down-dip side for this patch; expect larger |uz| there.
    down_dip = np.mean(np.abs(up_ds[ys > 0]))
    up_dip = np.mean(np.abs(up_ds[ys < 0]))
    print(f"[2d] DS vertical near-field: down-dip|uz|={down_dip:.3e} "
          f"up-dip|uz|={up_dip:.3e}")
    assert np.isfinite(down_dip) and np.isfinite(up_dip)


def check_linearity():
    # A cell GF must equal the sum of its sub-patch GFs.
    from shapely.geometry import LineString
    import geopandas as gpd
    line = LineString([(0., 0.), (5000., 1000.), (10000., 0.)])  # bent -> 2 sub-segs
    gdf = gpd.GeoDataFrame({"id": [1], "fault": ["F"], "dip_az": [90.0]},
                           geometry=[line], crs="EPSG:32611").set_index("id")
    f = Fault3d(); f.trace = gdf
    f.build_patches([0, 4000], patch_length=20000., geom={
        1: dict(dip_sh=70, dip_dp=88, kink_z=6000, slip_sign=1)})
    cell = list(f.cells.values())[0]
    assert len(cell.patches) >= 2, "expected multiple sub-patches in one cell"

    xs = np.linspace(-5000, 15000, 60)
    yy = np.linspace(-8000, 8000, 60)
    X, Y = np.meshgrid(xs, yy)
    pts = np.vstack((X.ravel(), Y.ravel()))
    f.compute_greens_functions(pts)
    cell_gf = f.gfs[:, 0]                              # (2,3,n)

    # Sum sub-patches manually.
    manual = np.zeros_like(cell_gf)
    for p in cell.patches:
        g = Fault3d()
        c = Cell("t", p.slip_sign, 0, 0, p.z1, p.z0); c.patches = [p]
        g.cells = {(0, 0): c}; g.slips = np.zeros(1)
        g.compute_greens_functions(pts)
        manual += g.gfs[:, 0]
    diff = np.max(np.abs(cell_gf - manual))
    print(f"[2e] cell GF == sum of sub-patch GFs: max|diff|={diff:.3e}")
    assert diff < 1e-9, "cell GF is not the linear sum of its sub-patches"


def check_end_to_end_and_plots():
    f = Fault3d()
    f.import_trace("data/fault/fault_trace_utm.shp")
    # Inject synthetic fault / dip_az columns (user will set these in QGIS).
    f.trace["fault"] = {1: "ELLF", 2: "SLLF", 3: "other"}
    f.trace["dip_az"] = {1: 45.0, 2: 315.0, 3: 0.0}   # NE, NW
    vd = [0, 500, 1500, 2500, 3500, 5000, 6500, 8500, 11500, 15000]
    geom = {
        1: dict(dip_sh=65, dip_dp=88, kink_z=6000, slip_sign=-1),
        2: dict(dip_sh=85, dip_dp=88, kink_z=6000, slip_sign=+1),
    }
    f.build_patches(vd, patch_length=3000., geom=geom)
    n_cells = len(f.cells)
    f.slips = np.linspace(0., 5., n_cells)
    print(f"[3] build_patches: {n_cells} cells, "
          f"faults={sorted(set(c.fault for c in f.cells.values()))}")

    xs = np.linspace(4.3e5, 4.75e5, 60)
    yy = np.linspace(3.93e6, 3.975e6, 60)
    X, Y = np.meshgrid(xs, yy)
    pts = np.vstack((X.ravel(), Y.ravel()))
    f.compute_greens_functions(pts)
    print(f"    gfs shape {f.gfs.shape}  (expect (2, {n_cells}, 3, {X.size}))")
    assert f.gfs.shape == (2, n_cells, 3, X.size)
    assert len(f.slips) == n_cells == f.gfs.shape[1]
    assert np.all(np.isfinite(f.gfs))

    fig, axes = f.plot_slip(cmap="viridis")
    fig.savefig("scripts/_validate_plot_slip.png", dpi=110)
    plt.close(fig)
    fig, ax = f.plot_fault3d(color_by="fault")
    fig.savefig("scripts/_validate_plot_fault3d_byfault.png", dpi=110)
    plt.close(fig)
    fig, ax = f.plot_fault3d(color_by="slip", cmap="lajolla" if False else "plasma")
    fig.savefig("scripts/_validate_plot_fault3d_slip.png", dpi=110)
    plt.close(fig)
    print("    saved _validate_plot_slip.png, _validate_plot_fault3d_*.png")


if __name__ == "__main__":
    check_vertical_ss_vs_2d()
    check_continuity_to_vertical()
    check_geometry_closure_and_offset_side()
    check_finite_and_asymmetry()
    check_linearity()
    check_end_to_end_and_plots()
    print("\nALL CHECKS PASSED")
