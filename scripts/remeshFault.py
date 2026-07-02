#!/usr/bin/env python3
"""Remesh a GOCAD TSurf community fault model (CFM) into a depth-layered
triangular mesh suitable for Okada/Meade Green's-function calculation.

Design goals (Ridgecrest doublet inversion)
--------------------------------------------
1. **Separable at depth contours.** The mesh is built layer by layer between a
   user-supplied set of horizontal depth contours. Adjacent layers *share* the
   exact resampled contour vertices, so edges are continuous and the mesh can be
   split cleanly at any contour (each patch carries its ``layer`` index).
2. **Coarser with depth.** The user gives a point count for the top contour and
   the bottom contour; intermediate contours are resampled to a count
   interpolated linearly in *layer index* (so the triangle count drops roughly
   linearly per layer even when depth spacing is uneven). Counts are forced even.
3. **Flat free surface.** Okada/Meade assume a half-space with a flat surface at
   z = 0. The CFM top trace follows topography (here 542-757 m). We take the
   datum as the *lowest* top-trace elevation, cut the uppermost ("zero") contour
   there, discard the topographic wedge above it, and shift all z so the datum
   sits at 0. User depth intervals are measured downward from this datum.

Algorithm
---------
For each depth ``d`` (positive-down, relative to the datum):
  * slice the original mesh with the horizontal plane z = datum - d
    (reusing ``extract_surface_trace.triangle_plane_segments`` + ``stitch_segments``);
  * keep the longest stitched polyline (degenerate single-touch crossings are
    dropped) and orient it along a common strike axis (PCA of the top contour);
  * resample it to N(d) points evenly by arc length.
Consecutive contours are stitched into a triangle strip by a two-pointer merge
over *normalized along-strike length*: a deeper vertex connects to the two
shallower vertices it falls between.

**Deep extension (Ridgecrest-specific).** The CFM bottom trace is uneven and
only fully defined along strike to ~9 km. Any requested contour deeper than
``PROJECT_BELOW_DEPTH`` (9 km) is therefore *not* sliced from the CFM; its x-y
shape is the deepest fully-defined (9 km) contour, projected straight down to
that depth. This extends the fault vertically below 9 km while keeping the mesh
watertight and avoiding splices of partial CFM traces.

Outputs
-------
  * a list of :class:`TrianglePatch` (and a :class:`RemeshedFault` bundle);
  * a GOCAD TSurf ``.txt`` matching the input format (shifted to z=0 datum);
  * an ``.npz`` for fast reload / plotting (see ``remeshFault_plot.py``).

Usage
-----
    python remeshFault.py INPUT.txt [-o OUT.txt] \
        [--depths 0,1000,2000,3000,4500,6500,9000,12000,15000] \
        [--n-top 50] [--n-bottom 12] [--keep-elevation] [--plot]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Reuse the TSurf parser and slicing primitives that live next door.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_surface_trace as est  # noqa: E402


# Ridgecrest-specific: the CFM geometry is only fully defined along strike to
# ~9 km (its bottom trace is uneven, reaching ~13 km in places but not all the
# way along). Any requested contour *deeper* than this is not sliced from the
# CFM; instead the deepest fully-defined contour (at this depth) is projected
# vertically straight down. This keeps the deep mesh watertight and avoids
# splicing partial CFM traces. See remesh_handover.md / the module docstring.
PROJECT_BELOW_DEPTH = 10000.0


# --------------------------------------------------------------------------- #
#  Patch container
# --------------------------------------------------------------------------- #
class TrianglePatch:
    """A single triangular dislocation element.

    Vertices are stored as a (3, 3) array of ``(x, y, z)`` in the *shifted*
    frame: z = elevation - datum, so the top (zero) contour is at z = 0 and
    everything below is negative (ZPOSITIVE Elevation convention, datum-shifted).

    ``layer`` is the index of the depth band the patch belongs to (band k lies
    between depth contour k and k+1), used to keep the mesh separable.
    """

    __slots__ = ("vertices", "layer")

    def __init__(self, v0, v1, v2, layer=None):
        self.vertices = np.asarray([v0, v1, v2], dtype=float)
        self.layer = layer

    @property
    def centroid(self):
        return self.vertices.mean(axis=0)

    @property
    def area(self):
        a = self.vertices[1] - self.vertices[0]
        b = self.vertices[2] - self.vertices[0]
        return 0.5 * float(np.linalg.norm(np.cross(a, b)))

    @property
    def unit_normal(self):
        """Unit normal (v1-v0) x (v2-v0). Winding is made consistent across the
        mesh by :func:`orient_triangles`, so these all point to the same fault
        face (the +``winding_ref`` side)."""
        a = self.vertices[1] - self.vertices[0]
        b = self.vertices[2] - self.vertices[0]
        n = np.cross(a, b)
        norm = np.linalg.norm(n)
        return n / norm if norm else n

    @property
    def depth(self):
        """Mean depth, positive downward (metres below the datum)."""
        return float(-self.vertices[:, 2].mean())

    def depth_vertices(self):
        """Vertices with z positive-downward, for Okada/Meade input."""
        v = self.vertices.copy()
        v[:, 2] *= -1.0
        return v

    def __repr__(self):
        c = self.centroid
        return (f"TrianglePatch(layer={self.layer}, "
                f"centroid=({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}), "
                f"area={self.area / 1e6:.3f} km^2)")


@dataclass
class RemeshedFault:
    """Bundle of everything the remesh produces, plus provenance."""
    patches: list = field(default_factory=list)
    vertices: np.ndarray = None          # (V, 3) shifted frame (z = elev - datum)
    triangles: np.ndarray = None         # (T, 3) int, 0-based into vertices
    layers: np.ndarray = None            # (T,) int band index per triangle
    contours: list = field(default_factory=list)  # list of (Ni, 3) arrays
    depths: np.ndarray = None            # (K,) positive-down depths of contours
    datum: float = 0.0                   # original elevation taken as z = 0
    name: str = "remeshed_fault"
    winding_ref: np.ndarray = None       # (3,) all unit_normals point to its +side

    def __repr__(self):
        return (f"RemeshedFault(name={self.name!r}, "
                f"{len(self.patches)} patches, {len(self.contours)} contours, "
                f"datum={self.datum:.1f} m)")


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def _even(n):
    """Round to the nearest positive even integer (>= 2)."""
    n = int(round(n))
    if n % 2:
        n += 1
    return max(2, n)


def _polyline_length(line):
    p = np.asarray(line, dtype=float)
    return float(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum())


def top_datum(verts, tris, **kw):
    """Datum elevation = lowest point of the topographic top edge."""
    segs = est.top_edges(verts, tris, **kw)
    if not segs:
        raise RuntimeError("Could not extract a top edge to define the datum.")
    return min(p[2] for s in segs for p in s)


def strike_axis(line):
    """Unit XY strike direction from the principal axis (PCA) of a contour."""
    p = np.asarray(line, dtype=float)[:, :2]
    p = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    return vt[0]


def cut_contour(verts, tris, z_level, axis, tol=1.0):
    """Return the longest stitched polyline at z = z_level, as an (N, 3) array.

    The slice is horizontal, so every point's z is exactly ``z_level``. The
    polyline is oriented so it runs in the +``axis`` direction (consistent
    across contours).
    """
    segs = est.triangle_plane_segments(verts, tris, z_level)
    if not segs:
        return None
    lines = est.stitch_segments(segs, tol=tol)
    if not lines:
        return None
    line = max(lines, key=_polyline_length)          # drop degenerate touches
    pts = np.column_stack([np.asarray(line, dtype=float),
                           np.full(len(line), z_level)])
    # orient along +axis
    proj = pts[:, :2] @ axis
    if proj[0] > proj[-1]:
        pts = pts[::-1]
    return pts


def resample_polyline(pts, n):
    """Resample an (M, 3) polyline to ``n`` points evenly by arc length."""
    pts = np.asarray(pts, dtype=float)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    out = np.empty((n, 3))
    for k in range(3):
        out[:, k] = np.interp(target, s, pts[:, k])
    return out


def stitch_two(i_top0, m, i_bot0, n):
    """Two-pointer triangle strip between a top row (m verts, global base
    ``i_top0``) and a bottom row (n verts, base ``i_bot0``), matched by
    normalized position. Returns a list of (a, b, c) global-index triples.
    """
    tp = np.linspace(0.0, 1.0, m)
    bp = np.linspace(0.0, 1.0, n)
    tris = []
    i = j = 0
    while i < m - 1 or j < n - 1:
        if j >= n - 1:                       # bottom exhausted: fan from top
            tris.append((i_top0 + i, i_top0 + i + 1, i_bot0 + j))
            i += 1
        elif i >= m - 1:                     # top exhausted: fan from bottom
            tris.append((i_top0 + i, i_bot0 + j, i_bot0 + j + 1))
            j += 1
        elif tp[i + 1] <= bp[j + 1]:         # advance whichever's next is nearer
            tris.append((i_top0 + i, i_top0 + i + 1, i_bot0 + j))
            i += 1
        else:
            tris.append((i_top0 + i, i_bot0 + j, i_bot0 + j + 1))
            j += 1
    return tris


def perp_horizontal(axis):
    """Horizontal unit vector perpendicular to the strike ``axis`` (xy)."""
    sx, sy = axis[0], axis[1]
    return np.array([-sy, sx, 0.0])


def orient_triangles(vertices, triangles, ref):
    """Return ``triangles`` with consistent winding: every triangle's normal
    ``(v1-v0) x (v2-v0)`` is flipped (by swapping its 2nd/3rd vertices) to point
    to the +``ref`` side. The strip stitch emits mixed orders, so without this
    the per-element normals flip from triangle to triangle -- which scrambles
    the strike-slip/dip-slip sign convention of a TDE Green's function.
    """
    tris = np.asarray(triangles, dtype=int).copy()
    v = np.asarray(vertices, dtype=float)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(v[b] - v[a], v[c] - v[a])
    flip = (n @ np.asarray(ref, dtype=float)) < 0
    tris[flip, 1], tris[flip, 2] = tris[flip, 2].copy(), tris[flip, 1].copy()
    return tris


def enforce_consistent_winding(rf, ref=None):
    """Re-orient an existing :class:`RemeshedFault` in place so all element
    normals point to one fault face; rebuilds ``rf.patches``. Handy after
    :func:`load_npz`. If ``ref`` is None it is derived from the top contour's
    strike (or a PCA of all vertices)."""
    if ref is None:
        base = rf.contours[0] if rf.contours else rf.vertices
        ref = perp_horizontal(strike_axis(base))
    rf.triangles = orient_triangles(rf.vertices, rf.triangles, ref)
    rf.winding_ref = np.asarray(ref, dtype=float)
    rf.patches = [TrianglePatch(*rf.vertices[t], layer=int(lyr))
                  for t, lyr in zip(rf.triangles, rf.layers)]
    return rf


# --------------------------------------------------------------------------- #
#  Main remeshing routine
# --------------------------------------------------------------------------- #
def remesh_fault(path, depths, n_top, n_bottom, tol=1.0, name=None,
                 taper_warn_frac=0.85, clip_to_fault=True):
    """Remesh a CFM TSurf file into a depth-layered triangular mesh.

    Parameters
    ----------
    path : path to the GOCAD TSurf ``.txt``.
    depths : sequence of positive-down depths (m) for the contours, relative to
        the datum; should start at 0 (the datum / zero contour) and increase.
    n_top, n_bottom : even point counts for the shallowest and deepest contours;
        intermediate contours interpolate linearly in layer index.
    tol : endpoint snapping tolerance (m) for stitching slice segments.
    taper_warn_frac : warn if a contour's strike span falls below this fraction
        of the top contour's span (normalized stitching stretches such contours).
    clip_to_fault : drop requested depths that lie below the fault's deepest
        extent (keeps the same call working across strands of differing depth,
        e.g. for batch runs).
    """
    path = Path(path)
    verts, tris = est.parse_tsurf(path)
    if not tris:
        raise RuntimeError(f"No TRGL triangles in {path}")

    depths = np.asarray(sorted(set(float(d) for d in depths)), dtype=float)
    if depths[0] != 0.0:
        print(f"[remesh] note: prepending datum contour (depth 0) to {depths}")
        depths = np.concatenate([[0.0], depths])

    datum = top_datum(verts, tris)

    # Contours at or above PROJECT_BELOW_DEPTH are sliced from the CFM geometry;
    # deeper ones are projected vertically from the deepest sliced contour.
    geom_depths = depths[depths <= PROJECT_BELOW_DEPTH + max(1.0, tol)]
    proj_depths = depths[depths > PROJECT_BELOW_DEPTH + max(1.0, tol)]

    if clip_to_fault:
        # Only the geometry contours can be clipped by the fault's real extent;
        # projected contours are drawn below it by design.
        fault_max_depth = datum - min(v[2] for v in verts.values())
        keep = geom_depths < fault_max_depth - max(1.0, tol)
        if not keep.all():
            dropped = geom_depths[~keep]
            print(f"[remesh] clipping geometry depths below fault extent "
                  f"({fault_max_depth:.0f} m): dropped {dropped.astype(int)}")
            geom_depths = geom_depths[keep]
        if len(geom_depths) < 2:
            raise RuntimeError(f"Fewer than 2 geometry contours remain for "
                               f"{path.name} (fault only {fault_max_depth:.0f} "
                               f"m deep).")

    depths = np.concatenate([geom_depths, proj_depths])
    n_geom = len(geom_depths)

    print(f"[remesh] datum (z=0) at elevation {datum:.1f} m; "
          f"{len(depths)} contours from 0 to {depths[-1]:.0f} m depth "
          f"({n_geom} from CFM geometry, {len(proj_depths)} projected "
          f"vertically below {PROJECT_BELOW_DEPTH:.0f} m)")

    # axis from the zero contour (slice just below datum to avoid tangency)
    z0_seg = est.triangle_plane_segments(verts, tris, datum - max(1.0, tol))
    axis = strike_axis(max(est.stitch_segments(z0_seg, tol=tol),
                           key=_polyline_length))

    # point count per contour, linear in layer *index* (so triangle count drops
    # ~linearly per layer regardless of uneven depth spacing), forced even
    K = len(depths)
    fr = (np.arange(K) / (K - 1)) if K > 1 else np.zeros(K)
    counts = [_even(n_top + f * (n_bottom - n_top)) for f in fr]

    contours = []
    top_span = None
    deepest_geom_raw = None                # raw polyline of the deepest CFM slice
    for i, (d, n) in enumerate(zip(depths, counts)):
        if i >= n_geom:                    # projected: reuse the deepest CFM
            raw = deepest_geom_raw.copy()  # slice's x-y, dropped straight down
            raw[:, 2] = datum - d
        else:
            z_level = datum - d
            if d == 0.0:                   # nudge the datum slice off tangency
                z_level -= max(1.0, tol)
            raw = cut_contour(verts, tris, z_level, axis, tol=tol)
            if raw is None:
                raise RuntimeError(f"Mesh does not cross depth {d:.0f} m "
                                   f"(z={z_level:.1f}); trim your depth list.")
            deepest_geom_raw = raw         # last CFM slice = PROJECT_BELOW_DEPTH
        span = float(np.ptp(raw[:, :2] @ axis))
        if top_span is None:
            top_span = span
        elif span < taper_warn_frac * top_span and i < n_geom:
            print(f"[remesh] WARNING: depth {d:.0f} m contour spans "
                  f"{span / 1000:.1f} km ({span / top_span:.0%} of top); "
                  f"normalized stitching will stretch it laterally. Consider a "
                  f"shallower max depth.")
        contours.append(resample_polyline(raw, n))

    # shift to datum frame (z = elevation - datum) and assemble global vertices
    contours = [c - np.array([0.0, 0.0, datum]) for c in contours]
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in contours])])
    vertices = np.vstack(contours)

    triangles, layers = [], []
    for k in range(len(contours) - 1):
        strip = stitch_two(offsets[k], len(contours[k]),
                           offsets[k + 1], len(contours[k + 1]))
        triangles.extend(strip)
        layers.extend([k] * len(strip))
    triangles = np.asarray(triangles, dtype=int)
    layers = np.asarray(layers, dtype=int)

    # consistent winding: all element normals point to the +ref (one fault face)
    ref = perp_horizontal(axis)
    triangles = orient_triangles(vertices, triangles, ref)

    patches = [TrianglePatch(*vertices[t], layer=int(lyr))
               for t, lyr in zip(triangles, layers)]

    print(f"[remesh] {len(vertices)} vertices, {len(triangles)} triangles, "
          f"{len(contours) - 1} layers")
    return RemeshedFault(patches=patches, vertices=vertices, triangles=triangles,
                         layers=layers, contours=contours, depths=depths,
                         datum=datum, name=name or path.stem, winding_ref=ref)


# --------------------------------------------------------------------------- #
#  Writers
# --------------------------------------------------------------------------- #
def write_gocad(rf, out_path, keep_elevation=False):
    """Write the remeshed fault as a GOCAD TSurf ``.txt`` matching the input.

    By default z is written in the datum-shifted frame (top contour at 0). With
    ``keep_elevation`` the original CFM elevations are restored.
    """
    out_path = Path(out_path)
    V = rf.vertices.copy()
    if keep_elevation:
        V[:, 2] += rf.datum
    lines = [
        "GOCAD TSurf 1",
        "HEADER {",
        f"name:{rf.name}",
        "*visible:true",
        "}",
        "GOCAD_ORIGINAL_COORDINATE_SYSTEM",
        "NAME Default",
        'AXIS_NAME "X" "Y" "Z"',
        'AXIS_UNIT "m" "m" "m"',
        "ZPOSITIVE Elevation",
        "END_ORIGINAL_COORDINATE_SYSTEM",
        f"# remeshed from CFM; datum elevation {rf.datum:.3f} m taken as z=0"
        + ("" if keep_elevation else " (z shifted)"),
        "TFACE",
    ]
    for i, (x, y, z) in enumerate(V, start=1):
        lines.append(f"VRTX {i} {x:.6f} {y:.6f} {z:.6f}")
    for a, b, c in rf.triangles + 1:                    # GOCAD is 1-based
        lines.append(f"TRGL {a} {b} {c}")
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[remesh] wrote GOCAD TSurf -> {out_path}")


def save_npz(rf, out_path):
    """Save arrays for fast reload / plotting."""
    out_path = Path(out_path)
    np.savez(out_path, vertices=rf.vertices, triangles=rf.triangles,
             layers=rf.layers, depths=rf.depths, datum=rf.datum, name=rf.name,
             winding_ref=(rf.winding_ref if rf.winding_ref is not None
                          else np.full(3, np.nan)))
    print(f"[remesh] wrote arrays -> {out_path}")


def load_npz(path):
    """Reload a :class:`RemeshedFault` from :func:`save_npz` output. Triangles
    are saved already wound consistently, so patches reload pre-oriented."""
    d = np.load(path, allow_pickle=True)
    V, T, L = d["vertices"], d["triangles"], d["layers"]
    patches = [TrianglePatch(*V[t], layer=int(lyr)) for t, lyr in zip(T, L)]
    wref = d["winding_ref"] if "winding_ref" in d.files else None
    if wref is not None and np.isnan(wref).any():
        wref = None
    return RemeshedFault(patches=patches, vertices=V, triangles=T, layers=L,
                         depths=d["depths"], datum=float(d["datum"]),
                         name=str(d["name"]), winding_ref=wref)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path,
                    help="GOCAD TSurf .txt CFM file, OR a directory to batch "
                         "(every *.txt that is not already a *_remesh.txt)")
    ap.add_argument("-o", "--output", type=Path,
                    help="single file: output .txt (default <input>_remesh.txt); "
                         "directory input: output dir (default alongside source)")
    ap.add_argument("--depths",
                    default="0,1500,3000,5500,8500,11500,15000",
                    help="comma-separated positive-down depths (m) for contours; "
                         f"depths below {PROJECT_BELOW_DEPTH:.0f} m are projected "
                         "vertically from the deepest CFM contour (Ridgecrest)")
    ap.add_argument("--n-top", type=int, default=40,
                    help="point count on the shallowest contour (even)")
    ap.add_argument("--n-bottom", type=int, default=8,
                    help="point count on the deepest contour (even)")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="endpoint snapping tolerance (m)")
    ap.add_argument("--keep-elevation", action="store_true",
                    help="write original CFM elevations instead of z=0 datum")
    ap.add_argument("--plot", action="store_true",
                    help="show an interactive 3D plot when done")
    args = ap.parse_args(argv)

    depths = [float(s) for s in args.depths.split(",") if s.strip() != ""]

    if args.input.is_dir():
        inputs = sorted(p for p in args.input.glob("*.txt")
                        if not p.stem.endswith("_remesh"))
        if not inputs:
            sys.exit(f"No *.txt CFM files in {args.input}")
        out_dir = args.output if args.output else None
        ok = 0
        for src in inputs:
            print(f"\n=== {src.name} ===")
            try:
                rf = remesh_fault(src, depths, args.n_top, args.n_bottom,
                                  tol=args.tol)
            except Exception as exc:                       # keep batch going
                print(f"[remesh] SKIPPED {src.name}: {exc}")
                continue
            dst = (out_dir / (src.stem + "_remesh.txt")) if out_dir \
                else src.with_name(src.stem + "_remesh.txt")
            dst.parent.mkdir(parents=True, exist_ok=True)
            write_gocad(rf, dst, keep_elevation=args.keep_elevation)
            save_npz(rf, dst.with_suffix(".npz"))
            ok += 1
        print(f"\n[remesh] batch done: {ok}/{len(inputs)} succeeded")
        return

    rf = remesh_fault(args.input, depths, args.n_top, args.n_bottom, tol=args.tol)
    out = args.output or args.input.with_name(args.input.stem + "_remesh.txt")
    write_gocad(rf, out, keep_elevation=args.keep_elevation)
    save_npz(rf, out.with_suffix(".npz"))

    if args.plot:
        from remeshFault_plot import plot_fault3d_mesh
        plot_fault3d_mesh(rf)


if __name__ == "__main__":
    main()
