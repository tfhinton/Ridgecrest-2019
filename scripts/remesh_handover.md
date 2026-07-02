# Handover: triangular patch format from `remeshFault.py`

For implementing Meade-style triangular dislocation element (TDE) Green's
functions against the mesh produced by
[`scripts/remeshFault.py`](remeshFault.py). The plotting module
[`scripts/remeshFault_plot.py`](remeshFault_plot.py) is irrelevant to the GF work.

## What you get

`remesh_fault(...)` returns a `RemeshedFault`; `load_npz(path)` rebuilds the same
object from the saved `.npz`. The two things you care about:

- `rf.patches` — a list of `TrianglePatch` (one per triangle).
- `rf.vertices` (V,3) float + `rf.triangles` (T,3) int — the same mesh as a
  shared vertex array + 0-based connectivity (better for vectorised GF builds,
  since many triangles share vertices).

```python
from remeshFault import load_npz
rf = load_npz("SNFA-LLFZ-EAST-..._remesh.npz")
V, T = rf.vertices, rf.triangles          # (V,3) metres, (T,3) int 0-based
for p in rf.patches:
    tri = p.vertices                      # (3,3): rows are (x,y,z) in metres
```

**Available meshes** (already built; in `data/fault/`, one `.txt` + `.npz` each):

| strand | file stem | datum (m) | verts / tris |
|---|---|---|---|
| Eastern Little Lake | `SNFA-LLFZ-EAST-..._remesh` | 542.366 | 206 / 340 |
| Southern Little Lake | `SNFA-LLFZ-SOUT-..._remesh` | 564.659 | 206 / 340 |

Re-run / re-tune any strand (or the whole folder) with
`python remeshFault.py data/fault [--depths ...] [--n-top N] [--n-bottom N]`.

## Coordinate conventions (read this carefully — sign/frame bugs live here)

- **Horizontal**: UTM zone 11N, **EPSG:32611**, metres. `x` = easting,
  `y` = northing. (Same CRS as the source CFM.)
- **Vertical**: `z = original_CFM_elevation - datum`, in metres.
  - `datum` = the lowest point of that strand's topographic top trace (per-fault;
    542.366 m for EAST, 564.659 m for SOUT). Read it from `rf.datum`.
  - So the mesh top (the "zero" depth contour) sits at **z = 0**, and the fault
    extends **downward to negative z** (z is positive-**up**, datum-shifted).
  - This already matches the half-space convention used by Meade (2007) and the
    Nikkhoo–Walter TDE codes: **free surface at z = 0, sources at z ≤ 0**. You
    can feed `p.vertices` straight in if your TDE code expects elevation/up-z.
  - If your TDE implementation instead wants **depth positive-down**, use
    `p.depth_vertices()` → returns the same triangle with `z → -z` (z ≥ 0 down).
- Units are **metres everywhere** (vertices, areas, depths). No km scaling is
  applied in the data — only the plotter rescales for display.

The datum shift is recorded in the GOCAD `.txt` header comment; `rf.datum` holds
it numerically. To recover true CFM elevation: `elev = z + rf.datum`.

## `TrianglePatch` API

```
p.vertices         # (3,3) ndarray, rows (x,y,z) metres, z up / datum-shifted
p.depth_vertices() # (3,3) with z positive-down (z >= 0)
p.centroid         # (3,) mean of the three vertices
p.area             # scalar m^2  (0.5 |(v1-v0) x (v2-v0)|)
p.unit_normal      # (3,) unit normal; consistently wound (see below)
p.depth            # scalar, mean depth positive-down (= -mean z)
p.layer            # int: depth band index (0 = shallowest band)
```

`rf.layers` (T,) gives the same `layer` per triangle in array form.
`rf.winding_ref` (3,) is the reference vector all element normals point toward.

## Mesh structure you can rely on

- The mesh is built as horizontal **depth bands** between resampled contours.
  `rf.depths` (K,) lists the contour depths (positive-down, e.g. the default
  `[0,1000,2000,3000,4500,6500,9000,12000,15000]`); band/layer `k` lies between
  contour `k` and `k+1`. Default mesh: 320 vertices, 562 triangles, 8 layers.
- **Vertices are shared** between adjacent bands (the bottom row of band `k` is
  literally the top row of band `k+1`), so the surface is watertight/continuous
  and **separable** at any contour by filtering on `layer`. No duplicate
  coincident vertices to merge.
- Contours coarsen with depth (even point counts, e.g. 50→46→40→34→24→12), so
  triangles get larger downward.

## Caveats that matter for TDEs

1. **Winding is already consistent — but confirm it matches your TDE's
   convention.** The remesher now enforces a uniform winding: every triangle is
   ordered so its normal `(v1-v0) x (v2-v0)` points to the +`rf.winding_ref`
   side (a horizontal vector perpendicular to strike, i.e. all normals face the
   same fault wall). Verified: all 340 normals share sign; the raw strip stitch
   was genuinely mixed (≈189/151) before. This means `p.unit_normal` is stable
   across the mesh and the strike-slip/dip-slip/tensile sign convention of a TDE
   is coherent. **You still must check that "+winding_ref side" equals the
   positive sense your TDE code expects** — if a uniform sign flip of your dip-
   slip/tensile GFs looks needed, flip `winding_ref` and re-orient rather than
   patching signs downstream. To re-orient a loaded/edited mesh (e.g. with your
   own reference), call `remeshFault.enforce_consistent_winding(rf, ref=...)`;
   meshes loaded via `load_npz` are already oriented (triangles are saved wound).
2. **No strike/dip metadata is stored.** If your TDE formulation needs a local
   strike/dip frame per element, derive it from the vertices, e.g. the fault
   strike from the horizontal projection of the longest edge, dip from the
   normal's plunge. (The whole fault is near-vertical strike-slip, dipping
   ~steeply ENE; per-triangle dip varies slightly.)
3. **Slip vector is yours to define.** Patches carry geometry only — no slip.
   You'll attach a slip (ss, ds, ts) per patch when assembling GFs.
4. The deep contours taper in strike (the fault narrows with depth). The CFM is
   only fully defined along strike to ~9 km, so **contours below 9 km are
   projected vertically straight down from the 9 km contour** (same x-y, deeper
   z) rather than sliced from the CFM — the deep fault is a vertical wall from
   9 km to the max depth (15 km by default). Nothing to handle on your side, just
   don't be surprised that (a) deep bands are narrower, and (b) the 9/12/15 km
   contours share the same map trace. (Requested *geometry* depths below a
   strand's actual extent are still auto-dropped, so strands may differ in layer
   count; projected depths are always kept.)

## File/format reference

- GOCAD `.txt` output (matches input format): standard `VRTX id x y z` (1-based)
  + `TRGL a b c` (1-based). z is datum-shifted unless written with
  `--keep-elevation`. Parse with `extract_surface_trace.parse_tsurf`.
- `.npz` keys: `vertices` (V,3), `triangles` (T,3 int, 0-based, wound), `layers`
  (T,), `depths` (K,), `datum` (float), `name` (str), `winding_ref` (3,).
- `TrianglePatch` / `RemeshedFault` / `load_npz` / `enforce_consistent_winding`
  all live in `remeshFault.py`.
