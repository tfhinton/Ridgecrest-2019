#!/usr/bin/env python3
"""Extract the surface trace of a GOCAD TSurf triangular-mesh fault.

Two modes:

  top   (default) -- the top edge of the modelled fault, which the CFM5 meshes
        clip to the topographic ground surface. A boundary edge is one used by
        a single triangle; a top edge is a boundary edge whose triangle hangs
        below it (the opposite vertex is deeper than both endpoints). This
        follows topography and keeps each vertex's elevation. No DEM required.

  slice -- the intersection of the mesh with a horizontal plane (--z-level),
        e.g. Z = 0 for a sea-level trace or -1000 for a depth contour.

Either way, segments are stitched into continuous polylines and written as
GeoJSON (and optionally a WKT CSV) that drags straight into QGIS.

Usage:
    python extract_surface_trace.py INPUT.txt [-o OUTPUT.geojson]
            [--mode top|slice] [--z-level 0.0] [--epsg 32611] [--csv]

The CFM5 files use UTM zone 11N coordinates in metres, hence the EPSG:32611
default. Z is positive-up (elevation), per the file's ZPOSITIVE Elevation flag.
"""
import argparse
import json
import sys
from pathlib import Path


def parse_tsurf(path):
    """Return (verts, tris): verts is {id: (x, y, z)}, tris is list of (i, j, k)."""
    verts = {}
    tris = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag in ("VRTX", "PVRTX"):
                # VRTX id x y z  (PVRTX may carry extra trailing properties)
                vid = int(parts[1])
                verts[vid] = (float(parts[2]), float(parts[3]), float(parts[4]))
            elif tag == "TRGL":
                tris.append((int(parts[1]), int(parts[2]), int(parts[3])))
    return verts, tris


def triangle_plane_segments(verts, tris, z_level):
    """Intersect each triangle with the plane Z = z_level; return list of segments.

    Each segment is ((x0, y0), (x1, y1)).
    """
    segments = []
    for tri in tris:
        try:
            p = [verts[i] for i in tri]
        except KeyError:
            continue  # triangle references a vertex we never saw
        d = [v[2] - z_level for v in p]

        crossings = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            da, db = d[a], d[b]
            if (da > 0) == (db > 0):
                continue  # both on same side (or both exactly on plane handled below)
            if da == db:
                continue
            t = da / (da - db)  # parameter where Z crosses z_level
            x = p[a][0] + t * (p[b][0] - p[a][0])
            y = p[a][1] + t * (p[b][1] - p[a][1])
            crossings.append((x, y))

        # A clean crossing produces exactly two edge intersections.
        if len(crossings) == 2:
            segments.append((crossings[0], crossings[1]))
    return segments


def top_edges(verts, tris, depth_frac=0.5, dip_max=45.0):
    """Return the mesh's top boundary edges as segments ((x,y,z), (x,y,z)).

    A boundary edge is used by a single triangle; together they trace the mesh
    perimeter -- the top edge, the deep bottom edge and the two lateral edges.
    The top edge is isolated by two conditions:

      * orientation -- it runs sub-horizontally along strike, so its dip
        (atan |dz| / horizontal length) is below dip_max; the lateral edges
        plunge steeply down-dip and are rejected.
      * elevation -- its midpoint lies in the upper part of the mesh depth
        range (above zmin + depth_frac * (zmax - zmin)); this rejects the
        equally shallow-dipping bottom edge.

    The two together follow topography and stop cleanly at the fault tips, with
    no DEM needed -- the CFM5 meshes are already clipped to the ground surface.
    """
    import math
    from collections import defaultdict

    # edge (sorted vertex ids) -> count of triangles using it
    edge_count = defaultdict(int)
    for tri in tris:
        if not all(i in verts for i in tri):
            continue
        a, b, c = tri
        for u, v in ((a, b), (b, c), (c, a)):
            edge_count[tuple(sorted((u, v)))] += 1

    zs = [p[2] for p in verts.values()]
    z_cut = min(zs) + depth_frac * (max(zs) - min(zs))

    segments = []
    for (u, v), n in edge_count.items():
        if n != 1:
            continue  # interior edge, shared by two triangles
        a, b = verts[u], verts[v]
        if (a[2] + b[2]) / 2 <= z_cut:
            continue  # bottom edge
        dl = math.hypot(a[0] - b[0], a[1] - b[1])
        dip = math.degrees(math.atan2(abs(a[2] - b[2]), dl)) if dl else 90.0
        if dip > dip_max:
            continue  # steep lateral (tip) edge
        segments.append((a, b))
    return segments


def stitch_segments(segments, tol=1.0):
    """Join segments sharing endpoints into continuous polylines.

    tol is the snapping distance (metres) for treating two endpoints as equal.
    """
    def key(pt):
        return (round(pt[0] / tol), round(pt[1] / tol))

    # adjacency: node -> list of (neighbor_node, point_a, point_b)
    from collections import defaultdict

    adj = defaultdict(list)
    coords = {}
    for a, b in segments:
        ka, kb = key(a), key(b)
        coords[ka], coords[kb] = a, b
        adj[ka].append(kb)
        adj[kb].append(ka)

    visited_edges = set()

    def edge_id(u, v):
        return (u, v) if u <= v else (v, u)

    lines = []
    # Start traces from endpoints (degree 1) first, then any remaining loops.
    starts = [n for n in adj if len(adj[n]) == 1] + list(adj.keys())
    for start in starts:
        for nxt in adj[start]:
            if edge_id(start, nxt) in visited_edges:
                continue
            line = [coords[start]]
            u, v = start, nxt
            while True:
                visited_edges.add(edge_id(u, v))
                line.append(coords[v])
                # pick an unvisited continuation
                nxts = [w for w in adj[v] if edge_id(v, w) not in visited_edges]
                if not nxts:
                    break
                u, v = v, nxts[0]
            if len(line) >= 2:
                lines.append(line)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="GOCAD TSurf .txt file")
    ap.add_argument("-o", "--output", type=Path,
                    help="output GeoJSON path (default: <input>_trace.geojson)")
    ap.add_argument("--mode", choices=("top", "slice"), default="top",
                    help="'top' = top edge of mesh at topography (default); "
                         "'slice' = intersection with the --z-level plane")
    ap.add_argument("--z-level", type=float, default=0.0,
                    help="elevation of the plane to slice in 'slice' mode (default 0)")
    ap.add_argument("--depth-frac", type=float, default=0.5,
                    help="'top' mode: keep boundary edges in the upper this "
                         "fraction of the mesh depth range (default 0.5)")
    ap.add_argument("--dip-max", type=float, default=45.0,
                    help="'top' mode: max edge dip in degrees; steeper lateral "
                         "tip edges are excluded (default 45)")
    ap.add_argument("--epsg", type=int, default=32611,
                    help="EPSG code of the input coords (default 32611 = UTM 11N)")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="endpoint snapping tolerance in metres (default 1)")
    ap.add_argument("--csv", action="store_true",
                    help="also write a WKT CSV alongside the GeoJSON")
    args = ap.parse_args(argv)

    verts, tris = parse_tsurf(args.input)
    if not tris:
        sys.exit(f"No TRGL triangles found in {args.input}")

    if args.mode == "top":
        segments = top_edges(verts, tris, depth_frac=args.depth_frac,
                             dip_max=args.dip_max)
        if not segments:
            sys.exit("No top boundary edges found.")
    else:
        segments = triangle_plane_segments(verts, tris, args.z_level)
        if not segments:
            sys.exit(f"Mesh does not cross Z = {args.z_level}; no surface trace.")

    lines = stitch_segments(segments, tol=args.tol)

    out = args.output or args.input.with_name(args.input.stem + "_trace.geojson")
    fc = {
        "type": "FeatureCollection",
        # Legacy CRS member; QGIS honours it on import.
        "crs": {"type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{args.epsg}"}},
        "features": [
            {"type": "Feature",
             "properties": {"name": args.input.stem, "mode": args.mode},
             "geometry": {"type": "LineString",
                          "coordinates": [list(pt) for pt in line]}}
            for line in lines
        ],
    }
    out.write_text(json.dumps(fc))

    total_len = sum(
        sum(((line[i + 1][0] - line[i][0]) ** 2 +
             (line[i + 1][1] - line[i][1]) ** 2) ** 0.5
            for i in range(len(line) - 1))
        for line in lines
    )
    where = "topographic top" if args.mode == "top" else f"Z={args.z_level}"
    print(f"{len(segments)} segments -> {len(lines)} polylines "
          f"({total_len / 1000:.1f} km total) at {where}")
    print(f"Wrote {out}  (EPSG:{args.epsg})")

    if args.csv:
        csv_path = out.with_suffix(".csv")
        with open(csv_path, "w") as fh:
            fh.write("id;wkt\n")
            for n, line in enumerate(lines):
                wkt = ("LINESTRING(" +
                       ", ".join(f"{pt[0]} {pt[1]}" for pt in line) + ")")
                fh.write(f"{n};{wkt}\n")
        print(f"Wrote {csv_path}  (delimiter ';', WKT geometry)")


if __name__ == "__main__":
    main()
