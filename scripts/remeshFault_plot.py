#!/usr/bin/env python3
"""Interactive 3D plotting for the depth-layered triangular fault mesh produced
by ``remeshFault.py``.

Kept separate from the remeshing so a heavy meshing run need not import
matplotlib, and so plots can be regenerated from a saved ``.npz`` without
recomputing the mesh.

Usage
-----
    python remeshFault_plot.py MESH.npz [--color-by layer|depth] [--save out.png]

or from code::

    from remeshFault import remesh_fault
    from remeshFault_plot import plot_fault3d_mesh
    rf = remesh_fault("fault.txt", [0, 1000, 2500, 5000], n_top=50, n_bottom=12)
    plot_fault3d_mesh(rf)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _set_equal_aspect(ax, V):
    """Give the 3D axes a 1:1:1 data aspect (true fault geometry)."""
    mins, maxs = V.min(axis=0), V.max(axis=0)
    centre = (mins + maxs) / 2
    r = (maxs - mins).max() / 2
    ax.set_xlim(centre[0] - r, centre[0] + r)
    ax.set_ylim(centre[1] - r, centre[1] + r)
    ax.set_zlim(centre[2] - r, centre[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_fault3d_mesh(rf, ax=None, color_by="layer", cmap="viridis",
                      edgecolor="k", linewidth=0.2, alpha=1.0,
                      show_contours=True, km=True):
    """Render the remeshed fault as a 3D triangular surface.

    Parameters
    ----------
    rf : a ``RemeshedFault`` (has ``.vertices``, ``.triangles``, ``.layers``,
        ``.contours``), or anything exposing those attributes.
    color_by : ``"layer"`` (discrete depth band) or ``"depth"`` (continuous
        patch-centroid depth, positive down).
    show_contours : overlay the shared resampled depth contours as thick lines.
    km : scale axes to kilometres for readability (geometry unchanged).
    """
    V = np.asarray(rf.vertices, dtype=float)
    T = np.asarray(rf.triangles, dtype=int)
    scale = 1e-3 if km else 1.0
    Vp = V * scale

    tri_xyz = Vp[T]                                  # (T, 3, 3)

    if color_by == "depth":
        vals = -tri_xyz[:, :, 2].mean(axis=1)        # positive down
        label = "depth (km)" if km else "depth (m)"
    else:
        vals = np.asarray(rf.layers, dtype=float)
        label = "layer"

    norm = plt.Normalize(vals.min(), vals.max())
    facecolors = plt.get_cmap(cmap)(norm(vals))

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    coll = Poly3DCollection(tri_xyz, facecolors=facecolors,
                            edgecolors=edgecolor, linewidths=linewidth,
                            alpha=alpha)
    ax.add_collection3d(coll)

    if show_contours and getattr(rf, "contours", None) is not None:
        for c in rf.contours:                        # ragged lengths -> plot each
            c = np.asarray(c) * scale
            ax.plot(c[:, 0], c[:, 1], c[:, 2], color="k", linewidth=1.0)

    _set_equal_aspect(ax, Vp)
    ax.set_xlabel("E (km)" if km else "E (m)")
    ax.set_ylabel("N (km)" if km else "N (m)")
    ax.set_zlabel("z (km)" if km else "z (m)")
    ax.set_title(getattr(rf, "name", "remeshed fault"))

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(vals)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(label)
    fig.tight_layout()
    return ax


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, help=".npz written by remeshFault.save_npz")
    ap.add_argument("--color-by", choices=("layer", "depth"), default="layer")
    ap.add_argument("--save", type=Path, help="save a PNG instead of showing")
    args = ap.parse_args(argv)

    from remeshFault import load_npz
    rf = load_npz(args.npz)
    plot_fault3d_mesh(rf, color_by=args.color_by)
    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
