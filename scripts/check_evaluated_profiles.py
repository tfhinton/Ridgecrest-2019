#!/usr/bin/env python3
'''Step through the evaluated profiles (evaluate_profiles.py) for visual
confirmation. Right/left arrows (or n/p) to step, q to quit.

Each screen shows the fault-aligned displacements with the quick picked
evaluation overlaid for comparison, the stacked shear strain used for the
relocation (with the detected fault-zone edges), and the profile location on
the optical minimap cached by pick_profiles.py.
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt

import config

EVALUATED_PICKLE = config.TMP_DIR / 'evaluated_profiles.pickle'
MINIMAP_NPZ = config.TMP_DIR / 'optical_minimap.npz'


class Stepper:
    def __init__(self, profiles, bg=None, extent=None, trace=None):
        self.profiles = profiles
        self.i = 0

        self.fig = plt.figure(figsize=(13, 7))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2.4, 1],
                                   height_ratios=[2, 1],
                                   left=0.07, right=0.98, top=0.92, bottom=0.09,
                                   hspace=0.25)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.axs = self.fig.add_subplot(gs[1, 0], sharex=self.ax)
        self.axmap = self.fig.add_subplot(gs[:, 1])

        if bg is not None:
            vmax = np.nanpercentile(np.abs(bg), 98)
            self.axmap.imshow(bg, extent=extent, origin='upper', cmap='RdBu_r',
                              vmin=-vmax, vmax=vmax, interpolation='nearest')
        if trace is not None:
            for g in trace.geometry:
                xy = np.asarray(g.coords)
                self.axmap.plot(xy[:, 0], xy[:, 1], 'k-', lw=0.6)
        self.axmap.set_aspect('equal')
        self.axmap.set_xticks([]); self.axmap.set_yticks([])
        self.profline, = self.axmap.plot([], [], 'r-', lw=2)
        self.profdot, = self.axmap.plot([], [], 'ro', ms=4)

        # drop matplotlib's default key bindings (left/right = back/forward,
        # p = pan), which would otherwise shadow ours
        if self.fig.canvas.manager is not None:
            self.fig.canvas.mpl_disconnect(
                self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._show()

    def _show(self):
        p = self.profiles[self.i]
        q = p.picked

        self.ax.clear()
        # quick picked evaluation for comparison: its xs=0 is on the DRAWN
        # trace (shift into the relocated frame) and its parallel row is the
        # projection onto -strike (flip to match the fault-aligned sign).
        self.ax.plot(q.xs - p.total_shift, -q.displacements[0], '-',
                     color='lightpink', lw=0.8, label='quick parallel (aligned)')
        self.ax.plot(q.xs - p.total_shift, q.displacements[1], '-',
                     color='lightskyblue', lw=0.8, label='quick normal (aligned)')
        self.ax.fill_between(p.xs, p.displacements[0] - p.displacements_std[0],
                             p.displacements[0] + p.displacements_std[0],
                             color='crimson', alpha=0.12, lw=0)
        self.ax.plot(p.xs, p.displacements[0], '-', color='crimson', lw=1.2,
                     label='fault-parallel')
        self.ax.plot(p.xs, p.displacements[1], '-', color='steelblue', lw=1.2,
                     label='fault-normal')
        self.ax.axvline(0., color='lightgray', ls='--', zorder=0)
        self.ax.set_ylabel('Displacement (m)')
        self.ax.legend(loc='upper right', fontsize=8)

        self.axs.clear()
        self.axs.plot(p.xs, p.strain_shear, '-', color='darkorange', lw=1.,
                      label='shear strain (detrended)')
        # NB the peak sits at x=0 BY CONSTRUCTION (profiles are aligned on
        # their strain maxima before stacking); judge it against the threshold
        # line and the displacement step, not by its tightness.
        thr = getattr(p, 'strain_threshold', None)
        if thr is not None and np.isfinite(thr):
            self.axs.axhline(thr, color='gray', lw=0.7, ls='--',
                             label='FZW threshold (3$\\sigma$ bg)')
        for xedge in (p.x_min, p.x_max):
            if np.isfinite(xedge):
                self.axs.axvline(xedge, color='gray', ls=':', lw=1.)
        self.axs.axvline(0., color='lightgray', ls='--', zorder=0)
        self.axs.set_xlabel('Distance from relocated fault (m)')
        self.axs.set_ylabel('Shear strain')
        self.axs.legend(loc='upper right', fontsize=8)

        fzw = f'{p.fzw:.0f} m' if np.isfinite(p.fzw) else 'n/a'
        shift = f'trace shift {p.total_shift:.0f} m'
        if getattr(p, 'step_shift', 0.):
            shift += f' (step {p.step_shift:.0f} + strain {p.total_shift - p.step_shift:.0f})'
        self.ax.set_title(
            f'{self.i + 1}/{len(self.profiles)}   fault {p.fault_id}   '
            f'along-strike {p.x_along_fault / 1000.:.1f} km   '
            f'offset {p.offset_near:.2f}±{p.offset_near_std:.2f} m   '
            f'FZW {fzw}   {shift}')

        xy = np.asarray(p.linestring.coords)
        self.profline.set_data(xy[:, 0], xy[:, 1])
        self.profdot.set_data([p.fault_utm_refined[0]], [p.fault_utm_refined[1]])
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ('right', 'n'):
            self.i = min(len(self.profiles) - 1, self.i + 1)
            self._show()
        elif event.key in ('left', 'p'):
            self.i = max(0, self.i - 1)
            self._show()
        elif event.key == 'q':
            plt.close(self.fig)


def main():
    profiles = pickle.load(open(EVALUATED_PICKLE, 'rb'))
    print(f'{len(profiles)} evaluated profiles')

    bg = extent = trace = None
    if MINIMAP_NPZ.exists():
        z = np.load(MINIMAP_NPZ)
        bg, extent = z['bg'], z['extent']
    if config.FAULT_PICKLE.exists():
        trace = pickle.load(open(config.FAULT_PICKLE, 'rb')).trace

    # keep a reference: mpl_connect only holds the callback weakly
    stepper = Stepper(profiles, bg=bg, extent=extent, trace=trace)
    plt.show()
    return stepper


if __name__ == '__main__':
    main()
