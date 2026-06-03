#%%

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
from codes import Styles

Styles.set_styles()


####    FILEPATHS    ####
data_dir = Path("/data/cycle/hintont/projects/ridgecrest/profile_inversion/in02/")


####    PARAMETER METADATA    ####
param_names = ["dz_halfwidth", "modulus_ratio", "slip0", "slip1", "slip2"]
param_labels = [
    "DZ half-width (m)",
    "Modulus ratio",
    r"Slip$_0$ (m)",
    r"Slip$_1$ (m)",
    r"Slip$_2$ (m)",
]


####    PSET SPECS (mirrors profile_inversion_check_convergence.py)    ####
pset_specs = {
    0: {"draws": 25,   "tune": 10,  "chains": 4},
    1: {"draws": 100,  "tune": 40,  "chains": 4},
    2: {"draws": 250,  "tune": 100, "chains": 4},
    3: {"draws": 1000, "tune": 400, "chains": 4},
    4: {"draws": 100,  "tune": 40,  "chains": 10},
    5: {"draws": 25,   "tune": 10,  "chains": 40},
}


def _chain_colors(n):
    if n <= 10:
        return plt.cm.tab10(np.linspace(0, 0.9, max(n, 1)))
    return plt.cm.rainbow(np.linspace(0, 1, n))


def _kde(samples, n_pts=300):
    if len(samples) < 2 or np.std(samples) == 0:
        return None
    kde = gaussian_kde(samples, bw_method="scott")
    x = np.linspace(samples.min(), samples.max(), n_pts)
    return x, kde(x)


####    LOAD ALL PSETS    ####
inversions = {}
for pkl_path in sorted(data_dir.glob("pset_*.pickle"),
                       key=lambda p: int(p.stem.split("_")[1])):
    j = int(pkl_path.stem.split("_")[1])
    with open(pkl_path, "rb") as f:
        inversions[j] = pickle.load(f)

print(f"Loaded {len(inversions)} parameter sets.")


####    PER-PSET DIAGNOSTIC PLOTS    ####
for j, inv in sorted(inversions.items()):
    if inv.result is None:
        print(f"pset_{j}: no result, skipping")
        continue

    posterior = inv.result.posterior
    n_chains = posterior[param_names[0]].shape[0]
    n_draws  = posterior[param_names[0]].shape[1]

    has_warmup = "warmup_posterior" in inv.result.groups()
    if has_warmup:
        n_tune = inv.result.warmup_posterior[param_names[0]].shape[1]

    spec   = pset_specs.get(j, {})
    colors = _chain_colors(n_chains)

    # Scale lw/alpha for legibility at high chain counts
    lw_kde   = max(0.4, 1.2  - 0.02  * n_chains)
    lw_trace = max(0.3, 0.8  - 0.01  * n_chains)
    al_kde   = max(0.20, 0.85 - 0.015 * n_chains)
    al_trace = max(0.20, 0.75 - 0.012 * n_chains)

    fig, axes = plt.subplots(2, 5, figsize=(12.5, 4.5))
    fig.suptitle(
        (f"pset {j} — draws={spec.get('draws', '?')}, "
         f"tune={spec.get('tune', '?')}, chains={spec.get('chains', '?')}"),
        fontsize=11,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12,
                        wspace=0.45, hspace=0.55)

    for col, (name, lbl) in enumerate(zip(param_names, param_labels)):
        ax_kde   = axes[0, col]
        ax_trace = axes[1, col]

        for c in range(n_chains):
            chain_samps = posterior[name].values[c, :]
            color = colors[c]

            # KDE per chain
            result = _kde(chain_samps)
            if result is not None:
                x_k, y_k = result
                ax_kde.plot(x_k, y_k, color=color,
                            linewidth=lw_kde, alpha=al_kde)

            # Trace per chain
            if has_warmup:
                warmup_samps = inv.result.warmup_posterior[name].values[c, :]
                ax_trace.plot(np.arange(-n_tune, 0), warmup_samps,
                              color=color, linewidth=lw_trace,
                              alpha=al_trace * 0.55)
                ax_trace.plot(np.arange(n_draws), chain_samps,
                              color=color, linewidth=lw_trace,
                              alpha=al_trace)
            else:
                ax_trace.plot(np.arange(n_draws), chain_samps,
                              color=color, linewidth=lw_trace,
                              alpha=al_trace)

        if has_warmup:
            ax_trace.axvline(0, color="0.25", linewidth=1.0,
                             linestyle="--", zorder=5)
            ax_trace.set_xlabel("Draw  (0 = tuning cutoff)", fontsize=8)
        else:
            ax_trace.set_xlabel("Draw", fontsize=8)

        ax_kde.set_xlabel(lbl, fontsize=8)
        ax_kde.set_yticks([])
        ax_kde.tick_params(axis="x", labelsize=7)

        ax_trace.set_ylabel(lbl, fontsize=7)
        ax_trace.tick_params(axis="both", labelsize=7)

        if col == 0:
            ax_kde.set_ylabel("Density", fontsize=8)

    save_path = data_dir / f"pset_{j}.png"
    fig.savefig(save_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path.name}")


####    SUMMARY PLOT    ####
# Adjust this to move the legend up (less negative) or further down (more negative)
LEGEND_Y = -0.1

summary_colors = plt.cm.tab10(np.linspace(0, 0.9, 6))

fig_sum, axes_sum = plt.subplots(1, 5, figsize=(12, 3.))
fig_sum.suptitle("Convergence check — posterior distributions across parameter sets",
                 fontsize=10)
fig_sum.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.18,
                        wspace=0.40)

sorted_items = sorted(
    [(j, inv) for j, inv in inversions.items() if inv.result is not None],
    key=lambda x: x[0],
)

for idx, (j, inv) in enumerate(sorted_items):
    spec  = pset_specs.get(j, {})
    color = summary_colors[j % len(summary_colors)]
    d, c  = spec.get('draws', '?'), spec.get('chains', '?')
    label = f"draws={d}, chains={c}" if idx == 0 else f"d={d}, c={c}"

    for col, (name, lbl) in enumerate(zip(param_names, param_labels)):
        ax = axes_sum[col]
        samples = inv.result.posterior[name].values.flatten()
        result = _kde(samples)
        if result is None:
            continue
        x_k, y_k = result
        ax.plot(x_k, y_k, color=color, linewidth=1.5,
                label=label if col == 0 else "_nolegend_")
        ax.set_xlabel(lbl, fontsize=9)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=8)

axes_sum[0].set_ylabel("Density", fontsize=9)

handles, labels_leg = axes_sum[0].get_legend_handles_labels()
fig_sum.legend(
    handles, labels_leg,
    loc="lower center",
    ncol=len(sorted_items),
    frameon=False,
    fontsize=8,
    bbox_to_anchor=(0.5, LEGEND_Y),
)

save_path_sum = data_dir / "convergence_summary.png"
fig_sum.savefig(save_path_sum, dpi=450, bbox_inches="tight")
plt.close(fig_sum)
print(f"Saved {save_path_sum.name}")

print("Done.")
