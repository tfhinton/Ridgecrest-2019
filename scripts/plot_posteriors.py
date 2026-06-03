#%%

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import gaussian_kde
from codes import Styles, HamiltonianInversion

Styles.set_styles()


####    FILEPATHS    ####
data_dir = Path("/data/cycle/hintont/projects/ridgecrest/profile_inversion/in01/")


####    PARAMETER METADATA    ####
param_names = ["dz_halfwidth", "modulus_ratio", "slip0", "slip1", "slip2"]
param_labels = [
    "DZ half-width (m)",
    "Modulus ratio",
    "Slip$_0$ (m)",
    "Slip$_1$ (m)",
    "Slip$_2$ (m)",
]


####    LOOP PROFILES    ####
profile_paths = sorted(
    (p for p in data_dir.glob("profile_*.pickle") if p.stem.split("_")[1].isdigit()),
    key=lambda p: int(p.stem.split("_")[1]),
)

for pkl_path in profile_paths:
    i = int(pkl_path.stem.split("_")[1])

    with open(pkl_path, "rb") as f:
        inv = pickle.load(f)

    if inv.result is None:
        print(f"Profile {i}: no result, skipping")
        continue

    ####    POSTERIOR SAMPLES    ####
    samples = {
        name: inv.result.posterior[name].values.flatten()
        for name in param_names
    }

    ####    MEDIAN MODEL FIT    ####
    median_params = np.array([np.median(samples[name]) for name in param_names])
    xs = inv.forward.xs          # metres, fault-perpendicular
    pred = inv.forward.pred_func(median_params)
    data = inv.data

    ####    FIGURE LAYOUT: 2-row × 5-col grid    ####
    # Cols 0-1: data+fit (spans both rows)
    # Col 2, row 0: dz_halfwidth   Col 3, row 0: modulus_ratio   Col 4, row 0: empty
    # Col 2, row 1: slip0          Col 3, row 1: slip1            Col 4, row 1: slip2
    fig = plt.figure(figsize=(10, 4.5))
    gs = gridspec.GridSpec(
        2, 5,
        figure=fig,
        left=0.07, right=0.97,
        top=0.90, bottom=0.13,
        wspace=0.1, hspace=0.1,
        width_ratios=[1, 1, 1, 1, 1],
    )

    ax_fit = fig.add_subplot(gs[:, 0:2])

    kde_axes = [
        fig.add_subplot(gs[0, 2]),   # dz_halfwidth
        fig.add_subplot(gs[0, 3]),   # modulus_ratio
        # gs[0, 4] intentionally left empty
        fig.add_subplot(gs[1, 2]),   # slip0
        fig.add_subplot(gs[1, 3]),   # slip1
        fig.add_subplot(gs[1, 4]),   # slip2
    ]

    ####    DATA + MODEL FIT    ####
    ax_fit.scatter(xs / 1e3, data, s=5, color="C0", alpha=0.55, label="Data", zorder=2, linewidths=0)
    ax_fit.plot(xs / 1e3, pred, color="C1", linewidth=1.8, label="Median fit", zorder=3)
    ax_fit.axhline(0., color="0.6", linewidth=0.7, linestyle=":")
    ax_fit.axvline(0., color="0.6", linewidth=0.7, linestyle=":")
    ax_fit.set_xlabel("Fault-perpendicular distance (km)")
    ax_fit.set_ylabel("Displacement (m)")
    ax_fit.legend(fontsize=8, frameon=False)
    ax_fit.set_title(f"Profile {i}", fontsize=10)

    ####    POSTERIOR KDEs    ####
    for ax, name, label in zip(kde_axes, param_names, param_labels):
        s = samples[name]
        kde = gaussian_kde(s, bw_method="scott")
        x_grid = np.linspace(s.min(), s.max(), 300)
        y_grid = kde(x_grid)
        ax.plot(x_grid, y_grid, color="C0", linewidth=1.5)
        ax.fill_between(x_grid, y_grid, alpha=0.20, color="C0")
        ax.axvline(np.median(s), color="C1", linewidth=1.2, linestyle="--")
        ax.set_xlabel(label, fontsize=8)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7)
        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)

    ####    SAVE    ####
    save_path = data_dir / f"profile_{i}.png"
    fig.savefig(save_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path.name}")

print("Done.")
