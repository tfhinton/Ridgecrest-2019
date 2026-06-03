#%%

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from matplotlib import colors as mplcolors
import matplotlib.cm as cm
import cmcrameri.cm as cmc
from codes import Styles, HamiltonianInversion

Styles.set_styles()


####    FILEPATHS    ####
data_dir = Path("/data/cycle/hintont/projects/ridgecrest/profile_inversion/in01/")
save_path = data_dir / "dz_along_strike.png"


####    PARAMETERS    ####
rotation_angle = 43.         # degrees anticlockwise for the optical map view
lon_c = -117.60              # map/rotation center longitude
lat_c = 35.70                # map/rotation center latitude
xlim_rot = (-0.29, 0.32)    # rotated x-extent as offsets from center (degrees)
ylim_rot = (-0.05, 0.14)    # rotated y-extent as offsets from center (degrees)
optical_clim = (-1.5, 1.5)  # EW displacement colorbar limits (m)

# Colorbar appearance
cbar_width_frac = 0.025      # colorbar width relative to main plot (GridSpec ratio)
cbar_pad_left   = 0.02      # extra left gap added after layout (figure fraction)
cbar_pad_tb     = 0.12       # top/bottom padding as fraction of colorbar height

# Profile overlay
profile_swathe_half_width_m = 500.   # metres (0.5 km)


####    LOAD DATA    ####
profiles, along_strike_xs = pickle.load(open(data_dir / "profile_geoms.pickle", "rb"))
optical = pickle.load(open(data_dir / "optical.pickle", "rb"))
along_strike_xs = np.array(along_strike_xs)

n_profiles = len(profiles)


####    LOAD INVERSIONS AND EXTRACT POSTERIORS    ####
dz_med = np.full(n_profiles, np.nan)
dz_lo  = np.full(n_profiles, np.nan)
dz_hi  = np.full(n_profiles, np.nan)
mr_med = np.full(n_profiles, np.nan)

for i in range(n_profiles):
    pkl_path = data_dir / f"profile_{i}.pickle"
    if not pkl_path.exists():
        continue
    with open(pkl_path, "rb") as f:
        inv = pickle.load(f)
    if inv.result is None:
        continue
    dz = inv.result.posterior["dz_halfwidth"].values.flatten()
    mr = inv.result.posterior["modulus_ratio"].values.flatten()
    dz_med[i] = np.median(dz)
    dz_lo[i]  = np.percentile(dz, 16)
    dz_hi[i]  = np.percentile(dz, 84)
    mr_med[i] = np.median(mr)

valid = ~np.isnan(dz_med)


####    FIGURE LAYOUT    ####
fig = plt.figure(figsize=(10, 6.5))
gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1, cbar_width_frac],
                       wspace=0.02, hspace=0.2)
ax_dz   = fig.add_subplot(gs[0, 0])
cax_dz  = fig.add_subplot(gs[0, 1])
ax_opt  = fig.add_subplot(gs[1, 0])
cax_opt = fig.add_subplot(gs[1, 1])


####    TOP ROW: DZ HALF-WIDTH ALONG STRIKE    ####
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    return mplcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )

cmap_mr = truncate_colormap(cmc.batlow, minval=0., maxval=0.75)
mr_norm = mplcolors.Normalize(vmin=np.nanmin(mr_med[valid]), vmax=np.nanmax(mr_med[valid]))

for i in np.where(valid)[0]:
    color = cmap_mr(mr_norm(mr_med[i]))
    med, lo, hi = dz_med[i], dz_lo[i], dz_hi[i]
    errs = np.array([[med - lo], [hi - med]])
    ax_dz.errorbar(along_strike_xs[i], med, yerr=errs,
                   fmt="s", color=color, ecolor="black", elinewidth=1.,
                   capsize=3., ms=6.5, zorder=5)

# x-axis ticks and labels on top
ax_dz.xaxis.tick_top()
ax_dz.xaxis.set_label_position("top")
ax_dz.set_xlabel("Along-strike distance (km)")
ax_dz.set_ylabel("DZ half-width (m)")
ax_dz.grid(True, which="major", linestyle=":", color="gray", alpha=0.7)

sm_mr = plt.cm.ScalarMappable(cmap=cmap_mr, norm=mr_norm)
sm_mr.set_array([])
cbar_mr = fig.colorbar(sm_mr, cax=cax_dz)
cbar_mr.set_label("Median modulus ratio", rotation=270, labelpad=15)
cbar_mr.outline.set_visible(False)
cax_dz.minorticks_off()


####    BOTTOM ROW: ROTATED OPTICAL MAP    ####

# Reproject EW optical data to EPSG:4326
ew_latlon = optical.ew.rio.reproject("EPSG:4326")

# Compute lon/lat bounding box that fully covers the rotated view rectangle
theta = np.radians(rotation_angle)
cos_t, sin_t = np.cos(theta), np.sin(theta)
corners_rot = np.array([
    [xlim_rot[0], ylim_rot[0]],
    [xlim_rot[1], ylim_rot[0]],
    [xlim_rot[0], ylim_rot[1]],
    [xlim_rot[1], ylim_rot[1]],
])
# Inverse rotation: rotated offsets -> lon/lat offsets
corners_lon = corners_rot[:, 0] * cos_t + corners_rot[:, 1] * sin_t + lon_c
corners_lat = -corners_rot[:, 0] * sin_t + corners_rot[:, 1] * cos_t + lat_c
pad = 0.05
lon_lo, lon_hi = corners_lon.min() - pad, corners_lon.max() + pad
lat_lo, lat_hi = corners_lat.min() - pad, corners_lat.max() + pad

# Crop raster (y stored north-to-south, so slice hi→lo)
ew_crop = ew_latlon.sel(x=slice(lon_lo, lon_hi), y=slice(lat_hi, lat_lo))
LON, LAT = np.meshgrid(ew_crop.x.values, ew_crop.y.values)
data_2d = ew_crop.values

# Rotation transform: input is lon/lat; output lands in "rotated lon/lat" space
# which becomes the axes data coordinate system (governed by xlim/ylim below)
rot = (mtransforms.Affine2D()
       .translate(-lon_c, -lat_c)
       .rotate_deg(rotation_angle)
       .translate(lon_c, lat_c))
trans = rot + ax_opt.transData

# Axes limits in the rotated coordinate space
ax_opt.set_xlim(lon_c + xlim_rot[0], lon_c + xlim_rot[1])
ax_opt.set_ylim(lat_c + ylim_rot[0], lat_c + ylim_rot[1])

# No ticks, labels, or grid on the map but still a frame
ax_opt.set_xticks([])
ax_opt.set_yticks([])
ax_opt.set_xlabel("")
ax_opt.set_ylabel("")
ax_opt.grid(False)
ax_opt.set_facecolor((0.95, 0.95, 0.95))

# Plot optical displacement
im = ax_opt.pcolormesh(LON, LAT, data_2d,
                        transform=trans,
                        cmap=cmc.vik,
                        vmin=optical_clim[0], vmax=optical_clim[1],
                        shading="auto", rasterized=True)

# Profile swathes (translucent) and centerlines
for profile in profiles:
    # Swathe polygon in UTM, reprojected to lon/lat
    trace_buf = profile.trace.copy()
    trace_buf["geometry"] = trace_buf.geometry.buffer(profile_swathe_half_width_m)
    for geom in trace_buf.to_crs("EPSG:4326").geometry:
        coords = np.array(geom.exterior.coords)
        ax_opt.fill(coords[:, 0], coords[:, 1],
                    color="deeppink", alpha=0.25, linewidth=0,
                    transform=trans, zorder=4)
    # Centerline
    for geom in profile.trace.to_crs("EPSG:4326").geometry:
        coords = np.array(geom.coords)
        ax_opt.plot(coords[:, 0], coords[:, 1],
                    color="deeppink", linewidth=0.8,
                    transform=trans, zorder=5)


####    SCALE BAR — 10 km, lower left    ####
# A horizontal bar in the rotated space spans the rotated x-direction, which
# points at (cos_t, sin_t) in (lon, lat) space.  Convert 10 km to rotated degrees:
#   d[km] = L[rot-deg] * 111.32 * sqrt((cos_t * cos(lat_c))^2 + sin_t^2)
bar_km = 10.
km_per_rot_deg = 111.32 * np.sqrt((cos_t * np.cos(np.radians(lat_c)))**2 + sin_t**2)
bar_len = bar_km / km_per_rot_deg

x_range = xlim_rot[1] - xlim_rot[0]
y_range = ylim_rot[1] - ylim_rot[0]
bar_x0 = lon_c + xlim_rot[0] + 0.05 * x_range
bar_y0 = lat_c + ylim_rot[0] + 0.05 * y_range
tick_h = 0.015 * y_range

for xs, ys in [
    ([bar_x0, bar_x0 + bar_len], [bar_y0, bar_y0]),
    ([bar_x0, bar_x0],           [bar_y0, bar_y0 + tick_h]),
    ([bar_x0 + bar_len]*2,       [bar_y0, bar_y0 + tick_h]),
]:
    ax_opt.plot(xs, ys, color="black", linewidth=2.5, zorder=7, solid_capstyle="butt")
ax_opt.text(bar_x0 + bar_len / 2., bar_y0 + 0.022 * y_range,
            f"{bar_km:.0f} km", ha="center", va="bottom",
            fontsize=9., color="black", zorder=7)


####    NORTH ARROW — lower right    ####
# Geographic north in the rotated coordinate space is direction (-sin_t, cos_t)
arrow_x   = lon_c + xlim_rot[1] - 0.05 * x_range
arrow_y   = lat_c + ylim_rot[0] + 0.07 * y_range
arrow_len = 0.06 * y_range
north_dx  = -sin_t * arrow_len
north_dy  =  cos_t * arrow_len

ax_opt.annotate(
    "", xy=(arrow_x + north_dx, arrow_y + north_dy), xytext=(arrow_x, arrow_y),
    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=18),
    zorder=7
)
ax_opt.text(arrow_x + north_dx * 1.5, arrow_y + north_dy * 1.5,
            "N", ha="center", va="center",
            fontsize=11., color="black", fontweight="bold", zorder=7)


####    COLORBAR FOR OPTICAL DATA    ####
cbar_opt = fig.colorbar(im, cax=cax_opt)
cbar_opt.set_label("EW displacement (m)", rotation=270, labelpad=15)
cbar_opt.outline.set_visible(False)
cax_opt.minorticks_off()


####    COLORBAR PADDING    ####
# Compute layout first, then nudge colorbars inward from top/bottom/left.
# Must call set_constrained_layout(False) last to lock positions for saving.
fig.canvas.draw()
for cax in (cax_dz, cax_opt):
    pos = cax.get_position()
    cax.set_position([
        pos.x0 + cbar_pad_left,
        pos.y0 + pos.height * cbar_pad_tb,
        pos.width,
        pos.height * (1. - 2. * cbar_pad_tb),
    ])
fig.set_constrained_layout(False)


####    SAVE    ####
fig.savefig(save_path, dpi=450, bbox_inches="tight")
plt.show()
