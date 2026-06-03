#%%

####    IMPORTS    ####
import numpy as np
import matplotlib.pyplot as plt
import pickle
from codes import CSIWrapper, OpticalData, AltarOutput, TwoDDzForwardModel, PatchTwoD, UniformDist, HamiltonianInversion
from pathlib import Path


####    FILEPATHS    ####
csi_pickle = "/data/cycle/hintont/projects/ridgecrest/inversion_results/in15/inputs/csi.pickle"
altar_dir = "/data/cycle/hintont/projects/ridgecrest/inversion_results/in15/outputs/"
resdir = Path("/data/cycle/hintont/projects/ridgecrest/profile_inversion/in02/")
optical_ew_datapath = "/data/cycle/hintont/projects/ridgecrest/data/optical/EW_Ridgecrest_1m_utm_detrended.tif"
optical_ns_datapath = "/data/cycle/hintont/projects/ridgecrest/data/optical/NS_Ridgecrest_1m_utm_detrended.tif"
resdir.mkdir(parents=True, exist_ok=True)


####    LOAD CSI, ALTAR    ####
multi, faults, datasets, trans = pickle.load(open(csi_pickle, "rb"))
csi = CSIWrapper(multi, faults, datasets, trans)
altar = AltarOutput(altar_dir)


####    GENERATE PROFILES    ####
profiles, vertical_profiles = csi.gen_profiles(half_length=4., n_profiles=24)
s = slice(4, -4)
profiles = profiles[s]
vertical_profiles = vertical_profiles[s]


####    LOAD OPTICAL DATA    ####
print("Loading optical data...")
if not (resdir / "optical.pickle").exists():
    optical = OpticalData(ew_filepath=optical_ew_datapath, ns_filepath=optical_ns_datapath)
    optical = optical.clear_nan()
    optical = optical.decimate(10)
    with open(resdir / "optical.pickle", "wb") as f:
        pickle.dump(optical, f)
    fig, ax =optical.plot(ns=False, profiles=profiles, profile_swathe_width=1000.)
    fig.savefig(resdir / "optical.png", dpi=300)
else:
    optical = pickle.load(open(resdir / "optical.pickle", "rb"))

####    FOR EACH PROFILE    ####
i = 14
profile = profiles[i]

####    EVALUATE PROFILES    ####
print("Evaluating profile...")
profile = optical.evaluate_profile(profile, swathe_half_width=500.)
profile = profile.bin_average(n_bins=200, n_near_fault_bins=100, near_fault_dist=1000.)


####    ESTIMATE COVARIANCE    ####
print("Preparing inversion...")
cd_estimator = profile.displacements[1,:25]
x = np.arange(len(cd_estimator))
coeffs = np.polyfit(x, cd_estimator, 1)
trend = np.polyval(coeffs, x)
detrended = cd_estimator - trend

std = np.std(detrended)
data_covariance = std**2 * np.eye(profile.xs.size)


####    CENTRE DATA    ####
data = profile.displacements[1]
data -= np.mean(data)

c = np.argmax(data<0)
r = -data[c]/(data[c-1]-data[c])
x0 = profile.xs[c] - r*(profile.xs[c]-profile.xs[c-1])

profile.xs -= x0



####    BUILD FORWARD MODEL    ####
vd = [0., 500., 1000., 1500., 2750., 4000., 6000., 8000., 11000., 14000.]
model = TwoDDzForwardModel()
model.patches = [PatchTwoD(0., (vd[i]+vd[i+1])/2., vd[i+1]-vd[i]) for i in range(len(vd)-1)]
model.slips = np.zeros(len(model.patches))
model.xs = profile.xs

posteriors = []
for p in model.patches:
    z =+ p.z
    matching = filter(lambda x: x["intersection_top"]*1000 <= z and x["intersection_bot"]*1000 >= z, vertical_profiles[i])
    match = next(matching)
    id = match["patch_idx"]
    posterior = altar.final["ParameterSets"]["strikeslipmain"][:, id]

    posteriors.append(posterior)

model.slips[3:] = np.median(posteriors[3:], axis=1)


####    DEFINE PRIORS    ####
priors = [
    UniformDist("dz_halfwidth", 0., 1000.),
    UniformDist("modulus_ratio", 0.01, 0.9),
    UniformDist(f"slip0", -5., 0.),
    UniformDist(f"slip1", -10., 0.),
    UniformDist(f"slip2", -10., 0.)]

psets = [
    {"draws": 25, "tune": 10, "chains": 4},
    {"draws": 100, "tune": 40, "chains": 4},
    {"draws": 250, "tune": 100, "chains": 4},
    {"draws": 1000, "tune": 400, "chains": 4},
    {"draws": 100, "tune": 40, "chains": 10},
    {"draws": 25, "tune": 10, "chains": 40}
]

for j, pset in enumerate(psets):

    try:
        ####    RUN INVERSION    ####
        print("Running inversion...")
        inversion = HamiltonianInversion(model, priors, data, data_covariance)
        inversion = inversion.run(**pset)

        ####    SAVE RESULTS    ####
        with open(resdir / f"pset_{j}.pickle", "wb") as f:
            pickle.dump(inversion, f)
    except Exception as e:
        print(f"Pset {j} failed: {type(e).__name__}: {e}")
    
print("Done.")