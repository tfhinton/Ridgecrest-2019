from pathlib import Path

# resolved relative to the repo so the same file works on the Mac and the cluster
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

WORKING_DIR = RESULTS_DIR / "working"
TMP_DIR = WORKING_DIR / "tmp"          # scratch: pickles + preprocessing figures
TMP_DIR.mkdir(exist_ok=True, parents=True)

# AlTar Bayesian inversion runs (each run dir holds the step_*.h5 outputs).
ALTAR_DIR = RESULTS_DIR / "tri01"

FAULT_DIR = DATA_DIR / "fault"
INSAR_DIR = DATA_DIR / "insar"
GNSS_DIR = DATA_DIR / "gnss"
OPTICAL_DIR = DATA_DIR / "optical"

# Assembled joint system, pickled by prep_files_for_altar and reused downstream.
INVERSION_PICKLE = TMP_DIR / "inversion_manager.pickle"

EW_TIF = OPTICAL_DIR / "EW_Ridgecrest_1m_utm_detrended.tif"
NS_TIF = OPTICAL_DIR / "NS_Ridgecrest_1m_utm_detrended.tif"

INSAR_TRACKS = {
    # "A064": INSAR_DIR / "A064_20190704-0710",
    "D071": INSAR_DIR / "D071_20190704-0716",
}

FAULT_FILEPATHS = [
    FAULT_DIR / "SNFA-LLFZ-EAST-Eastern_Little_Lake_main_fault-CFM5.txt",
    FAULT_DIR / "SNFA-LLFZ-SOUT-Southern_Little_Lake_main_fault-CFM5.txt",
]
FAULT_PICKLE = FAULT_DIR / "ridgecrest_faults.pickle"

UTM_ZONE = 11
