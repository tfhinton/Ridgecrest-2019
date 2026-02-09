import sys
sys.path.append("/Users/hintont/Dev/packages")
import os
import csi.insar as ir
import csi.TriangularPatches as triangleFault
import csi.imagedownsampling as imdown
import csi.imagecovariance as imcov
import csi.multifaultsolve as multiflt
import csi.transformation as transformation
import numpy as np
import h5py
import matplotlib.pyplot as plt



####    CONFIG    ####
main_dir = "/Users/hintont/Dev/projects/Ridgecrest"
insar_dir = os.path.join(main_dir, "data/insar")

utm_zone = "11"
lon0 = 360 - 117.5
lat0 = 35.7



PLOT = False
def main():

    ####    FAULTS    ####
    fault_dir = os.path.join(main_dir, "data/fault")
    fault_filenames = ["fault1.mesh.triangles", "fault3.mesh.triangles"]

    faults = []
    for i, filename in enumerate(fault_filenames):

        fault = triangleFault(f"fault{i}", utmzone=utm_zone, lon0=lon0, lat0=lat0)
        fault.readGocadPatches( os.path.join(fault_dir, filename) )
        
        fault.setTrace(delta_depth=0.2)

        faults.append(fault)



    ####    COMPUTE FAULT PATCH AREAS    ####
    areas = []
    for fault in faults:
        fault.computeArea()
        areas += fault.area
    areas = np.array(areas)



    ####    INSAR DATA    ####
    insar_names = ["A064_20190704-0710", "D071_20190704-0716"]
    datasets = []

    for insar_name in insar_names:

        ####    LOAD INSAR DATA    ####
        insar_subdir = os.path.join(insar_dir, insar_name)

        MINLON, MAXLON, MINLAT, MAXLAT = 360-118.1, 360-117.0 ,35.3, 36.2

        unw = os.path.join(insar_subdir, "unwrapped.grd")
        e = os.path.join(insar_subdir, "east.grd")
        n = os.path.join(insar_subdir, "north.grd")
        u = os.path.join(insar_subdir, "up.grd")

        sar = ir(insar_name, utmzone=utm_zone, lon0=lon0, lat0=lat0)
        sar.read_from_grd( unw, los=[ e, n, u ], factor=1.)

        sar.select_pixels(MINLON, MAXLON, MINLAT, MAXLAT)
        sar.checkNaNs()

        if PLOT:
            sar.plot()
            plt.show(block=True)



        ####    COMPUTE COVARIANCE    ####
        covar = imcov(insar_name, sar, verbose=True)

        covar.selectedZones = []
        covar.maskOut([ 360-117.88,360-117.3,35.53,35.9])

        covar.computeCovariance(function='exp', frac=0.005, every=0.5, distmax=35.,tol=1e-10)

        if PLOT:
            covar.plot(data='all', plotData=True)
            plt.show(block=True)

        sigma, lamda = covar.datasets[insar_name]["Sigma"], covar.datasets[insar_name]["Lambda"]
        print("SAR covariance", (sigma, lamda))



        ####    DISTANCE-BASED DOWNSAMPLING    ####
        startWindowSize, minimumWindowSize, chardist, expodist, tol, reject_distance = 10., 1.25, 1.5, 0.7, 0.005, 0.5

        downsampler = imdown(insar_name, sar, faults)
        downsampler.initialstate(startWindowSize, minimumWindowSize, tolerance=tol, plot=False)
        downsampler.distanceBased(chardist=chardist, expodist=expodist, plot=PLOT)
        downsampler.reject_pixels_fault(reject_distance, faults)

        print(f"Original number of pixels: {len(sar.x)}")
        print(f"Downsampled number of pixels: {len(downsampler.newimage.x)}")

        sar = downsampler.newimage

        if PLOT:
            sar.plot(title="Final SAR", plotType="flat")
            sar.plot(title="Final SAR", plotType="scatter")
            plt.show(block=True)

        sar.buildCd(sigma, lamda, function='exp')
        datasets.append(sar)



    ####   BUILD GREENS FUNCTIONS    ####
    for fault in faults:
        fault.initializeslip()
        for dataset in datasets:
            fault.buildGFs(dataset, slipdir="sd")
        fault.assembleGFs(datasets, slipdir="sd", polys=None)
        fault.assembled(datasets)
        fault.assembleCd(datasets)
    


    ####    DEFINE RAMP    ####
    trans = transformation('Orbits and reference frame', utmzone="11", lon0=lon0, lat0=lat0)
    trans.buildGFs(datasets, [3]*len(datasets))
    trans.assembleGFs(datasets)
    trans.assembled(datasets)
    trans.assembleCd(datasets)



    ####    ASSEMBLE RAMP AND FAULT GFs    ####
    multi = multiflt("datasets", faults+[trans])
    multi.assembleGFs()

    print("Checking shapes of GFs, etc")
    print("GFs shape:", multi.G.shape)
    print("Data shape:", multi.d.shape)
    print("Cd shape:", multi.Cd.shape)



    ####    WRITE TO H5 FILES    ####
    inputs_dir = os.path.join(main_dir, "results/01/inputs")

    multi.writeGFs2H5File(os.path.join(inputs_dir, "greens_functions.h5"), name="gf")
    multi.writeData2H5File(os.path.join(inputs_dir, "data.h5"), name="data")
    multi.writeCd2H5File(os.path.join(inputs_dir, "covariance.h5"), name="covariance")
    with h5py.File(os.path.join(inputs_dir, "patch_areas.h5"), "w") as f:
        f.create_dataset("patch_areas", data=areas)

    return multi, faults, datasets, trans



    


if __name__ == '__main__':    
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    main()