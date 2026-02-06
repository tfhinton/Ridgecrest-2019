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
    fault_filenames = ["fault1.basemesh.triangles", "fault3.basemesh.triangles"]

    faults = []
    for i, filename in enumerate(fault_filenames):

        fault = triangleFault(f"fault{i}", utmzone=utm_zone, lon0=lon0, lat0=lat0)
        fault.readPatchesFromFile(
            os.path.join(fault_dir, filename),
            donotreadslip=True, readpatchindex=False)
        
        fault.setTrace(delta_depth=0.2)
        fault.discretize(every=0.2, fracstep=0.05)

        fault.xf = fault.xi
        fault.yf = fault.yi

        faults.append(fault)



    # ####    INSAR DATA    ####
    insar_name = "A064_20190704-0710"
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



    ####    COMPUTE COVARIANCE    ####
    covar = imcov(insar_name + "_covariance", sar, verbose=True)

    covar.selectedZones = []
    covar.maskOut([ 360-117.88,360-117.3,35.53,35.9])

    covar.computeCovariance(function='exp', frac=0.005, every=0.5, distmax=35.,tol=1e-10)

    if PLOT:
        covar.plot(data='all', plotData=True)
    # covar.write2file(savedir=datadir)

    sigma, lamda = covar.datasets[insar_name + "_covariance"]["Sigma"], covar.datasets[insar_name + "_covariance"]["Lambda"]
    print("SAR covariance", (sigma, lamda))



    ####    DISTANCE-BASED DOWNSAMPLING    ####
    startWindowSize, minimumWindowSize, chardist, expodist, tol, reject_distance = 10., 1.25, 1.5, 0.7, 0.005, 0.5
    # d_path = os.path.join(insar_subdir, "downsampled")

    downsampler = imdown(f"downsampling_{insar_name}", sar, faults)
    downsampler.initialstate(startWindowSize, minimumWindowSize, tolerance=tol, plot=False)
    downsampler.distanceBased(chardist=chardist, expodist=expodist, plot=PLOT)
    downsampler.reject_pixels_fault(reject_distance, faults)
    # downsampler.writeDownsampled2File(d_path, rsp=True)

    sar = downsampler.newimage
    sar.buildCd(sigma, lamda, function='exp')

    print(f"Original number of pixels: {len(sar.x)}")
    print(f"Downsampled number of pixels: {len(downsampler.newimage.x)}")
    

    ####    COMPUTE FAULT PATCH AREAS    ####
    areas = []
    for fault in faults:
        fault.computeArea()
        areas += fault.area
    with h5py.File(os.path.join(insar_subdir, "patch_areas.h5"), "w") as f:
        f.create_dataset("patch_areas", data=np.array(areas))



    ####   BUILD GREENS FUNCTIONS    ####
    for fault in faults:
        fault.initializeslip()
        fault.buildGFs(sar, slipdir="sd")
        fault.assembleGFs(sar, slipdir="sd", polys=None)
        fault.assembled(sar)
        fault.assembleCd(sar)
    


    ####    DEFINE RAMP    ####
    trans = transformation('Orbits and reference frame', utmzone="11", lon0=lon0, lat0=lat0)
    trans.buildGFs(sar, [3])
    trans.assembleGFs(sar)
    trans.assembled(sar)
    trans.assembleCd(sar)



    ####    ASSEMBLE RAMP AND FAULT GFs    ####
    multi = multiflt(insar_name, faults+[trans])
    multi.assembleGFs()

    print("Checking shapes of GFs, etc")
    print("GFs shape:", multi.G.shape)
    print("Data shape:", multi.d.shape)
    print("Cd shape:", multi.Cd.shape)



    ####    WRITE TO H5 FILES    ####
    multi.writeGFs2H5File(os.path.join(insar_subdir, f"greens_functions.h5"), name="gf")
    multi.writeData2H5File(os.path.join(insar_subdir, f"data.h5"), name="data")
    multi.writeCd2H5File(os.path.join(insar_subdir, f"covariance.h5"), name="covariance")



    


if __name__ == '__main__':    
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    main()