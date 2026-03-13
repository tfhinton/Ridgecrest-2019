# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:36:30 2016

@author: Théa Ragon

Creates altar run files: 
Gfs
data
Cd
Cp files : kernels, cov and ini model
"""

# Import Python Libraries
import numpy as np
import mpmath as mp
import os
import subprocess
import sys
from scipy.stats import linregress
import pdb
import copy

# Import CSI Libraries
import csi.RectangularPatches as rectFault
import csi.multifaultsolve as multiflt

# import my lib
import fault_geom as fg


def buildCdCor(co, sigma_e, lam_e, sigma_n, lam_n, function='exp',vertical=False):
        '''
        Builds the full Covariance matrix from values of sigma and lambda.

        If function='exp':

            :math:`C_d(i,j) = \sigma^2  e^{-\\frac{d[i,j]}{\lambda}}`

        elif function='gauss':

            :math:`C_d(i,j) = \sigma^2 e^{-\\frac{d_{i,j}^2}{2*\lambda}}`

        Args:
            * sigma             : Sigma term of the covariance
            * lam               : Caracteristic length of the covariance

        Kwargs:
            * function          : Can be 'gauss' or 'exp'

        Returns:
            * None
        '''

        # Assert
        assert function in ('exp', 'gauss'), \
                'Unknown functional form for Covariance matrix'

        # positions
        x = co.x
        y = co.y
        distance = np.sqrt( (x[:,None] - x[None,:])**2 + (y[:,None] - y[None,:])**2)

        # Compute Cd
        if function == 'exp':
            CdEast = sigma_e*sigma_e*np.exp(-1.0*distance/lam_e)
            CdNorth = sigma_n*sigma_n*np.exp(-1.0*distance/lam_n)
        elif function == 'gauss':
            CdEast = sigma_e*sigma_e*np.exp(-1.0*distance*distance/(2*lam_e))
            CdNorth = sigma_n*sigma_n*np.exp(-1.0*distance*distance/(2*lam_n))
        
        nd = np.shape(CdEast)[0]
        if vertical is False:
            Cd = np.vstack( (np.hstack((CdEast, np.zeros((nd,nd)))), np.hstack((np.zeros((nd,nd)), CdNorth))) )
        else:
            Cd = np.block( [[CdEast, np.zeros((nd,nd)), np.zeros((nd,nd))],
                        [np.zeros((nd,nd)), CdNorth, np.zeros((nd,nd))],
                        [np.zeros((nd,nd)), np.zeros((nd,nd)), np.zeros((nd,nd))]])
        
        return Cd
    
# ----------------------------------------------------------------------
def buildGFsallkernels(faults, gpsData, insarData, optiData, gfsdir, **edks_params):
    '''
    Build Green's Functions using EDKS for Triangle Tents
    :Args:
        * faults: list of fault CSI 
        * datasets
        * gfsdir: where to output gfs
    
    :Kwargs:
        * edks : If True, GFs calculated using a layered Earth model calculated with EDKS.
                 If False, GFs with Okada
                 
    Please specify in **edks_params: 
        * edksdir
        * modelname : xxx.edks = Filename of the EDKS kernels
        * sourceSpacing      : source spacing to calculate the Green's Functions
            OR sourceNumber   : Number of sources per patches.
            OR sourceArea     : Maximum Area of the sources.
            
    
            
    :Returns:

    '''
    try:
        owd = os.getcwd()
        os.chdir(edks_params['edksdir'])
#        pdb.set_trace()
        for fault in faults:
            fault.kernelsEDKS = edks_params['modelname']+'.edks'
            if 'Spacing' in str(edks_params.keys()):
                fault.sourceSpacing = edks_params['sourceSpacing']
            elif 'Number' in str(edks_params.keys()):
                fault.sourceNumber = edks_params['sourceNumber']
            else:
                fault.sourceArea = edks_params['sourceArea']
            fault.keepTrackOfSources = True
            
        # Get kernels to iterate over
#        files = os.listdir('./')
        listOfFiles = []
        for (dirpath, dirnames, filenames) in os.walk('./'):
            listOfFiles += [[dirpath, file] for file in filenames]
        kernels = []
        for dirpath, file in listOfFiles:
            if ('.edks' in file) and ('hdr.' not in file) and ('config' not in file) :
                kernels.append([dirpath,file])
        kernels.sort()  
        
        # Iterate
        for dirpath,kernel in kernels:
            owd2 = os.getcwd()
            os.chdir(dirpath)
            print('-----------------------------------')
            print('Dealing with Kernel: {}'.format(kernel))
        
            for fault in faults:
                # Copy the original fault
                flt = copy.copy(fault)
        
                # Change its name
                kname = kernel.replace('_',' ').replace('.edks','')
                flt.name = '{}_{}'.format(fault.name,kname)
        
                # Set up kernels
                flt.kernelsEDKS = kernel
        
                # Build the GFs
                for g in gpsData+insarData:
                    flt.buildGFs(g, slipdir='ds', method='edks', vertical=True)
        
                for g in optiData:
                    flt.buildGFs(g, slipdir='ds', method='edks', vertical=False)
        
                # Write GFs to file
                flt.saveGFs(outputDir=gfsdir)
                os.chdir(owd2)  
        
        os.chdir(owd)
    
    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)        
    return 
    # ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def buildGFs(faults, geoData, gfsdir, **edks_params):
    '''
    Build Green's Functions using EDKS for Triangle Tents
    :Args:
        * faults: list of fault CSI 
        * datasets
        * gfsdir: where to output gfs
    
    :Kwargs:
        * edks : If True, GFs calculated using a layered Earth model calculated with EDKS.
                 If False, GFs with Okada
                 
    Please specify in **edks_params: 
        * edksdir
        * modelname : xxx.edks = Filename of the EDKS kernels
        * sourceSpacing      : source spacing to calculate the Green's Functions
            OR sourceNumber   : Number of sources per patches.
            OR sourceArea     : Maximum Area of the sources.
            
    
            
    :Returns:

    '''
    try:
        owd = os.getcwd()
        os.chdir(edks_params['edksdir']+edks_params['modelname'])
#        pdb.set_trace()
        for fault in faults:
            fault.kernelsEDKS = edks_params['modelname']+'.edks'
            if 'Spacing' in str(edks_params.keys()):
                fault.sourceSpacing = edks_params['sourceSpacing']
            elif 'Number' in str(edks_params.keys()):
                fault.sourceNumber = edks_params['sourceNumber']
            else:
                fault.sourceArea = edks_params['sourceArea']
            fault.keepTrackOfSources = True
        
        kernel = edks_params['modelname']+'.edks'
        print('-----------------------------------')
        print('Dealing with Kernel: {}'.format(kernel))
    
        for fault in faults:
            # Copy the original fault
            flt = copy.copy(fault)
    
            # Change its name
            kname = kernel.replace('_',' ').replace('.edks','')
            flt.name = '{}_{}'.format(fault.name,kname)
    
            # Set up kernels
            flt.kernelsEDKS = kernel
    
            # Build the GFs
            for g in geoData:
                flt.buildGFs(g, slipdir='sd', method='edks', vertical=False if g.dtype == 'opticorr' else True)
    
            # Write GFs to file
            flt.saveGFs(outputDir=gfsdir)
        
        os.chdir(owd)
    
    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)        
    return 
    # ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def calcGFs(fault, datasets, edks=False, **edks_params):
    '''
    Calculate Green's Functions using Okada or EDKS 
    Used in class uncertainties
    
    :Args:
        * fault: fault CSI 
    
    :Kwargs:
        * edks : If True, GFs calculated using a layered Earth model calculated with EDKS.
                 If False, GFs with Okada
                 
    If edks is True, please specify in **edks_params: 
        ex: Cp_dip(fault,datasets,[40,50],multi_segments=2,edks=True,edksdir='PATH',modelname='CIA',sourceSpacing=0.5)
        * edksdir
        * modelname : xxx.edks = Filename of the EDKS kernels
        * sourceSpacing      : source spacing to calculate the Green's Functions
            OR sourceNumber   : Number of sources per patches.
            OR sourceArea     : Maximum Area of the sources.
            
    :Returns:
        * Gassembled
    '''
    if edks is False:
        fault.initializeslip()
        for data in datasets:
            fault.buildGFs(data, slipdir='sd')
        fault.assembleGFs(datasets, slipdir='sd', polys=None)
    else:
        owd = os.getcwd()
        os.chdir(edks_params['edksdir'])
        fault.initializeslip()
        for data in datasets:
            writeEDKSsubParams(fault, data, edks_params['modelname'])
        if 'Spacing' in str(edks_params.keys()):
            fault.sourceSpacing = edks_params['sourceSpacing']
        elif 'Number' in str(edks_params.keys()):
            fault.sourceNumber = edks_params['sourceNumber']
        else:
            fault.sourceArea = edks_params['sourceArea']
        fault.kernelsEDKS = edks_params['modelname']+'.edks'
        for data in datasets:
            fault.buildGFs(data, slipdir='sd', method= 'edks')
        fault.assembleGFs(datasets, slipdir='sd', polys=None)
        os.chdir(owd)   
    return fault.Gassembled   
    # ----------------------------------------------------------------------
        
def ini(dataSets,faultfi,faultpar,name,confdir,resdir,utmzone=None,cp=False,csi_inv=False,l_curve=False,**vals):
    '''
    If csi_inv is True, please specify the step, slip bounds, sigma, l_range or lambda, lambda0 in **vals:
    step, bounds for strike slip, bounds for dip slip, sigma, lambda, Lambda0
    Lambda0 is the distance between subfaults (approx)
    ini(dataSets,fdir='',faultfi,name,cp=False,csi_inv=True,step,[-20.,20.],[-400,0],sigma,lambda,lambda0)
    '''
    
    try:   
        lon0 = faultpar[5]
        lat0 =  faultpar[6]
        
        faultini = rectFault(name, lon0=lon0, lat0=lat0)
        faultini.readPatchesFromFile(faultfi, readpatchindex=False)
        faultini.setTrace(delta_depth=1.0)
#        fault.initializeslip()

        for data in dataSets:
            faultini.buildGFs(data, slipdir='sd')
        faultini.assembleGFs(dataSets, slipdir='sd', polys=None)
        
#        fault = gfok(name, lon0=lon0, lat0=lat0)
#        fault.readpatchesfromfile(faultfi, utmzone=utmzone, readpatchindex=False)
#        fault.setTrace(delta_depth=1.0)
#        GFs=[]
#        for data in dataSets:
#            GFs.append(fault.build_gf_okada(data, slipdir='sd', polys=None))
        
#        import seaborn as sns
#        import matplotlib.pyplot as plt
#        fig=plt.figure(1, figsize=(20,20))
#        cmap = sns.cubehelix_palette(16, rot=-.32,light=0.9,dark=0.3)
#        sns.heatmap(fault.Gassembled,cmap=cmap,cbar=True,xticklabels=False, yticklabels=False)
        
        if csi_inv is True:
            faultini.assembled(dataSets)
            faultini.assembleCd(dataSets)
            
#            p1 = np.matmul(np.transpose(fault.Gassembled),np.linalg.inv(fault.Cd))
#            Cm = np.linalg.inv(np.matmul(p1,fault.Gassembled))
#            np.savetxt(confdir+'amat.cm.optiwcp.txt', Cm)
            
            bounds = []
            for i in range(faultini.N_slip):
                bounds.append(vals['boundss']) #ss
            for i in range(faultini.N_slip):
                bounds.append(vals['boundds']) #ds           
            
            if l_curve is True:
                Sigma = vals['sigma']    #cm    
                l_range = vals['Lambda']   # km

                misfit=[]
                maxslip=[]
                for l in l_range: 
                    faultini.buildCm(Sigma, l, vals['Lambda0'])
                    slv = multiflt('fault', [faultini])
                    slv.assembleGFs()
                    slv.assembleCm()
                    slv.Cd = faultini.Cd
                    slv.ConstrainedLeastSquareSoln(bounds=bounds,iterations=1e10,method='SLSQP',
                                                        mprior=None,tolerance=1e-07,checkIter=True)
                    slv.distributem(verbose=True)
                    faults = slv.faults
                    RMS=[]
                    for data in dataSets:
                        data.buildsynth(faults, direction='sd')
                        RMS.append(data.getRMS()[1])
                    misfit.append(np.sum(RMS)) 
                    maxslip.append(np.amax(np.abs(slv.mpost)))
                    
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(figsize=(10,10))        
                plt.plot(maxslip,misfit)
                ax.scatter(maxslip,misfit)
                for i, txt in enumerate(l_range):
                    ax.annotate(txt, (maxslip[i],misfit[i]))
                fig.savefig(resdir+'lcurve.pdf',format='pdf', dpi=450)
            
            else:
                step = vals['step']
                
                Sigma = vals['sigma']   #cm
                Lambda = vals['Lambda'] #km
                faultini.buildCm(Sigma, Lambda, vals['Lambda0'])
                slv = multiflt('fault', [faultini])
                slv.assembleGFs()
                slv.assembleCm()
                slv.ConstrainedLeastSquareSoln(bounds=bounds,iterations=1e10,method='SLSQP',
                                                    mprior=None,tolerance=1e-07,checkIter=True)
                slv.distributem(verbose=True)
                faults = slv.faults
                RMS=[]
                fi = open(resdir+'step_'+step+'_rms.dat', 'wb')
                for data in dataSets:
                    data.buildsynth(faults, direction='sd')
                    RMS.append(data.getRMS())
                    print('RMS '+data.name,data.getRMS()[1])
                    fi.write("%s %.6f\n" % (data.name,data.getRMS()[1]))
                fi.close()
                    
                faultini.writePatches2File(resdir+'step_'+step+'_slip_dip.dat', add_slip='dipslip')
                faultini.writePatches2File(resdir+'step_'+step+'_slip_stk.dat', add_slip='strikeslip')
                faultini.writePatches2File(resdir+'step_'+step+'_slip.dat', add_slip='total')
                faultini.writeSlipDirection2File(resdir+'step_'+step+'_slipdir.dat')
    #            faultini.writeSlipDirection2File(savedir+'step_'+step+'_slipdir_tot_scaled.dat', scale='total', factor=0.1)
                fi = open(resdir+'step_'+step+'_slipdirrakescaled.dat', 'wb')
                for p in range(len(faultini.patch)):  
                    xc, yc, zc, width, length, strike, dip = faultini.getpatchgeometry(p, center=True)  
                    lonc, latc = faultini.xy2ll(xc, yc)
                    slip = faultini.getslip(faultini.patch[p]) 
                    rake = np.arctan2(slip[1],slip[0])
                    direc = rake*180/np.pi + strike*180/np.pi -180
                    leng = np.sqrt(slip[0]**2 + slip[1]**2)
                    fi.write("%.6f %.6f %.6f %.6f %.6f\n" % (lonc,latc,zc,direc,leng*0.002))
                fi.close()
                for data in dataSets:
                    if 'gps' in data.name:
                        data.write2file('step_'+step+'_'+data.name+'_data.dat', outDir=resdir, data='data')
                        data.write2file('step_'+step+'_'+data.name+'_synth.dat', outDir=resdir, data='synth')
                    else:
                        data.write2file(resdir+'step_'+step+'_'+data.name+'_data.dat', data='data')
                        data.write2file(resdir+'step_'+step+'_'+data.name+'_synth.dat', data='synth')

            
#            faultini.buildCm(Sigma, Lambda, 1)
#            slv = multiflt('faultini', [faultini])
#            slv.assembleGFs()
#            slv.assembleCm()
#            slv.ConstrainedLeastSquareSoln(bounds=bounds,iterations=1e10,method='SLSQP',
#                                                mprior=None,tolerance=1e-07,checkIter=True)
#            slv.distributem(verbose=True)
#            faults = slv.faults
#            for data in dataSets:
#                data.buildsynth(faults, direction='sd')
                
#            step='111'
#            faultini.writePatches2File(savedir+'step_'+step+'_slip_dip.dat', add_slip='dipslip')
#            faultini.writePatches2File(savedir+'step_'+step+'_slip_stk.dat', add_slip='strikeslip')
#            faultini.writePatches2File(savedir+'step_'+step+'_slip.dat', add_slip='total')
#            faultini.writeSlipDirection2File(savedir+'step_'+step+'_slipdir.dat')
##            faultini.writeSlipDirection2File(savedir+'step_'+step+'_slipdir_tot_scaled.dat', scale='total', factor=0.1)
#            fi = open(altardir+'step_'+step+'_slipdirrakescaled.dat', 'wb')
#            for p in range(len(faultini.patch)):  
#                xc, yc, zc, width, length, strike, dip = faultini.getpatchgeometry(p, center=True)  
#                lonc, latc = faultini.xy2ll(xc, yc)
#                slip = faultini.getslip(faultini.patch[p]) 
#                rake = np.arctan2(slip[1],slip[0])
#                direc = rake*180/np.pi + strike*180/np.pi -180
#                leng = np.sqrt(slip[0]**2 + slip[1]**2)
#                fi.write("%.6f %.6f %.6f %.6f %.6f\n" % (lonc,latc,zc,direc,leng*0.002))
#            fi.close()
#            for sar in insar:
#                sar.write2file(savedir+'step_'+step+'_{}_data.dat'.format(sar.name.replace(' ','_')), data='data')
#                sar.write2file(savedir+'step_'+step+'_{}_synth.dat'.format(sar.name.replace(' ','_')), 
#                        data='synth')
#            gps.write2file('step_'+step+'_gps_data.dat', outDir=savedir, data='data')
#            gps.write2file('step_'+step+'_gps_synth.dat', outDir=savedir, data='synth')
#    
#            #---------------------------------------------------------------
#            # Plot slip and synthetics
#            script1 = """
#            #!/bin/bash
#            cd /u/moana/user/ragon/fig/amatrice/res/
#            source /u/moana/user/ragon/fig/amatrice/gmt/func/gps_synth.sh
#            source /u/moana/user/ragon/fig/amatrice/gmt/func/plot_slip.sh
#            source /u/moana/user/ragon/fig/amatrice/gmt/func/insar_synth.sh
#    
#            plotslip {di} {st}
#            gpssynth {di} {st}
#            insarsynth6_points {di} {st} aa ad alos
#            insarsynth6_points {di} {st} sa sd sentinel
#            """.format(di=savedir, st=step)
#            subprocess.call(script1, shell=True)
            
    except Exception as err:
        sys.stderr.write('ERROR: %sn' % str(err))
    if cp is False:
        np.savetxt(confdir+name+'.gf.txt', faultini.Gassembled)
        faultini.assembled(dataSets)
        np.savetxt(confdir+name+'.d.txt', faultini.dassembled)
        faultini.assembleCd(dataSets)
        np.savetxt(confdir+name+'.cd.txt', faultini.Cd)
#        cdgps =  0.01*np.identity(np.shape(gps.Cd)[0])
#        cdsar = 0.01*np.identity(np.shape(faultini.Cd)[0]-np.shape(gps.Cd)[0])
#        zero= np.zeros((np.shape(gps.Cd)[0],np.shape(cdsar)[1]))
#        cd1 = np.concatenate((cdgps,zero), axis=1)
#        cd2 = np.concatenate((np.transpose(zero),cdsar), axis=1)
#        Cd = np.concatenate((cd1, cd2), axis=0)
#        np.savetxt(altardir+name+'.cd.txt', Cd)
        print('done! You can find files in ---->  {}'.format(confdir))
        return faultini
    elif cp is True:
        return faultini.Gassembled
    
def ini_edks(dataSets,edksdir,modname,faultfi,faultpar,name,confdir,source_spacing=0.5,cpfg=False,cpmu=False):
    '''
    Compute GFs with EDKS. 
    Input:
    - modname: xxx.edks = kernels file (generally model_name.edks): only model_names, without ".edks" !
    '''
    try:
        lon0 = faultpar[5]
        lat0 =  faultpar[6]
        
        owd = os.getcwd()
        fault = rectFault(name, lon0=lon0, lat0=lat0)
        fault.readPatchesFromFile(faultfi, readpatchindex=False)
        fault.setTrace(delta_depth=1.0)
        fault.trace2ll()
        fault.initializeslip()

        os.chdir(edksdir+modname)
        for data in dataSets:
            fe=writeEDKSsubParams(fault, data=data, edksfilename=modname)
            print(fe[0],fe[1],fe[2],fe[3])
            
        fault.kernelsEDKS = modname+'.edks'
        fault.sourceSpacing = source_spacing
    
        for data in dataSets:
            fault.buildGFs(data, slipdir='sd', method= 'edks')
    
        fault.assembleGFs(dataSets, slipdir='sd', polys=None)
#        import seaborn as sns
#        import matplotlib.pyplot as plt
#        fig=plt.figure(1, figsize=(20,20))
#        cmap = sns.cubehelix_palette(16, rot=-.32,light=0.9,dark=0.3)
#        sns.heatmap(fault.Gassembled,cmap=cmap,cbar=True,xticklabels=False, yticklabels=False)

    except Exception as err:
        sys.stderr.write('ERROR: %sn' % str(err))
    if cpmu is False and cpfg is False:
        np.savetxt(confdir+name+'.gf.txt', fault.Gassembled)
        fault.assembled(dataSets)
        np.savetxt(confdir+name+'.d.txt', fault.dassembled)
        fault.assembleCd(dataSets)
        np.savetxt(confdir+name+'.cd.txt', fault.Cd)
        print('done! You can find files in ---->  {}'.format(confdir))
        os.chdir(owd)
        return
    elif cpfg is True or cpmu is True:
        os.chdir(owd)
        return fault.Gassembled

def writeEDKSsubParams(fault, data, edksfilename, amax=None, plot=False, w_file=True):
    '''
    Write the subParam file needed for the interpolation of the green's function in EDKS.
    Francisco's program cuts the patches into small patches, interpolates the kernels to get the GFs at each point source, 
    then averages the GFs on the pacth. To decide the size of the minimum patch, it uses St Vernant's principle.
    If amax is specified, the minimum size is fixed.
    Args:
        * data          : Data object from gps or insar.
        * edksfilename  : Name of the file containing the kernels.
        * amax          : Specifies the minimum size of the divided patch. If None, uses St Vernant's principle.
        * plot          : Activates plotting.
        * w_file        : if False, will not write the subParam fil (default=True)
    Returns:
        * filename         : Name of the subParams file created (only if w_file==True)
        * RectanglePropFile: Name of the rectangles properties file
        * ReceiverFile     : Name of the receiver file
        * method_par       : Dictionary including useful EDKS parameters
    '''

    # print
    print ("---------------------------------")
    print ("---------------------------------")
    print ("Write the EDKS files for fault {} and data {}".format(fault.name, data.name))

    # Write the geometry to the EDKS file
    writeEDKSgeometry(fault)

    # Write the data to the EDKS file
    if data.dtype == 'gps':
        writeEDKSgps(data)
    elif data.dtype == 'insar':
        writeEDKSinsar(data)
    else:
        print("oops")

    # Create the variables
    if len(fault.name.split())>1:
        fltname = fault.name.split()[0]
        for s in fault.name.split()[1:]:
            fltname = fltname+'_'+s
    else:
        fltname = fault.name
    RectanglePropFile = 'edks_{}.END'.format(fltname)
    if len(data.name.split())>1:
        datname = data.name.split()[0]
        for s in data.name.split()[1:]:
            datname = datname+'_'+s
    else:
        datname = data.name
    ReceiverFile = 'edks_{}.idEN'.format(datname)

    if data.dtype == 'insar':
        useRecvDir = True # True for InSAR, uses LOS information
    else:
        useRecvDir = False # False for GPS, uses ENU displacements
    EDKSunits = 1000.0
    EDKSfilename = '{}'.format(edksfilename)
    prefix = 'edks_{}_{}'.format(fltname, datname)
    plotGeometry = '{}'.format(plot)

    # Build usefull outputs
    parNames = ['useRecvDir', 'Amax', 'EDKSunits', 'EDKSfilename', 'prefix']
    parValues = [ useRecvDir ,  amax ,  EDKSunits ,  EDKSfilename ,  prefix ]
    method_par = dict(zip(parNames, parValues))

    # Open the EDKSsubParams.py file        
    if w_file:
        filename = 'EDKSParams_{}_{}.py'.format(fltname, datname)
        fout = open(filename, 'w')

        # Write in it
        fout.write("# File with the rectangles properties\n")
        fout.write("RectanglesPropFile = '{}'\n".format(RectanglePropFile))
        fout.write("# File with id, E[km], N[km] coordinates of the receivers.\n")
        fout.write("ReceiverFile = '{}'\n".format(ReceiverFile))
        fout.write("# read receiver direction (# not yet implemented)\n")
        fout.write("useRecvDir = {} # True for InSAR, uses LOS information\n".format(useRecvDir))
        fout.write("# Maximum Area to subdivide triangles. If None, uses Saint-Venant's principle.\n")
        if amax is None:
            fout.write("Amax = None # None computes Amax automatically. \n")
        else:
            fout.write("Amax = {} # Minimum size for the patch division.\n".format(amax))
            
        fout.write("EDKSunits = 1000.0 # to convert from kilometers to meters\n")
        fout.write("EDKSfilename = '{}'\n".format(edksfilename))
        fout.write("prefix = '{}'\n".format(prefix))
        fout.write("plotGeometry = {} # set to False if you are running in a remote Workstation\n".format(plot))
        
        # Close the file
        fout.close()
        return filename, RectanglePropFile, ReceiverFile, method_par
    else:
        return RectanglePropFile, ReceiverFile, method_par

def writeEDKSgps(gps):
        '''
        This routine prepares the data file as input for EDKS.
        '''

        # Get the x and y positions
        x = gps.x
        y = gps.y

        # Open the file
        datname = gps.name.replace(' ','_')
        filename = 'edks_{}.idEN'.format(datname)
        fout = open(filename, 'w')

        # Write a header
        fout.write("id E N\n")

        # Loop over the data locations
        for i in range(len(x)):
            string = '{:5d} {} {} \n'.format(i, x[i], y[i])
            fout.write(string)

        # Close the file
        fout.close()
        return datname,filename
        
def writeEDKSinsar(insar):
        '''
        This routine prepares the data file as input for EDKS.
        '''

        # Get the x and y positions
        x = insar.x
        y = insar.y

        # Get LOS informations
        los = insar.los

        # Open the file
        datname = insar.name.replace(' ','_')
        filename = 'edks_{}.idEN'.format(datname)
        fout = open(filename, 'w')

        # Write a header
        fout.write("id E N E_los N_los U_los\n")

        # Loop over the data locations
        for i in range(len(x)):
            string = '{:5d} {} {} {} {} {} \n'.format(i, x[i], y[i], los[i,0], los[i,1], los[i,2])
            fout.write(string)

        # Close the file
        fout.close()
        return datname,filename
        
def writeEDKSgeometry(fault, ref=None):
    '''
    This routine spits out 2 files:
    filename.lonlatdepth: Lon center | Lat Center | Depth Center (km) | Strike | Dip | Length (km) | Width (km) | patch ID
    filename.END: Easting (km) | Northing (km) | Depth Center (km) | Strike | Dip | Length (km) | Width (km) | patch ID

    These files are to be used with /home/geomod/dev/edks/MPI_EDKS/calcGreenFunctions_EDKS_subRectangles.py

    Args:
        * ref           : Lon and Lat of the reference point. If None, the patches positions is in the UTM coordinates.
    '''

    # Filename
    fltname = fault.name.replace(' ','_')
    filename = 'edks_{}'.format(fltname)

    # Open the output file
    flld = open(filename+'.lonlatdepth','w')
    flld.write('#lon lat Dep[km] strike dip length(km) width(km) ID\n')
    fend = open(filename+'.END','w')
    fend.write('#Easting[km] Northing[km] Dep[km] strike dip length(km) width(km) ID\n')

    # Reference
    if ref is not None:
        refx, refy = fault.putm(ref[0], ref[1])
        refx /= 1000.
        refy /= 1000.

    # Loop over the patches
    for p in range(len(fault.patch)):
        x, y, z, width, length, strike, dip = fault.getpatchgeometry(p, center=True)
        strike = strike*180./np.pi
        dip = dip*180./np.pi
        lon, lat = fault.xy2ll(x,y)
        if ref is not None:
            x -= refx
            y -= refy
        flld.write('{} {} {} {} {} {} {} {:5d} \n'.format(lon,lat,z,strike,dip,length,width,p))
        fend.write('{} {} {} {} {} {} {} {:5d} \n'.format(x,y,z,strike,dip,length,width,p))

    # Close the files
    flld.close()
    fend.close()
    return


def cp_mu(edksdir,modelname,m0,name,confdir,faultpar):
    try:
        length = faultpar[0]
        width = faultpar[1]   
        
        fi = open(edksdir+modelname+'/'+modelname+'.model','r')
        ri = fi.readlines()
        nbr_layers = int(ri[0].split(' ')[0])
        fi.close()
        gfs_ini=ini_edks(edksdir, modelname,cpmu=True)
            
        gfs=[]
        K = []
        mu=[]
        for l in range(1,nbr_layers+1):
            print ('----- Layer '+str(l))
            ro = float(ri[l].split('    ')[2])
            vs = float(ri[l].split('    ')[0])
            vp = float(ri[l].split('    ')[1])
            h  = float(ri[l].split('    ')[3])
            lamb = ro*(vp**2-2*vs**2)
            mu.append(ro*vs**2)
            mu2 = mu[l-1]+10
            vs2 = np.sqrt(mu2/ro)
            vp2=np.sqrt((lamb+2*mu2)/ro)
            
            script1 = """
            cd {}
            mkdir {}
            """.format(edksdir,modelname+'_l'+str(l))
            subprocess.call(script1, shell=True)
            
            # Create new velocity model
            dire = edksdir+modelname+"_l"+str(l)+"/"
            fil = open(dire+modelname+'_l'+str(l)+'.model','w')
            for li in range(len(ri)):
                if li != l:
                    fil.write("%s" %(ri[li]))
                else:
                    fil.write(" %f    %f    %f    %f \n" %(vs2,vp2,ro,h))
            fil.close()
            print(dire+modelname+'_l'+str(l)+'.model OK')
            
            # Calculate EDKS kernels
            if not os.path.isfile(os.path.join(edksdir,'PrepGFs_'+modelname+'.py')):
                print('PrepGFs python script not found!')
                print('---> please write it for model {}'.format(modelname))
                sys.exit(1)
            if not os.path.isfile(os.path.join(dire,'hdr.'+modelname+'_l'+str(l)+'.edks')):
                os.chdir(edksdir)
                os.system('python PrepGFs_'+modelname+'.py '+modelname+'_l'+str(l))
                print('python PrepGFs_'+modelname+'.py '+modelname+'_l'+str(l))
            else: 
                print(modelname+'_l'+str(l)+' edks kernels already computed')
            
            # Calculate GFs
            gfs.append(ini_edks(edksdir,modelname+'_l'+str(l),cpmu=True))
                 
            # Calculate Kernel for the l layer
            K.append( (gfs[l-1]-gfs_ini) / (np.log(mu[l-1]) - np.log(mu2)) )
            print(modelname+'_l'+str(l)+' kernel OK')
            np.savetxt(confdir+name+'.kernel.mul'+str(l)+'.txt', K[l-1])
        
        Cmu = np.zeros((nbr_layers,nbr_layers)) # uncertainty on log(mu)!!
        for l in range(nbr_layers):
            if l <= 3:
                Cmu[l,l] = 2*(0.22)**2  # log(mu+20%mu) - log(mu)
            elif l > 3:
                Cmu[l,l] = 2*(0.04)**2 # log(mu+4%mu) - log(mu)
        np.savetxt(confdir+name+'.cmu.txt', Cmu)
        
        Mo = m0  # from GCMT catalog, in dyne.m
        meanslip = Mo*10**(-7)/(3*10**10*length*width)
        imod = np.empty(K[0].shape[1])
        imod[:] = meanslip*100  # slip in cm!
        np.savetxt(confdir+name+'.inimodel.txt', imod)
    
        print('Done!  You can find files here   ---->   {}'.format(confdir))
    except Exception as err:
        sys.stderr.write('ERROR: %sn' % str(err))
    return    
    
    
    
def cp_dip(faultpar,diprg,name,odir,confdir,edks=False,opti=False,**vals):
    '''
    inputs
    faultpar: vector of fault parameters
            ex: faultparams = [length,width,nstrike,ndip,strike,lon,lat]
    diprange = vector of tested dip limits
            ex: diprg = [40,50]
            
    If edks is True, please specify in **vals: edksdir, modelname
    '''
    try:
        length = faultpar[0]
        width = faultpar[1]
        nstrike = faultpar[2]
        ndip = faultpar[3]
        strike = faultpar[4]
        lon = faultpar[5]
        lat =  faultpar[6]
        edksdir=vals['edksdir']
        modelname=vals['modelname']
        
        #---------------------------- DIP ------------------------------
        
        # Définir le range de dip sur lequel calculer les kernels
        dip_range = range(diprg[0],diprg[1])#(40,51)
        
        # Calc Green's functions
        gfs = []
        for d in dip_range: 
            # Dessiner la faille avec un dip donné
            if opti is True:
                fg.patchesopti(lon, lat, 0, length, strike ,d, width, nstrike, ndip,odir+name+'_dip{}.rectangles'.format(str(d)))
            elif opti is False:
                fg.patches(lon, lat, 0, length, strike ,d, width, nstrike, ndip,odir+name+'_dip{}.rectangles'.format(str(d)))
            # Calculer la GF associée
            if edks is True:
                gfs.append(ini_edks(edksdir,modelname,fdir=odir,faultfi=name+'_dip{}.rectangles'.format(str(d)),cpfg=True,cpmu=False))
            else:
                gfs.append(ini(fdir=odir,faultfi=name+'_dip{}.rectangles'.format(str(d)),cp=True))
        
        #-------------------------------------------------
        
        pente = np.empty(gfs[0].shape) # kernel
        rvalue = np.empty(gfs[0].shape)
        pvalue = np.empty(gfs[0].shape)
        stderr = np.empty(gfs[0].shape)
        inter = np.empty(gfs[0].shape) # pente
        r = np.empty((len(dip_range),gfs[0].shape[0], gfs[0].shape[1])) # residuals
        
        for i in range(gfs[0].shape[0]):
            for j in range(gfs[0].shape[1]):
                # faire la régression linéaire pour chaque couple paramètre/donnée, sur le range de dip
                pente[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(dip_range,[gfs[k][i,j] for k in range(len(dip_range))])
        
        Kdip= pente
        np.savetxt(confdir+name+'.kernel.fgdip.txt', Kdip)
        
        script1 = """
        cd {}
        rm -f {}_dip*.rectangles*
        """.format(odir,name)
        subprocess.call(script1, shell=True)
        
        ##---------------------------- STRIKE ------------------------------
        
        Kstk = np.zeros(Kdip.shape)
        
    #    stk_range = range(159,169)
    #    dip = 45
    #    gfs = []
    #    for stk in stk_range: 
    #        fg.patchesopti(lon, lat, 0, length, stk ,dip, width, nstrike, ndip,odir+name+'_stk{}.rectangles'.format(str(stk)))
    #        gfs.append(ini(odir,name+'_stk{}.rectangles'.format(str(stk)),cp=True))
    #    pente = np.empty(gfs[0].shape) # kernel
    #    rvalue = np.empty(gfs[0].shape)
    #    pvalue = np.empty(gfs[0].shape)
    #    stderr = np.empty(gfs[0].shape)
    #    inter = np.empty(gfs[0].shape) # pente
    #    r = np.empty((len(dip_range),gfs[0].shape[0], gfs[0].shape[1])) # residuals
    #    for i in range(pente.shape[0]):
    #        for j in range(pente.shape[1]):
    #            # faire la régression linéaire pour chaque couple paramètre/donnée, sur le range de dip
    #            pente[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(stk_range,[gfs[k][i,j] for k in range(len(stk_range))])
    #    Kstk= pente* stk_range[0]
        
        np.savetxt(confdir+name+'.kernel.fgstk.txt', Kstk)
        
        script1 = """
        cd {}
        rm -f {}_stk*.rectangles*
        """.format(odir,name)
        subprocess.call(script1, shell=True)
        
        ##---------------------------- COV ------------------------------
        sigma_dip = 7.
        sigma_stk = 0.    
        Cfg = np.zeros((2,2))
        Cfg[0,0] = sigma_dip**2
        Cfg[1,1] = sigma_stk**2
        np.savetxt(confdir+name+'.covdip.txt', Cfg)
    
        ##----------------------- INITIAL MODEL -------------------------
        Mo = m0   # from GCMT catalog, in dyne.m
        meanslip = -Mo*10**(-7)/(3*10**10*length*width)
#        meanslip = 25
        imod = np.empty(Kdip.shape[1])
        imod[0:Kdip.shape[1]/2] = 0  # slip in cm!
        imod[Kdip.shape[1]/2:Kdip.shape[1]] = meanslip*100
        np.savetxt(confdir+name+'.inimodel.txt', imod)
        
        print('Done!  You can find files here   ---->   {}'.format(confdir))    
    except Exception as err:
        sys.stderr.write('ERROR: %sn' % str(err))
    return


def cp_pos(faultpar,distrg,name,odir,confdir,edks=False,opti=False,**vals):
    '''
    inputs
    faultpar: vector of fault parameters
            ex: faultparams = [length,width,nstrike,ndip,strike,lon,lat]
    diprange = vector of tested distance limits in km (better if centered on zero
                OR make sure your strike is given toward the South
            ex: distrg = [-3,+3]
            
    If edks is True, please specify in **vals: edksdir, modelname
    '''

    def fh(lon, lat, z, strike, distance):
        """
        Given a start point (lon lat), bearing (degrees), and distance (m),
        calculates the destination point (lon lat)
        """
        theta = strike
        delta = distance / 6371000.
        def to_radians(theta):
            return theta * np.pi / 180.
        def to_degrees(theta):
            return theta * 180.0 / np.pi
        theta = to_radians(theta)
        lat1 = to_radians(lat)
        lon1 = to_radians(lon)
        lat2 = np.arcsin( np.sin(lat1) * np.cos(delta) + \
        np.cos(lat1) * np.sin(delta) * np.cos(theta) )
        lon2 = lon1 + np.arctan2( np.sin(theta) * np.sin(delta) * np.cos(lat1), \
        np.cos(delta) - np.sin(lat1) * np.sin(lat2))
        return (to_degrees(lon2), to_degrees(lat2), z)

    try:
        length = faultpar[0]
        width = faultpar[1]
        nstrike = faultpar[2]
        ndip = faultpar[3]
        strike = faultpar[4]
        lon = faultpar[5]
        lat =  faultpar[6]
        dip = faultpar[7]
        edksdir=vals['edksdir']
        modelname=vals['modelname']
        
        #---------------------------- DIST ------------------------------
        
        # Définir le range de position sur lequel calculer les kernels
        dist_range = range(distrg[0],distrg[1])#(40,51)
        
        # Calc Green's functions
        gfs = []
        for d in dist_range: 
            if d >= 0:
                lon2, lat2, z = fh(lon, lat, 0, strike-90, d*1000.)
            else:
                lon2, lat2, z = fh(lon, lat, 0, strike+90, abs(d)*1000.)
            # Dessiner la faille avec une position donnée
            if opti is True:
                fg.patchesopti(lon2, lat2, 0, length, strike ,dip, width, nstrike, ndip,odir+name+'_pos{}.rectangles'.format(str(d)))
            elif opti is False:
                fg.patches(lon2, lat2, 0, length, strike ,dip, width, nstrike, ndip,odir+name+'_pos{}.rectangles'.format(str(d)))

            # Calculer la GF associée
            if edks is True:
                gfs.append(ini_edks(edksdir,modelname,fdir=odir,faultfi=name+'_pos{}.rectangles'.format(str(d)),cpfg=True,cpmu=False))
            else:
                gfs.append(ini(fdir=odir,faultfi=name+'_pos{}.rectangles'.format(str(d)),cp=True))
        #-------------------------------------------------
        pente = np.empty(gfs[0].shape) # kernel
        rvalue = np.empty(gfs[0].shape)
        pvalue = np.empty(gfs[0].shape)
        stderr = np.empty(gfs[0].shape)
        inter = np.empty(gfs[0].shape) # pente
        #r = np.empty((len(dip_range),gfs[0].shape[0], gfs[0].shape[1])) # residuals
        
        for i in range(gfs[0].shape[0]):
            for j in range(gfs[0].shape[1]):
                # faire la régression linéaire pour chaque couple paramètre/donnée, sur le range de distance
                pente[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(dist_range,[gfs[k][i,j] for k in range(len(dist_range))])
        
        Kpos= pente
        np.savetxt(confdir+name+'.kernel.fgpos.txt', Kpos)
        
        script1 = """
        cd {}
        rm -f {}_pos*.rectangles*
        """.format(odir,name)
        subprocess.call(script1, shell=True)
        
        ##---------------------------- COV ------------------------------
        sigma_dip = 2.
        sigma_stk = 0.  
        Cfg = np.zeros((2,2))
        Cfg[0,0] = sigma_dip**2
        Cfg[1,1] = sigma_stk**2
        np.savetxt(confdir+name+'.covpos.txt', Cfg)
    
        ##----------------------- INITIAL MODEL -------------------------
        Mo = m0   # from GCMT catalog, in dyne.m
        meanslip = -Mo*10**(-7)/(3*10**10*length*width)
#        meanslip = 25
        imod = np.empty(Kpos.shape[1])
        imod[0:Kpos.shape[1]/2] = 0  # slip in cm!
        imod[Kpos.shape[1]/2:Kpos.shape[1]] = meanslip*100
        np.savetxt(confdir+name+'.inimodel.txt', imod)
        
        print('Done!  You can find files here   ---->   {}'.format(confdir))    
    except Exception as err:
        sys.stderr.write('ERROR: %sn' % str(err))
    return

def cp_rough(faultpar,rough_rg,nbr_rough,name,odir,confdir,edks=False,opti=False,**vals):
    '''
    inputs
    faultpar: vector of fault parameters
            ex: faultparams = [length,width,nstrike,ndip,strike,lon,lat]
    rough_range = vector of rougness mean and std deviation for random roughness (in degrees!)
            ex: rough_rg = [0,0.5]
    nbr rough: nmber of random rough faults to compute to calculate the kernel
    
    If edks is True, please specify in **vals: edksdir, modelname
    '''
    try:
        length = faultpar[0]
        width = faultpar[1]
        nstrike = faultpar[2]
        ndip = faultpar[3]
        strike = faultpar[4]
        lon = faultpar[5]
        lat =  faultpar[6]
        dip = faultpar[7]
        edksdir=vals['edksdir']
        modelname=vals['modelname']
        
        gfs = []
        fg.patches(lon, lat, 0, length, strike ,dip, width, nstrike, ndip,odir+name+'_dip{}.rectangles'.format(str(dip)))
        if edks is True:
                gfs.append(ini_edks(edksdir,modelname,fdir=odir,faultfi=name+'_dip{}.rectangles'.format(str(dip)),cpfg=True,cpmu=False))
        else:
            gfs.append(ini(fdir=odir,faultfi=name+'_dip{}.rectangles'.format(str(dip)),cp=True))
        for i in range(1,nbr_rough): 
            if opti is False:
                fg.patches_rough(lon, lat, 0, length, strike ,dip, width, nstrike, ndip,rough_rg,odir+name+'_rough{}.rectangles'.format(str(i)))
            if edks is True:
                gfs.append(ini_edks(edksdir,modelname,fdir=odir,faultfi=name+'_rough{}.rectangles'.format(str(i)),cpfg=True,cpmu=False))
            else:
                gfs.append(ini(fdir=odir,faultfi=name+'_rough{}.rectangles'.format(str(i)),cp=True))
        
        #-------------------------------------------------
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_style("white")
        sns.despine(left=True)
        fig = plt.figure()
        plt.plot([gfs[k][1500,150] for k in range(nbr_rough)])
        fig = plt.figure()
        plt.plot([gfs[k][2500,275] for k in range(nbr_rough)])
        fig = plt.figure()
        plt.plot([gfs[k][10,400] for k in range(nbr_rough)])
        
        mean = np.mean(gfs[1:nbr_rough-1],axis=0)
        absgfs = np.abs(gfs)
        signgfs = np.sign(gfs[0])
        stdgfs = np.std(gfs,axis=0)
        rmax = signgfs*np.max(absgfs[1:nbr_rough-1],axis=0)
        np.savetxt(confdir+name+'.cov.fgrough.txt', stdgfs)
        
#        pente = np.empty(gfs[0].shape)
#        for i in range(gfs[0].shape[0]):
#            for j in range(gfs[0].shape[1]):
#                pente[i,j]=(gfs[0][i,j]-mean[i,j])/15
#        Kr = pente
#        np.savetxt(altardir+name+'.kernel.fgrough.txt', Kr)
        
        print('Done!  You can find files here   ---->   {}'.format(confdir))    
    except Exception as err:
        sys.stderr.write('ERROR: %sn' % str(err))
    return
    
    
def calcCp(length,width,m0,confdir,name):
    try:
        Kdip = np.loadtxt(confdir+'amat.kernel.fgdip.txt')
        Kpos = np.loadtxt(confdir+'amat.kernel.fgpos.txt')
        Kstk = np.loadtxt(confdir+'amat.kernel.fgstk.txt')
        Covdip = np.loadtxt(confdir+'amat.covdip.txt')
        Covpos = np.loadtxt(confdir+'amat.covpos.txt')
#        cdrough = np.loadtxt(confdir+'amat.cov.fgrough.txt')
#        
#        Covrough = np.dot(cdrough,np.transpose(cdrough))        
        
        Kmu = []
        for i in range(1,9):
            Kmu.append(np.loadtxt(confdir+'amat.kernel.mul'+str(i)+'.txt'))
        Kmu = np.array(Kmu)
        Cmu = np.loadtxt(confdir+'amat.cmu.txt')
        
        Mo = m0   # from GCMT catalog, in dyne.m
        meanslip = -Mo*10**(-7)/(3*10**10*length*width)
        imod = np.empty(Kpos.shape[1])
        imod[0:Kpos.shape[1]/2] = 0  # slip in cm!
        imod[Kpos.shape[1]/2:Kpos.shape[1]] = meanslip*100
        slipmean=imod
        #-------- Cp FG
        K = []
        K.append( Kdip )
        K.append( Kstk )
        kernels = np.asarray(K)
        k = np.transpose(np.matmul(kernels, slipmean))
        C1 = np.matmul(k, Covdip)
        Cpdip = np.matmul(C1, np.transpose(k))
        np.savetxt(confdir+name+'.cpdip.txt', Cpdip)
        
        #-------- Cp MU
        K = Kmu
        kernels = np.asarray(K)
        k = np.transpose(np.matmul(kernels, slipmean))
        C1 = np.matmul(k, Cmu)
        Cpmu = np.matmul(C1, np.transpose(k))
        np.savetxt(confdir+name+'.cpmu.txt', Cpmu)
        
        #-------- Cp FG
        Kp = []
        Kp.append( Kpos )
        Kp.append( Kstk )
        kernelsp = np.asarray(Kp)
        kp = np.transpose(np.matmul(kernelsp, slipmean))
        C1 = np.matmul(kp, Covpos)
        Cppos = np.matmul(C1, np.transpose(kp))
        np.savetxt(confdir+name+'.cppos.txt', Cppos)
        
        Cd=np.loadtxt(confdir+name+'.cd.txt')
        
        cdmu = Cd + Cpmu 
        cddip = Cd + Cpdip
        cdpos = Cd + Cppos
        cddippos = Cd + Cpdip + Cppos
#        cddipposrough = Cd + Cpdip + Cppos + Covrough
#        cdrough = Cd + Covrough
        cdmudip = Cd + Cpmu + Cpdip
        np.savetxt(confdir+name+'.cddip.txt', cddip)
        np.savetxt(confdir+name+'.cdpos.txt', cdpos)
        np.savetxt(confdir+name+'.cdmu.txt', cdmu)
#        np.savetxt(confdir+name+'.cdrough.txt', cdrough)
        np.savetxt(confdir+name+'.cddippos.txt', cddippos)
#        np.savetxt(confdir+name+'.cddipposrough.txt', cddipposrough)
        np.savetxt(confdir+name+'.cdmudip.txt', cdmudip)
        cdmudippos = Cd + Cpmu + Cppos + Cpdip
        cdmu10dippos = Cd + Cpmu + Cppos + 10*Cpdip
        cdmu10dipposx2 = 2*( Cd + Cpmu + Cppos + 10*Cpdip )
#        cdmudipposrough = Cd + Cpmu + Cppos + Cpdip + Covrough
        np.savetxt(confdir+name+'.cdmudippos.txt', cdmudippos)
        np.savetxt(confdir+name+'.cdmu10dippos.txt', cdmu10dippos)
        np.savetxt(confdir+name+'.cdmu10dipposx2.txt', cdmu10dipposx2)
#        np.savetxt(confdir+name+'.cdmudipposrough.txt', cdmudipposrough)
    except Exception as err:
        sys.stderr.write('ERROR: %sn' % str(err))
    return
    
if __name__ == "__main__":
    
    print('Please define the files you want to use in a distinct python file')
      
