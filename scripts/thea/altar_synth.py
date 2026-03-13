#------------------------------------------------------------------
# This script calculates synthetics from altar results
#------------------------------------------------------------------

# Import Python Libraries
import re
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import sys
import os
import h5py
from collections import defaultdict
import scipy
from scipy.stats import uniform
import seaborn as sns
import pandas as pd
import pdb

try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin

# Import CSI Libraries
import csi.RectangularPatches as rectFault
import csi.multifaultsolve as multiflt

# Import my libs
altarfig = __import__('altar_fig')

def getSamps(keys, filename, resdir, idxperm=None, ssh=False, **params):
    try:
        if ssh is False:
            h5file =  h5py.File(resdir+filename+'.h5','r')
        else:
            sftp_client = params['sftp_client']
            sftp_client.get(ssh+'step_final.h5',resdir+filename+'.h5')
            h5file =  h5py.File(resdir+filename+'.h5','r')
            os.remove(resdir+filename+'.h5')
      
        ## ALTAR 1
    #        samp = np.array(h5file[u'Sample Set'])
    #        samp = np.transpose(samp)
        ## AlTar 2
    #        ss = h5file['ParameterSets']['strikeslip'][()]
    #        ds = h5file['ParameterSets']['dipslip'][()]
    #        samp = np.transpose(np.hstack((ss,ds)))
    #        keys = ['ss1', 'ds1', 'ss2', 'ds2']
        sampss = []
        sampds = []
        for k in keys:
            if 'ss' in k or 'strike' in k:
                sampss.append(h5file['ParameterSets'][k][()])
            elif 'ds' in k or 'dip' in k:
                sampds.append(h5file['ParameterSets'][k][()])
        stacks=[]
        if len(sampss) == len(sampds):
            for i in range(len(sampss)):
                stacks.append(np.hstack((sampss[i],sampds[i])))
            if len(stacks) > 1:
                samp = np.transpose( np.concatenate( stacks , axis = 1 ) )
            else:
                samp = np.transpose( stacks[0] )
        else:
            ss = np.concatenate([sampss[i] for i in range(len(sampss))], axis=1)
            ds = np.concatenate([sampds[i] for i in range(len(sampds))], axis=1)
            
            if idxperm is not None: 
                ss2 = ss
                ds2 = np.empty_like(ds)
                t = idxperm['ds'][0]
                s = idxperm['ds'][1]
                ds2[:,t] = ds[:,:len(t)]
                ds2[:,s] = ds[:,len(t):]
                stacks.append( np.hstack((ss2,ds2)) )
            else:
                stacks.append( np.hstack((ss,ds)) )
            samp = np.transpose( stacks[0] )
            
    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return samp
    
    
    
def getMAP(samp, Np):
    '''
    sample
    Number of patches
    '''
    Ns = float(samp.shape[1])
    xss = np.arange(np.amin(samp[0:Np,:]), np.amax(samp[0:Np,:]), 0.5) # steps in strike
    xds = np.arange(0,np.amax(samp[Np:Np*2,:]),0.2)  # steps in dip
    postss = np.zeros((Np,len(xss)-1))
    postds = np.zeros((Np,len(xds)-1))
    modeco = np.zeros(Np*2)
    for i in range(Np): ## count number of values for each patch
        postss[i,:] = np.histogram(samp[i,:],bins=xss)[0]
        postds[i,:] = np.histogram((samp[i+Np,:]),bins=xds)[0]
        modeco[i] = xss[postss[i,:].argmax()]
        modeco[i+Np] = xds[postds[i,:].argmax()]
    
    reshaped = np.vstack((modeco[0:Np].T,modeco[Np:2*Np].T)).T
    return modeco, reshaped

def getKL(samp,relative2,samp2):
    
    def kldiv(a,b):
        a = np.asarray(a, dtype=np.float)
        b = np.asarray(b, dtype=np.float)
        return np.sum(np.where(a != 0, a * np.log(a / b), 0))
        
    Np=samp.shape[0]
    x00 = np.arange(np.amin(samp[0:Np//2,:]), np.amax(samp[0:Np//2,:]), 1) # steps in strike
    x90 = np.arange(np.amin(samp[Np//2:Np,:]),np.amax(samp[Np//2:Np,:]),1)  # steps in dip
    post00 = np.zeros((Np//2,len(x00)-1))
    post90 = np.zeros((Np//2,len(x90)-1))
    for i in range(Np//2): ## count number of values for each patch
        post00[i,:] = np.histogram(samp[i,:],bins=x00)[0]
        post90[i,:] = np.histogram((samp[i+Np//2,:]),bins=x90)[0]
    
    u = uniform(loc=-300.,scale=600.)
    u00 = u.pdf(x00)
    u90 = u.pdf(x90)
    alpha = 0.2
    
    p00= [alpha*u00[:-1] + (1-alpha)*post00[i,:] for i in range(post00.shape[0]) ]
    p90= [alpha*u90[:-1] + (1-alpha)*post90[i,:] for i in range(post90.shape[0]) ]
    
    S1=[] #strike slip
    S2=[]
    
    if relative2 == 'uniform':
        for i in range(Np//2):
            S1.append( kldiv( p00[i],u00 ))
            S2.append( kldiv( p90[i],u90 ))
        ent = np.concatenate((S1,S2))
    elif relative2 == 'samp':
        for i in range(Np//2):
            S1.append([ kldiv( p00[i],p00[j] ) for j in range(Np/2) ])
            S2.append([ kldiv( p90[i],p90[j] ) for j in range(Np/2) ])
        ent=np.zeros((Np//2,))
        for i in range(Np//2):
            ent[i]=( np.array(S1[i]).sum()+ np.array(S2[i]).sum() )/480 
            ent[i]=np.array(S2[i]).sum()/240 
    elif relative2 == 'samp2':
        post002 = np.zeros((Np//2,len(x00)-1))
        post902 = np.zeros((Np//2,len(x90)-1))
        for i in range(Np//2): ## count number of values for each patch
            post002[i,:] = np.histogram(samp2[i,:],bins=x00)[0]
            post902[i,:] = np.histogram((samp2[i+Np//2,:]),bins=x90)[0]
        p002= [alpha*u00[:-1] + (1-alpha)*post002[i,:] for i in range(post002.shape[0]) ]
        p902= [alpha*u90[:-1] + (1-alpha)*post902[i,:] for i in range(post902.shape[0]) ]
        for i in range(Np//2):
            S1.append( kldiv( p00[i],p002[i] ))
            S2.append( kldiv( p90[i],p902[i] ))
        ent = np.concatenate((S1,S2))
    else:
        print("Choose a relative between 'samp', 'uniform' and 'samp2'")
        
    ent=np.divide(ent,np.amax(ent))
    
    return ent

def resolmatrix(G,Cd,faultpar,figdir,filename):
    
    nstrike = faultpar[2]
    ndip = faultpar[3]
    
    R = []
    fig = plt.figure(figsize = (nstrike+1,ndip))
    a1 = np.dot(G,np.transpose(G)) + Cd
    a2 = np.dot(np.transpose(G),np.linalg.inv(a1))
    R = np.dot(a2,G)
    cmap = sns.cubehelix_palette(rot=-.4,light=1, as_cmap=True)
    r = R[R.shape[0]/2:R.shape[0],R.shape[1]/2:R.shape[1]]
    diag = [r[i,i] for i in range(r.shape[0])]
    r = np.reshape(diag,(ndip,nstrike))
    sns.heatmap(r,vmin=-0.1,vmax=0.8,cmap=cmap,xticklabels=False, yticklabels=False)
    fig.savefig(figdir+filename+'_model_resolution.png',format='png', dpi=300)
    return
    
def relEnt(samp,relative2,samp2,nstrike,ndip,figdir,filename):
    
    def kldiv(x,y):
        S=0
        for i in range(len(x)):
            if x[i]!=0 and y[i]!=0:
                s1 = x[i]*np.log(x[i]/y[i])
            else:
                s1=0
            S=S+s1
        return S
        
    Np=samp.shape[0]
    x00 = np.arange(np.amin(samp[0:Np//2,:]), np.amax(samp[0:Np//2,:]), 1) # steps in strike
    x90 = np.arange(np.amin(samp[Np//2:Np,:]),np.amax(samp[Np//2:Np,:]),1)  # steps in dip
    post00 = np.zeros((Np//2,len(x00)-1))
    post90 = np.zeros((Np//2,len(x90)-1))
    for i in range(Np//2): ## count number of values for each patch
        post00[i,:] = np.histogram(samp[i,:],bins=x00)[0]
        post90[i,:] = np.histogram((samp[i+Np//2,:]),bins=x90)[0]
    
    u = uniform(loc=-300.,scale=600.)
    u00 = u.pdf(x00)
    u90 = u.pdf(x90)
    alpha = 0.2
    
    p00= [alpha*u00[:-1] + (1-alpha)*post00[i,:] for i in range(post00.shape[0]) ]
    p90= [alpha*u90[:-1] + (1-alpha)*post90[i,:] for i in range(post90.shape[0]) ]
    
    S1=[] #strike slip
    S2=[]
    
    if relative2 == 'uniform':
        for i in range(Np//2):
            S1.append( kldiv( p00[i],u00 ))
            S2.append( kldiv( p90[i],u90 ))
        ent = np.concatenate((S1,S2))
    elif relative2 == 'samp':
        for i in range(Np//2):
            S1.append([ kldiv( p00[i],p00[j] ) for j in range(Np/2) ])
            S2.append([ kldiv( p90[i],p90[j] ) for j in range(Np/2) ])
        ent=np.zeros((Np//2,))
        for i in range(Np//2):
            ent[i]=( np.array(S1[i]).sum()+ np.array(S2[i]).sum() )/480 
            ent[i]=np.array(S2[i]).sum()/240 
    elif relative2 == 'samp2':
        post002 = np.zeros((Np//2,len(x00)-1))
        post902 = np.zeros((Np//2,len(x90)-1))
        for i in range(Np//2): ## count number of values for each patch
            post002[i,:] = np.histogram(samp2[i,:],bins=x00)[0]
            post902[i,:] = np.histogram((samp2[i+Np/2,:]),bins=x90)[0]
        p002= [alpha*u00[:-1] + (1-alpha)*post002[i,:] for i in range(post002.shape[0]) ]
        p902= [alpha*u90[:-1] + (1-alpha)*post902[i,:] for i in range(post902.shape[0]) ]
        for i in range(Np//2):
            S1.append( kldiv( p00[i],p002[i] ))
            S2.append( kldiv( p90[i],p902[i] ))
        ent = np.concatenate((S1,S2))
    else:
        print("Choose a relative between 'samp', 'uniform' and 'samp2'")
        
    ent=np.divide(ent,np.amax(ent))
    
    ent2=np.reshape(S2,(ndip,nstrike))
    fig = plt.figure(figsize = (nstrike+1,ndip))
    cmap = sns.cubehelix_palette(rot=-.4,light=1, as_cmap=True)
    sns.heatmap(ent2,vmin=0.,vmax=1,cmap=cmap,xticklabels=False, yticklabels=False)
    fig.savefig(figdir+filename+'_ent_rel.png',format='png', dpi=300)
    
    return ent

def makefamilies(resdir,filename,nbrfam,Np,faultpar,figdir,tol1,tol2,valmin,valmax,rand_gif=False):
    try:
        length = faultpar[0]
        width = faultpar[1]
        nstrike = faultpar[2]
        ndip = faultpar[3]
        
        h5file =  h5py.File(resdir+filename+'.h5','r')
        samples = np.array(h5file[u'Sample Set'])
        samples = np.transpose(samples)     
            
        samp = samples[Np/2:Np,:]  
        s0 = np.median(samp,axis=1)
        f = defaultdict(list)
        
        for i in reversed(range(samp.shape[1])):
            s = samp[:,i]
            j=1
            running = True
            while running:
                try:
                    if j==1 and np.allclose(s0,s,rtol=0.1,atol=tol1) == True:
                        f[1].append(samples[:,i])
                        running = False
                    elif j!=1 and np.allclose(f[j][0][Np/2:Np],s,rtol=0.1,atol=tol2) == True:
                        f[j].append(samples[:,i])
                        running = False
                    elif j == nbrfam:
                        f[j].append(samples[:,i])
                        running = False
                except IndexError:
                    f[j].append(samples[:,i])
                    running = False
                j=j+1
                
        rand = defaultdict(list)
        med = defaultdict(list)
        mean = defaultdict(list) 
        std = defaultdict(list)
        for i in range(1,nbrfam+1):
            print(len(f[i]))
            random = np.random.randint(0,len(f[i]))
            rand[i].append(f[i][random])
            med[i].append(np.median(f[i],axis=0))
            mean[i].append(np.mean(f[i],axis=0))
            std[i].append(np.std(f[i],axis=0))
                   
        if rand_gif is True:
            script1 = """
            cd {dir1}
            mkdir {dir2}
            """.format(dir1=figdir, dir2=filename+'_rand_gif')
            subprocess.call(script1, shell=True)
            
            for i in range(40):
                rand=defaultdict(list)
                for j in range(1,nbrfam+1):
                    random = np.random.randint(0,len(f[j]))
                    rand[j].append(f[j][random])
                altarfig.famslip(filename, 'rand'+str(i), rand, 25, nstrike, ndip, length, width,valmin,valmax, figdir=figdir+filename+'_rand_gif/', distinct_fam = False)
    
            script1 = """
            cd {dir1}/{dir2}
            convert -delay 10 -loop 0 *.png {of}
            gifsicle --scale 0.1 {of} -o {of2}
            """.format(dir1=figdir, dir2=filename+'_rand_gif', of=filename+'_rand.gif',of2=filename+'_randlr.gif')
            subprocess.call(script1, shell=True)
    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
    return rand,med,mean,std


def bestPredFam(models,sigmas,datasets,fault,resdir,figdir,filename,alpharake=0.05):
    '''
    Which is the family of models that allows for the best predictions?
    '''
    
    try:
        
        slv = multiflt('multiflt', [fault])
        slv.assembleGFs()
        
        RMStot = []
        for i in range(1,len(models)+1):
            slv.mpost = models[i][0]
            slv.distributem(verbose=True)
            RMS=[]
            
            for data in datasets:
                data.buildsynth(slv.faults, direction='sd')
                RMS.append(data.getRMS())
            RMStot.append(np.sum(RMS))
            
        print(RMStot)
        ind_best_fam = np.argmin(RMStot)
        best_mod = models[ind_best_fam][0]
        best_sig = sigmas[ind_best_fam][0]
        
        slv.mpost = best_mod
        slv.distributem(verbose=True)
                
        fault.writePatches2File(resdir+filename+'best_slip_dip.dat', add_slip='dipslip')
        fault.writePatches2File(resdir+filename+'best_slip_stk.dat', add_slip='strikeslip')
        fault.writePatches2File(resdir+filename+'best_slip.dat', add_slip='total')
        fault.writeSlipDirection2File(resdir+filename+'best_slipdir.dat')
        fi = open(resdir+filename+'best_slipdirrakescaled.dat', 'w')
        for p in range(len(fault.patch)):  
            xc, yc, zc, widthp, lengthp, strikep, dipp = fault.getpatchgeometry(p, center=True)  
            lonc, latc = fault.xy2ll(xc, yc)
            slip = fault.getslip(fault.patch[p]) 
            rake = np.arctan2(slip[1],slip[0])
            direc = rake*180/np.pi + strikep*180/np.pi -180
            leng = np.sqrt(slip[0]**2 + slip[1]**2)
            fi.write("%.6f %.6f %.6f %.6f %.6f\n" % (lonc,latc,zc,direc,leng*alpharake))
        fi.close()
        
        slv.mpost = best_sig
        slv.distributem(verbose=True)
        fault.writePatches2File(resdir+filename+'best_sigma_dip.dat', add_slip='dipslip')
        fault.writePatches2File(resdir+filename+'best_sigma_stk.dat', add_slip='strikeslip')
        fault.writePatches2File(resdir+filename+'best_sigma.dat', add_slip='total')
    
        for data in datasets:
            if 'gps' in data.name:
                data.write2file(filename+'best_'+data.name+'_data.dat', outDir=resdir, data='data')
                data.write2file(filename+'best_'+data.name+'_synth.dat', outDir=resdir, data='synth')
                data.write2file(filename+'best_'+data.name+'_res.dat', outDir=resdir, data='res')
            else:
                data.write2file(resdir+filename+'best_'+data.name+'_data.dat', data='data')
                data.write2file(resdir+filename+'best_'+data.name+'_synth.dat', data='synth')
                data.writeDecim2file(resdir+filename+'best_'+data.name+'_data_rect.dat', data='data')
                data.writeDecim2file(resdir+filename+'best_'+data.name+'_synth_rect.dat', data='synth')
                data.writeDecim2file(resdir+filename+'best_'+data.name+'_res_rect.dat', data='res')

    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)  
    
    return
    
def bestPredSamp(nbrmin,datasets,fault,resdir,figdir,filename,alpharake=0.05):
    '''
    Which is the sample that allows for the best predictions?
    '''
    
    try:
        
        h5file =  h5py.File(resdir+filename+'.h5','r')
        samp = np.array(h5file[u'Sample Set'])
        samp = np.transpose(samp)
        slv = multiflt('multiflt', [fault])
        slv.assembleGFs()
        
        RMStot = []
        for i in range(np.shape(samp)[1]):
            slv.mpost = samp[:,i]
            slv.distributem(verbose=True)
            RMS=[]
            
            for data in datasets:
                data.buildsynth(slv.faults, direction='sd')
                RMS.append(data.getRMS())
            RMStot.append(np.sum(RMS))
        
        if nbrmin == 1:
            ind_best = np.argmin(RMStot)
            best_mod = samp[:,ind_best]
        else:
            ##### OU moyenne des 1000 meilleurs modeles
            ind_best = np.array(RMStot).argsort()[:nbrmin]
            best_mods = samp[:,ind_best]
            best_mod = np.mean(best_mods,axis=1)
            best_sig = np.std(best_mods,axis=1)
            

        slv.mpost = best_mod
        slv.distributem(verbose=True)
                
        fault.writePatches2File(resdir+filename+'best'+str(nbrmin)+'_slip_dip.dat', add_slip='dipslip')
        fault.writePatches2File(resdir+filename+'best'+str(nbrmin)+'_slip_stk.dat', add_slip='strikeslip')
        fault.writePatches2File(resdir+filename+'best'+str(nbrmin)+'_slip.dat', add_slip='total')
        fault.writeSlipDirection2File(resdir+filename+'best'+str(nbrmin)+'_slipdir.dat')
        fi = open(resdir+filename+'best'+str(nbrmin)+'_slipdirrakescaled.dat', 'w')
        for p in range(len(fault.patch)):  
            xc, yc, zc, widthp, lengthp, strikep, dipp = fault.getpatchgeometry(p, center=True)  
            lonc, latc = fault.xy2ll(xc, yc)
            slip = fault.getslip(fault.patch[p]) 
            rake = np.arctan2(slip[1],slip[0])
            direc = rake*180/np.pi + strikep*180/np.pi -180
            leng = np.sqrt(slip[0]**2 + slip[1]**2)
            fi.write("%.6f %.6f %.6f %.6f %.6f\n" % (lonc,latc,zc,direc,leng*alpharake))
        fi.close()
        
        try:
          best_sig
        except NameError:
          pass
        else:
            slv.mpost = best_sig
            slv.distributem(verbose=True)
            fault.writePatches2File(resdir+filename+'best'+str(nbrmin)+'_sigma_dip.dat', add_slip='dipslip')
            fault.writePatches2File(resdir+filename+'best'+str(nbrmin)+'_sigma_stk.dat', add_slip='strikeslip')
            fault.writePatches2File(resdir+filename+'best'+str(nbrmin)+'_sigma.dat', add_slip='total')
    
        for data in datasets:
            if 'gps' in data.name:
                data.write2file(filename+'best'+str(nbrmin)+'_'+data.name+'_data.dat', outDir=resdir, data='data')
                data.write2file(filename+'best'+str(nbrmin)+'_'+data.name+'_synth.dat', outDir=resdir, data='synth')
                data.write2file(filename+'best'+str(nbrmin)+'_'+data.name+'_res.dat', outDir=resdir, data='res')
            else:
                data.write2file(resdir+filename+'best'+str(nbrmin)+'_'+data.name+'_data.dat', data='data')
                data.write2file(resdir+filename+'best'+str(nbrmin)+'_'+data.name+'_synth.dat', data='synth')
                data.writeDecim2file(resdir+filename+'best'+str(nbrmin)+'_'+data.name+'_data_rect.dat', data='data')
                data.writeDecim2file(resdir+filename+'best'+str(nbrmin)+'_'+data.name+'_synth_rect.dat', data='synth')
                data.writeDecim2file(resdir+filename+'best'+str(nbrmin)+'_'+data.name+'_res_rect.dat', data='res')

    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    
    return
    
def calcSynth(datasets,keys,filename,faults,resdir,figdir, idxperm=None, cd=None,nalpha=False,alpharake=0.002,ssh=False,**params):
    '''
    nalpha: to smooth the slip distribution (average over several subfaults)
    ex: if the firsts 3 subfaults remain and the 6 deepest are averaged over 3 2x2 subfaults: nalpha = [3,3]
    if nalpha is not False, please specify in **params:
    - nstrike
    - ndip
    - boundss=[-5,5]
    - boundds=[0,60]
    - valmax=25
    
    zneg : if True, convert subfault depth to negative values
    alpharake: length of the slip vector to plot in map view with GMT
    idxperm: index of permutations in GFs (used to use different psets for priors)
    
    ssh: False, or path in ssh client to open result files
    if ssh is not False, please specify in **params:
    - sftp_client
    '''

    try:    
        if faults.__class__ is not list :
            lon0 = faults.lon0
            lat0 =  faults.lat0
        else:
            lon0 = faults[0].lon0
            lat0 = faults[0].lat0

        #---------------------------------------------------------------
        # get results from ALTAR

        if ssh is False:
            h5file =  h5py.File(resdir+filename+'.h5','r')
        else:
            sftp_client = params['sftp_client']
            sftp_client.get(ssh+'step_final.h5',resdir+filename+'.h5')
            h5file =  h5py.File(resdir+filename+'.h5','r')
            os.remove(resdir+filename+'.h5')
      
        ## ALTAR 1
#        samp = np.array(h5file[u'Sample Set'])
#        samp = np.transpose(samp)
        ## AlTar 2
#        ss = h5file['ParameterSets']['strikeslip'][()]
#        ds = h5file['ParameterSets']['dipslip'][()]
#        samp = np.transpose(np.hstack((ss,ds)))
#        keys = ['ss1', 'ds1', 'ss2', 'ds2']
        sampss = []
        sampds = []
        for k in keys:
            if 'ss' in k or 'strike' in k:
                sampss.append(h5file['ParameterSets'][k][()])
            elif 'ds' in k or 'dip' in k:
                sampds.append(h5file['ParameterSets'][k][()])
        stacks=[]
        if len(sampss) == len(sampds):
            for i in range(len(sampss)):
                stacks.append(np.hstack((sampss[i],sampds[i])))
            if len(stacks) > 1:
                samp = np.transpose( np.concatenate( stacks , axis = 1 ) )
            else:
                samp = np.transpose( stacks[0] )
        else:
            ss = np.concatenate([sampss[i] for i in range(len(sampss))], axis=1)
            ds = np.concatenate([sampds[i] for i in range(len(sampds))], axis=1)
            
            if idxperm is not None: 
                ss2 = ss
                ds2 = np.empty_like(ds)
                t = idxperm['ds'][0]
                s = idxperm['ds'][1]
                ds2[:,t] = ds[:,:len(t)]
                ds2[:,s] = ds[:,len(t):]
                stacks.append( np.hstack((ss2,ds2)) )
            else:
                stacks.append( np.hstack((ss,ds)) )
            samp = np.transpose( stacks[0] )
        
        if faults.__class__ is not list :   
            slv = multiflt('multiflt', [faults])
        else:
            slv = multiflt('multiflt', faults)
        slv.assembleGFs()
        
        #---- calculate posterior uncertainty on data and write Sigma files
        if faults.__class__ is not list :
            std = [np.std(samp[i,:]) for i in range(np.shape(samp)[0]) ]
            slv.mpost = np.array(std)
            slv.writeMpost2File(outfile=resdir+filename+'_sigma_csi.dat')
            slv.distributem(verbose=True)
            sigma = slv.mpost
        else:
            sigma = []
            for sta in stacks:
                std = [np.std( np.transpose(sta)[i,:] ) for i in range(np.shape(np.transpose(sta))[0]) ]
                sigma.append(std)
            std = [np.std(samp[i,:]) for i in range(np.shape(samp)[0]) ]
            slv.mpost = np.array(std)
            slv.writeMpost2File(outfile=resdir+filename+'_sigma_csi.dat')
            slv.distributem(verbose=True)
        errpost = {}
        for data in datasets:
            if data.vertical == True:
                data.buildsynth(slv.faults, direction='sd')
            else:
                data.buildsynth(slv.faults, direction='sd',vertical=False)
            if data.dtype != 'opticorr':
                errpost[data.name]=data.synth
#        if faults.__class__ is not list :
#            fault=faults
#            fault.writePatches2File(resdir + filename + '_sigma_dip.dat', add_slip='dipslip')
#            fault.writePatches2File(resdir + filename + '_sigma_stk.dat', add_slip='strikeslip')
#            fault.writePatches2File(resdir + filename + '_sigma.dat', add_slip='total')
#        else:
#            for fault in faults:
#                fault.writePatches2File(resdir+filename+'_'+fault.name+'_sigma_dip.dat', add_slip='dipslip')
#                fault.writePatches2File(resdir+filename+'_'+fault.name+'_sigma_stk.dat', add_slip='strikeslip')
#                fault.writePatches2File(resdir+filename+'_'+fault.name+'_sigma.dat', add_slip='total')

         #---- calculate synths
        
        if nalpha is False:
            moy = np.mean( samp, axis=1 )
            med = np.median( samp, axis=1 ) 
            
        elif nalpha is not False:
            nstrike = params['nstrike']
            ndip = params['ndip']
            boundss = params['boundss']
            boundds = params['boundds']
            valmax = params['valmax']

        #---- calculate synths with smoothed parameters (down dip)
            moyds,medds,moyss,medss= altarfig.pdf_moy(filename, nalpha, nstrike, ndip, resdir, savedir=figdir+'/pdf_filter/',boundss=boundss, boundds=boundds,valmax=valmax)
            moy_ds=[]
            med_ds=[]
            moy_ss=[]
            med_ss=[]
            for i in range(nalpha[0]*nstrike):
                moy_ds.append(moyds[i])
                med_ds.append(medds[i])
                moy_ss.append(moyss[i])
                med_ss.append(medss[i])
            rang = [k for k in range(nalpha[0]*nstrike,nalpha[0]*nstrike + nalpha[1]*nstrike/2,nstrike/2) for _ in range(2)]
            for j in rang:
                for i in range(nstrike/2):
                    moy_ds.append(moyds[j+i])
                    moy_ds.append(moyds[j+i])
                    med_ds.append(medds[j+i])
                    med_ds.append(medds[j+i])
                    moy_ss.append(moyss[j+i])
                    moy_ss.append(moyss[j+i])
                    med_ss.append(medss[j+i])
                    med_ss.append(medss[j+i])
            if len(nalpha)==3:
                rang2 = [k for k in range(nalpha[0]*nstrike+ nalpha[1]*nstrike/2,nalpha[0]*nstrike+nalpha[1]*nstrike/2+nalpha[2]*nstrike/3,nstrike/3) for _ in range(3)]
                for j in rang2:
                    for i in range(nstrike/3):
                        moy_ds.append(moyds[j+i])
                        moy_ds.append(moyds[j+i])
                        moy_ds.append(moyds[j+i])
                        med_ds.append(medds[j+i])
                        med_ds.append(medds[j+i])
                        med_ds.append(medds[j+i])
                        moy_ss.append(moyss[j+i])
                        moy_ss.append(moyss[j+i])
                        moy_ss.append(moyss[j+i])
                        med_ss.append(medss[j+i])
                        med_ss.append(medss[j+i])
                        med_ss.append(medss[j+i])
            moy = np.hstack([moy_ss,moy_ds])
#            med =  np.hstack([med_ss,med_ds])

            
        
        mpost = moy
        slv.mpost=mpost
        slv.writeMpost2File(outfile=resdir+filename+'_slip_csi.dat')
        slv.distributem(verbose=True)
        RMS=[]
        fi = open(resdir+filename+'_rms.dat', 'w')
        for data in datasets:
            if data.vertical == True:
                data.buildsynth(slv.faults, direction='sd')
            else:
                data.buildsynth(slv.faults, direction='sd',vertical=False)
            if data.dtype != 'opticorr':
                RMS.append(data.getRMS()[1])
                print('RMS '+data.name,data.getRMS()[1])
                fi.write("%s %.6f\n" % (data.name,data.getRMS()[1]))
        fi.close()
        print(RMS)
        rms_tot = np.sum(RMS)
        
        ##---------------------------------------------------------------
        ## Slip centers file and slip amplitude to plot contours iwth GMT
        slp = []
        for i in range(len(slv.faults[0].slip[:, 0])):
            if np.abs(slv.faults[0].slip[i, 1]) <= 4000.:
                slp.append(np.abs(slv.faults[0].slip[i, 1]) )
            else:
                slp.append( np.sqrt(slv.faults[0].slip[i, 0] ** 2 + slv.faults[0].slip[i, 1] ** 2) )
        slp = np.array(slp)
        cent = slv.faults[0].getcenters()
        cent2 = [slv.faults[0].xy2ll(cent[i][0], cent[i][1]) for i in range(len(cent))]
        fi = open(resdir + filename + '_slipcenterll.dat', 'w')
        try:
            for p in range(len(slv.faults[0].patch)):
                fi.write("%.6f %.6f %.6f\n" % (cent2[p][0], cent2[p][1], slp[p]))
        except IndexError:
            print('Tent fault detected')
            print("If that's not the case, IndexError exception")
            for p in range(len(slv.faults[0].tent)):
                fi.write("%.6f %.6f %.6f\n" % (cent2[p][0], cent2[p][1], slp[p]))
        fi.close()

        ## Slip direction to plot arrows with GMT
        if faults.__class__ is not list :
            fault=faults
#            fault.writePatches2File(resdir+filename+'_slip_dip.dat', add_slip='dipslip')
#            fault.writePatches2File(resdir+filename+'_slip_stk.dat', add_slip='strikeslip')
#            fault.writePatches2File(resdir+filename+'_slip.dat', add_slip='total')
            fault.writeSlipDirection2File(resdir+filename+'_slipdir.dat')
            fault.writeSlipDirection2File(resdir+filename+'_slipdir_tot_scaled.dat', scale='total', factor=alpharake)
            fi = open(resdir+filename+'_slipdirrakescaled.dat', 'w')
            for p in range(len(fault.patch)):  
                xc, yc, zc, widthp, lengthp, strikep, dipp = fault.getpatchgeometry(p, center=True)  
                lonc, latc = fault.xy2ll(xc, yc)
                slip = fault.getslip(fault.patch[p]) 
                rake = np.arctan2(slip[1],slip[0])
                direc = rake*180/np.pi + strikep*180/np.pi -180
                leng = np.sqrt(slip[0]**2 + slip[1]**2)
                fi.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (lonc,latc,zc,direc,leng*alpharake,rake*180./np.pi))
            fi.close()
        else:
            for fault in faults:
#                fault.writePatches2File(resdir+filename+'_'+fault.name+'_slip_dip.dat', add_slip='dipslip')
#                fault.writePatches2File(resdir+filename+'_'+fault.name+'_slip_stk.dat', add_slip='strikeslip')
#                fault.writePatches2File(resdir+filename+'_'+fault.name+'_slip.dat', add_slip='total')
                fault.writeSlipDirection2File(resdir+filename+'_'+fault.name+'_slipdir.dat')
                fault.writeSlipDirection2File(resdir+filename+'_'+fault.name+'_slipdir_tot_scaled.dat', scale='total', factor=alpharake)
                try:
                    fi = open(resdir+filename+'_'+fault.name.replace(' ','_')+'_slipdirrakescaled.dat', 'w')
                    for p in range(len(fault.patch)):  
                        xc, yc, zc, widthp, lengthp, strikep, dipp = fault.getpatchgeometry(p, center=True)  
                        lonc, latc = fault.xy2ll(xc, yc)
                        slip = fault.getslip(fault.patch[p]) 
                        rake = np.arctan2(slip[1],slip[0])
                        direc = rake*180/np.pi + strikep*180/np.pi -180
                        leng = np.sqrt(slip[0]**2 + slip[1]**2)
                        fi.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (lonc,latc,zc,direc,leng*alpharake,rake*180./np.pi))
                    fi.close()
                except IndexError:
                    print('Tent fault detected')
                    print("If that's not the case, IndexError exception")
                    fi = open(resdir+filename+'_'+fault.name.replace(' ','_')+'_slipdirrakescaled.dat', 'w')
                    for p in range(len(fault.tent)):  
                        xc, yc, zc, widthp, lengthp = fault.getTentInfo(p)  
                        lonc, latc = fault.xy2ll(xc, yc)
                        slip = fault.getslip(fault.patch[p]) 
                        rake = np.arctan2(slip[1],slip[0])
                        direc = rake*180/np.pi + strikep*180/np.pi -180
                        leng = np.sqrt(slip[0]**2 + slip[1]**2)
                        fi.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (lonc,latc,zc,direc,leng*alpharake,rake*180./np.pi))
                    fi.close()

        for data in datasets:
            if data.dtype == 'gps':
                if cd is not None:  ## if Cp
                    data.err_enu = np.reshape(cd[0:len(data.x)*3],(len(data.x),3))
                data.write2file(filename+'_'+data.name+'_data.dat', outDir=resdir, data='data')
                memory = data.err_enu
                data.err_enu=errpost[data.name]
                data.write2file(filename+'_'+data.name+'_synth.dat', outDir=resdir, data='synth')
                data.write2file(filename+'_'+data.name+'_res.dat', outDir=resdir, data='res')
                data.err_enu = memory
            elif data.dtype == 'insar':
                data.write2file(resdir+filename+'_'+data.name+'_data.dat', data='data')
                data.write2file(resdir+filename+'_'+data.name+'_synth.dat', data='synth')
                data.write2file(resdir+filename+'_'+data.name+'_res.dat', data='resid')
                try:
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_data_rect.dat', data='data')
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_synth_rect.dat', data='synth')
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_res_rect.dat', data='res')
                except:
                    print("Cannot write InSAR Rectangles to file, pass")
                    pass
            elif data.dtype == 'opticorr':
                try:
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_data_east.dat', data='dataEast')
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_synth_east.dat', data='synthEast')
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_data_north.dat', data='dataNorth')
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_synth_north.dat', data='synthNorth')
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_data_rect.dat', data='data')
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_synth_rect.dat', data='synth')
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_res_rect.dat', data='res')
                except:
                    print("Cannot write opticorr Rectangles to file, pass")
                    pass
#
        print('---------------------------------')
        print('---------------------------------')
        print('Done! You can find files here  --->  '+resdir)

    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    if np.shape(faults)==() or np.shape(faults)[0]==1:
        return faults, sigma, samp
    else:
        return faults, sigma, samp

def buildSynthCopostGPS(faults, gps, direction='sd',  vertical = True):
    '''
    faults = [faultco,faultpost]
    '''
    try:
        gps.synth = np.zeros((gps.x.shape[0],3))
        Nd = gps.x.shape[0]
        # Check components
        east     = False
        north    = False
        if not np.isnan(gps.vel_enu[:,0]).any():
            east = True
        if not np.isnan(gps.vel_enu[:,1]).any():
            north = True
        if not np.isnan(gps.vel_enu[:,2]).any() and vertical:
            vertical = True
        
        Gco = faults[0].G[gps.name] #Gco,co+post
        Gcopo = faults[0].G[gps.name] #Gpost,co+post= Gco,co+post
        if ('s' in direction):
            Gsco = Gco['strikeslip']
            Gscopo = Gcopo['strikeslip']
            Ssco = faults[0].slip[:,0]
            Sspo = faults[1].slip[:,0]
            ss_synth = np.dot(Gsco,Ssco) + np.dot(Gscopo,Sspo)
            N = 0
            if east:
                gps.synth[:,0] += ss_synth[0:Nd]
                N += Nd
            if north:
                gps.synth[:,1] += ss_synth[N:N+Nd]
                N += Nd
            if vertical:
                # if ss_synth.size > 2*Nd and east and north:
                gps.synth[:,2] += ss_synth[N:N+Nd]
        if ('d' in direction):
            Gdco = Gco['dipslip']
            Gdcopo = Gcopo['dipslip']
            Sdco = faults[0].slip[:,1]
            Sdpo = faults[1].slip[:,1]
            ds_synth = np.dot(Gdco, Sdco)+ np.dot(Gdcopo, Sdpo)
            N = 0
            if east:
                gps.synth[:,0] += ds_synth[0:Nd]
                N += Nd
            if north:
                gps.synth[:,1] += ds_synth[N:N+Nd]
                N += Nd
            if vertical:
                #if ds_synth.size > 2*Nd and east and north:
                gps.synth[:,2] += ds_synth[N:N+Nd]
    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return
    
def buildSynthCopostInSAR(faults, sar, direction='sd',  vertical = True):
    '''
    faults = [faultco,faultpost]
    '''
    try:
        direction='sd'
        faults[0].buildGFs(sar, slipdir='sd')
        faults[1].buildGFs(sar, slipdir='sd')
        Gco = faults[0].G[sar.name] #Gco,co+post
        Gpost = faults[1].G[sar.name] #Gpost,co+post= Gco,co+post
        sar.synth = np.zeros((sar.vel.shape))
        if ('s' in direction):
            Gsco = Gco['strikeslip']
            Gspost = Gpost['strikeslip']
            Ssco = faults[0].slip[:,0]
            Sspost = faults[1].slip[:,0]
            losss_synth = np.dot(Gsco,Ssco) + np.dot(Gspost,Sspost)
            sar.synth += losss_synth
        if ('d' in direction):
            Gdco = Gco['dipslip']
            Gdpost = Gpost['dipslip']
            Sdco = faults[0].slip[:,1]
            Sdpost = faults[1].slip[:,1]
            losds_synth = np.dot(Gdco, Sdco)+ np.dot(Gdpost, Sdpost)
            sar.synth += losds_synth
            
    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return
            
def calcSynthCopost(datasets,name,slv,resdir,figdir,filename,cdfile,edks=False,source_spacing=0.5,**params):
    '''
    If edks is True, please specify **params: edksdir and modelname
    
    datasets = [dataco, datacopost, datapost, datapost2, ...]
    slv.faults = [faultco, faultpost] !!
    
    '''

    try:    
        lon0 = slv.faults[0].lon0
        lat0 = slv.faults[0].lat0
                        
        h5file =  h5py.File(resdir+filename+'.h5','r')
        samp = np.array(h5file[u'Sample Set'])
        samp = np.transpose(samp)
        moy = np.mean( samp, axis=1 )
        med = np.median( samp, axis=1 ) 
        
        #---- calculate posterior uncertainty on data
        cov = np.array(h5file[u'Covariance'])  
        slv.mpost = np.sqrt(np.diagonal(cov))
        slv.distributem(verbose=True)
        errpost={}
        for data in datasets[0]:
            data.buildsynth(slv.faults[0], direction='sd')
            errpost[data.name]=data.synth/4
        for data in datasets[1]:    
            if data.dtype is 'insar':
                buildSynthCopostInSAR(slv.faults, data)
            elif data.dtype is 'gps':
                buildSynthCopostGPS(slv.faults, data)
            errpost[data.name]=data.synth/4
        for data in datasets[2]:
            data.buildsynth(slv.faults[1], direction='sd')
            errpost[data.name]=data.synth/4

      
        #---- calculate synths
        slv.mpost = moy[0:slv.faults[0].N_slip*2]
        slv.writeMpost2File(outfile=resdir+filename+'_slip_csi_co.dat')
        slv.mpost = moy[slv.faults[0].N_slip*2:slv.faults[0].N_slip*4]
        slv.writeMpost2File(outfile=resdir+filename+'_slip_csi_post.dat')
        
        slv.mpost = moy
        slv.writeMpost2File(outfile=resdir+filename+'_slip_csi_all.dat')
        slv.distributem(verbose=True)
        RMS=[]
        fi = open(resdir+filename+'_rms.dat', 'w')
        for data in datasets[0]:
            data.buildsynth(slv.faults[0], direction='sd')
            RMS.append(data.getRMS())
            print('RMS '+data.name,data.getRMS()[1])
            fi.write("%s %.6f\n" % (data.name,data.getRMS()[1]))
        for data in datasets[1]:    
            if data.dtype is 'insar':
                buildSynthCopostInSAR(slv.faults, data)
            elif data.dtype is 'gps':
                buildSynthCopostGPS(slv.faults, data)
            RMS.append(data.getRMS())
            print('RMS '+data.name,data.getRMS()[1])
            fi.write("%s %.6f\n" % (data.name,data.getRMS()[1]))
        for data in datasets[2]:
            data.buildsynth(slv.faults[1], direction='sd')
            RMS.append(data.getRMS())
            print('RMS '+data.name,data.getRMS()[1])
            fi.write("%s %.6f\n" % (data.name,data.getRMS()[1]))
        fi.close()

        ##---------------------------------------------------------------  
        slv.faults[0].writePatches2File(resdir+filename+'_slip_dip_co.dat', add_slip='dipslip')
        slv.faults[0].writePatches2File(resdir+filename+'_sigma_dip_co.dat', add_slip='dipslip',stdh5=resdir+filename+'.h5')
        slv.faults[0].writePatches2File(resdir+filename+'_slip_stk_co.dat', add_slip='strikeslip')
        slv.faults[0].writePatches2File(resdir+filename+'_sigma_stk_co.dat', add_slip='strikeslip',stdh5=resdir+filename+'.h5')
        slv.faults[0].writePatches2File(resdir+filename+'_slip_co.dat', add_slip='total')
        slv.faults[0].writePatches2File(resdir+filename+'_sigma_co.dat', add_slip='total',stdh5=resdir+filename+'.h5')
        slv.faults[0].writeSlipDirection2File(resdir+filename+'_slipdir_co.dat')
#        slv.faults[0].writeSlipDirection2File(resdir+filename+'_slipdir_tot_scaled.dat', scale='total', factor=0.01)
        fi1 = open(resdir+filename+'_slipdirrakescaled_co.dat', 'w')
        for p in range(len(slv.faults[0].patch)):  
            xc, yc, zc, widthp, lengthp, strikep, dipp = slv.faults[0].getpatchgeometry(p, center=True)  
            lonc, latc = slv.faults[0].xy2ll(xc, yc)
            slip = slv.faults[0].getslip(slv.faults[0].patch[p]) 
            rake = np.arctan2(slip[1],slip[0])
            direc = rake*180/np.pi + strikep*180/np.pi -180
            leng = np.sqrt(slip[0]**2 + slip[1]**2)
            fi1.write("%.6f %.6f %.6f %.6f %.6f\n" % (lonc,latc,zc,direc,leng*0.001))
        fi1.close()
        
        slv.faults[1].writePatches2File(resdir+filename+'_slip_dip_po.dat', add_slip='dipslip')
        slv.faults[1].writePatches2File(resdir+filename+'_sigma_dip_po.dat', add_slip='dipslip',stdh5=resdir+filename+'.h5')
        slv.faults[1].writePatches2File(resdir+filename+'_slip_stk_po.dat', add_slip='strikeslip')
        slv.faults[1].writePatches2File(resdir+filename+'_sigma_stk_po.dat', add_slip='strikeslip',stdh5=resdir+filename+'.h5')
        slv.faults[1].writePatches2File(resdir+filename+'_slip_po.dat', add_slip='total')
        slv.faults[1].writePatches2File(resdir+filename+'_sigma_po.dat', add_slip='total',stdh5=resdir+filename+'.h5',post=True)
        slv.faults[1].writeSlipDirection2File(resdir+filename+'_slipdir_po.dat')
#        slv.faults[1].writeSlipDirection2File(resdir+filename+'_slipdir_tot_scaled.dat', scale='total', factor=0.01)
        fi = open(resdir+filename+'_slipdirrakescaled_po.dat', 'w')
        for p in range(len(slv.faults[1].patch)):  
            xc, yc, zc, widthp, lengthp, strikep, dipp = slv.faults[1].getpatchgeometry(p, center=True)  
            lonc, latc = slv.faults[1].xy2ll(xc, yc)
            slip = slv.faults[1].getslip(slv.faults[1].patch[p]) 
            rake = np.arctan2(slip[1],slip[0])
            direc = rake*180/np.pi + strikep*180/np.pi -180
            leng = np.sqrt(slip[0]**2 + slip[1]**2)
            fi.write("%.6f %.6f %.6f %.6f %.6f\n" % (lonc,latc,zc,direc,leng*0.002))
        fi.close()

        slp = np.sqrt(slv.faults[0].slip[:,0]**2 + slv.faults[0].slip[:,1]**2)
        cent=slv.faults[0].getcenters()
        cent2=[slv.faults[0].xy2ll(cent[i][0],cent[i][1]) for i in range(len(cent))]
        fi = open(resdir+filename+'_slipcenterll_co.dat', 'w')
        for p in range(len(slv.faults[0].patch)): 
            fi.write("%.6f %.6f %.6f\n" % (cent2[p][0],cent2[p][1],slp[p]))
        fi.close()
        slp = np.sqrt(slv.faults[1].slip[:,0]**2 + slv.faults[1].slip[:,1]**2)
        cent=slv.faults[1].getcenters()
        cent2=[slv.faults[1].xy2ll(cent[i][0],cent[i][1]) for i in range(len(cent))]
        fi = open(resdir+filename+'_slipcenterll_po.dat', 'w')
        for p in range(len(slv.faults[1].patch)): 
            fi.write("%.6f %.6f %.6f\n" % (cent2[p][0],cent2[p][1],slp[p]))
        fi.close()
        
        for data in datasets[0]+datasets[1]+datasets[2]:
            if 'gps' in data.name:
                data.write2file(filename+'_'+data.name+'_data.dat', outDir=resdir, data='data')
                data.err_enu=errpost[data.name]
                data.write2file(filename+'_'+data.name+'_synth.dat', outDir=resdir, data='synth')
                data.write2file(filename+'_'+data.name+'_res.dat', outDir=resdir, data='res')
            else:
                data.write2file(resdir+filename+'_'+data.name+'_data.dat', data='data')
                data.write2file(resdir+filename+'_'+data.name+'_synth.dat', data='synth')
                data.writeDecim2file(resdir+filename+'_'+data.name+'_data_rect.dat', data='data')
                data.writeDecim2file(resdir+filename+'_'+data.name+'_synth_rect.dat', data='synth')
                data.writeDecim2file(resdir+filename+'_'+data.name+'_res_rect.dat', data='res')


        print('---------------------------------')
        print('---------------------------------')
        print('Done! You can find figures here  --->  '+figdir)

    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
    return

def makeSynth(datasets,filename,fault,modelsynth,resdir,noise=None,alpharake=0.002):
    '''
    filename: name of the synthetic model to be created

    Creates predictions for a synthetic slip model. For instance, a checkerboard test.
    modelsynth : the synthetic slip model
    noise: noise amplitude in percentage of the max amplitude of dataset
        
    ex: model = np.zeros(np.shape(one))
        model[Np/2+60+4:Np/2+60+8]=-150.
        model[Np/2+80+4:Np/2+80+8]=-150.
        model[Np/2+100+13:Np/2+100+16]=-150.
        model[Np/2+120+13:Np/2+120+16]=-150.
        model[Np/2+140+13:Np/2+140+16]=-150.
        
    Note: GFs already calculated and stored in fault!
    '''

    try:

        slv = multiflt(filename, [fault])
        slv.assembleGFs()
        slv.mpost=modelsynth
        slv.writeMpost2File(outfile=resdir+filename+'_slip_csi.dat')
        slv.distributem(verbose=True)
        faults = slv.faults
        for data in datasets:
            data.buildsynth(faults, direction='sd')
        
        if noise is None:
            pass
        elif isinstance(noise, (dict)):
            for data in datasets:
                if data.dtype == 'gps':
                    data.synth[:,0] += noise[data.name][:,0]
                    data.synth[:,1] += noise[data.name][:,1]
                    data.synth[:,2] += noise[data.name][:,2]
                    data.err_enu[:,0] = 1.2*noise[data.name][:,0]
                    data.err_enu[:,1] = 1.2*noise[data.name][:,1]
                    data.err_enu[:,2] = 1.2*noise[data.name][:,2]
                else:
                    data.synth += noise[data.name]
        elif isinstance(noise, (float)):   
            import gstools as gs
            corr_len = 10. ## 10 km
            corr_len2 = 50.
            for data in datasets:
                white_noise = np.random.normal(0, 1, size=len(data.x))
                x = np.linspace(np.amin(data.x),np.amax(data.x),100)
                y = np.linspace(np.amin(data.y),np.amax(data.y),100)
                
                model = gs.Gaussian(dim=2, var=1, len_scale=corr_len)
                srf = gs.SRF(model,seed=20200503)
                field = srf.structured([x, y])
                field /= np.amax(field)
                
                model2 = gs.Gaussian(dim=2, var=1, len_scale=corr_len2)
                srf2 = gs.SRF(model2,seed=202005032)
                field2 = srf2.structured([x, y])
                field2 /= np.amax(field2)
                
                corr_noise1 = []
                corr_noise2 = []
                for i in range(len(data.x)):
                    indx = np.argmin(np.abs(np.array(x)-data.x[i]))
                    indy = np.argmin(np.abs(np.array(y)-data.y[i]))
                    corr_noise1.append(field[indx,indy])
                    corr_noise2.append(field2[indx,indy])
                tot_noise = (white_noise + corr_noise1 + corr_noise2)/3
                
                if data.dtype == 'gps':
                    tot_noise *= (noise/100.)*np.amax(data.synth)
                    data.synth[:,0] += tot_noise
                    data.synth[:,1] += tot_noise
                    data.synth[:,2] += tot_noise
                else:
                    tot_noise *= (noise/100.)*np.amax(data.synth)
                    data.synth += tot_noise
                    
        if 'Tents' in str(type(fault)):
            fault.writeNodes2File(resdir+filename+'_slip_dip.dat', add_slip='dipslip')
            fault.writeNodes2File(resdir+filename+'_slip_stk.dat', add_slip='strikeslip')
            fault.writeNodes2File(resdir+filename+'_slip.dat', add_slip='total')
            # fault.writeSlipDirection2File(resdir+filename+'_slipdir.dat')
            # fault.writeSlipDirection2File(resdir+filename+'_slipdir_tot_scaled.dat', scale='total', factor=0.1)
        else:
            fault.writePatches2File(resdir+filename+'_slip_dip.dat', add_slip='dipslip')
            fault.writePatches2File(resdir+filename+'_slip_stk.dat', add_slip='strikeslip')
            fault.writePatches2File(resdir+filename+'_slip.dat', add_slip='total')
            # fault.writeSlipDirection2File(resdir+filename+'_slipdir.dat')
            # fault.writeSlipDirection2File(resdir+filename+'_slipdir_tot_scaled.dat', scale='total', factor=0.1)

        # with open(resdir+filename+'_slipdirrakescaled.dat', 'w') as fi:
        #     if 'Tents' in str(type(fault)):
        #         strikes = fault.getStrikes()
        #         for p in range(len(fault.Vertices)):
        #             xc, yc, zc = fault.Vertices[p]
        #             strikep = strikes[p]
        #             lonc, latc = fault.xy2ll(xc, yc)
        #             slip = fault.slip[p]
        #             rake = np.arctan2(slip[1],slip[0]) #because postive dip slip is towards the top of the fault
        #             direc = rake*180/np.pi + strikep*180/np.pi -180
        #             leng = np.sqrt(slip[0]**2 + slip[1]**2)
        #             fi.write("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (xc, yc,lonc,latc,zc,direc,leng*alpharake,rake*180./np.pi))
        #     else:
        #         for p in range(len(fault.patch)):
        #             xc, yc, zc, widthp, lengthp, strikep, dipp = fault.getpatchgeometry(p, center=True)
        #             lonc, latc = fault.xy2ll(xc, yc)
        #             slip = fault.getslip(fault.patch[p])
        #             rake = np.arctan2(slip[1],slip[0]) #because postive dip slip is towards the top of the fault
        #             direc = rake*180/np.pi + strikep*180/np.pi -180
        #             leng = np.sqrt(slip[0]**2 + slip[1]**2)
        #             fi.write("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (xc, yc, lonc,latc,zc,direc,leng*alpharake,rake*180./np.pi))
                    
        synths = {}
        for data in datasets:
            synths[data.name] = data.synth
            if data.dtype == 'gps':
                data.write2file(filename+'_'+data.name+'_data.dat', outDir=resdir, data='data')
                data.write2file(filename+'_'+data.name+'_synth.dat', outDir=resdir, data='synth')
                data.write2file(filename+'_'+data.name+'_res.dat', outDir=resdir, data='res')
            else:
                data.write2file(resdir+filename+'_'+data.name+'_data.dat', data='data')
                data.write2file(resdir+filename+'_'+data.name+'_synth.dat', data='synth')
                try:
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_data_rect.dat', data='data')
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_synth_rect.dat', data='synth')
                    data.writeDecim2file(resdir+filename+'_'+data.name+'_res_rect.dat', data='res')
#                    data.writeDecim2file(resdir+filename+'_'+data.name+'_err_rect.dat', data='err')
                except:
                    print("Cannot write InSAR Rectangles to file, pass")
                    pass
                
    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return fault, slv.mpost, synths

def isTargetinSample(filename,trgt,resdir,figdir,alpharake=0.002,ssh=False,**params):
    '''
    alpharake: length of the slip vector to plot in map view with GMT

    ssh: False, or path in ssh client to open result files
    if ssh is not False, please specify in **params:
    - sftp_client
    '''

    try:    
        #---------------------------------------------------------------
        # get results from ALTAR
        if ssh is False:
            h5file =  h5py.File(resdir+filename+'.h5','r')
        else:
            sftp_client = params['sftp_client']
            sftp_client.get(ssh+'step_final.h5',resdir+filename+'.h5')
            h5file =  h5py.File(resdir+filename+'.h5','r')
            os.remove(resdir+filename+'.h5')
      
        ## ALTAR 1
        # samp = np.array(h5file[u'Sample Set'])
        # samp = np.transpose(samp)
        ## AlTar 2
        ss = h5file['ParameterSets']['strikeslip'][()]
        ds = h5file['ParameterSets']['dipslip'][()]
        samp = np.transpose(np.hstack((ss,ds)))
#        ss1 = h5file['ParameterSets']['ss1'][()]
#        ds1 = h5file['ParameterSets']['ds1'][()]   
#        ss2 = h5file['ParameterSets']['ss2'][()]
#        ds2 = h5file['ParameterSets']['ds2'][()]  
#        A = np.hstack((ss1,ds1))
#        B = np.hstack((ss2,ds2))
#        samp = np.transpose(np.hstack((A,B)))
 
        moy = np.mean( samp, axis=1 )
        med = np.median( samp, axis=1 ) 
        
        rms = []
        for s in range(np.shape(samp)[1]):
            res = np.sqrt(np.sum((samp[0:70,s]-trgt[0:70])**2))
            rms.append(res)
            
#        ind_best = np.argmin(rms)
#        sample_trgt = samp[:,ind_best]
        ind_best = np.argsort(rms)[:10]
        samples = [samp[:,i] for i in ind_best]
        
#        mini = []
#        for s in range(np.shape(samples)[0]):
#            mini.append(np.minimum(samples[0:70,s],trgt[0:70]))
#        
#        inde = np.argwhere
#        ind_best = np.argmin(l2)
#        sample_trgt = samp[:,ind_best]
            
    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)    
    return sample_trgt

def calcM0(moy,cov,surface):
    '''
    M0 in N.m
    
    moy: slip in m
    surface is an array, or a float
    '''
#    cov=np.sqrt(np.diagonal(np.array(h5file[u'Covariance'])))  
    sumco = np.sum( surface*moy[:,0] ) + np.sum( surface*moy[:,1] )
    M0co = sumco * 3.*10**10
    
    cov1 = surface*(cov[0:len(moy)])
    cov2 = surface*(cov[len(moy)-1:-1])
    M0cosd = (0.5* np.median(cov1) + 0.5* np.median(cov2) ) * 3.*10**10 
#    pdb.set_trace()
    Mw_eq = (2./3.)*np.log10(M0co* 10**7)-10.7
    Mw_cov = (2./3.)*np.log10(M0cosd* 10**7)-10.7
    
    print('Moment: '+str(M0co)+' +- '+str(M0cosd))
    print('Equivalent magnitude: '+str(Mw_eq)+' +- '+str(Mw_cov))
    return M0co,M0cosd,Mw_eq,Mw_cov

#def writeSCRMODfsp(fault, resname, resdir, srcmodpar):
#    '''
#    
#    '''
#    
#    filename = resdir+resname+'.srcmod.fsp'
#    with open(filename, 'w') as f:
#        f.write('% ------------------------- FINITE-SOURCE RUPTURE MODEL ---------------------------' + '\n')
#        f.write( '%' + '\n')
#        f.write( 'Evnt :'+srcmodpar['eqname'])
#        f.write( ' ('+srcmodpar['country']+') ')
#        f.write(srcmodpar['date']+' ['+srcmodpar['author']+'] \n')
#        
#        f.write( 'EventTAG: '+srcmodpar['tag'] + '\n')
#        
#        f.write( '% Loc : LAT = '+srcmodpar['LAT']+' ')
#        f.write( 'LON = '+srcmodpar['LON']+' ')
#        f.write( 'DEP = '+srcmodpar['DEP'] + '\n')
#        
#        f.write( '% Size : LEN = '+srcmodpar['LEN']+' km ' )
#        f.write( 'WID = '+srcmodpar['WID']+' km ')
#        f.write( 'Mw = '+srcmodpar['Mw']+' ')
#        f.write( 'Mo = '+srcmodpar['Mo']+' Nm\n')
#        
#        f.write( '% Mech : STRK = '+srcmodpar['STRK']+' ')
#        f.write( 'DIP = '+srcmodpar['DIP']+' ')
#        f.write( 'RAKE = '+srcmodpar['RAKE']+' ')
#        f.write( 'Htop = '+srcmodpar['Htop'])+' km\n')
#        
#        f.write( '% Rupt : HypX = '+srcmodpar['Rupt'])+' km ')
#        f.write( 'HypZ = '+srcmodpar['HypZ'])+' km ')
#        f.write( 'avTr = '+srcmodpar['avTr'])+' s ')
#        f.write( 'avVr = '+srcmodpar['avVr'])+' km/s\n')
#        
#        f.write( '% Invs : inDx = '+srcmodpar['inDx'])+' km ')
#        f.write( 'inDz = '+srcmodpar['inDz'])+' km ')
#        f.write( 'Fmin = '+srcmodpar['Fmin'])+' Hz ')
#        f.write( 'Fmax = '+srcmodpar['Fmax'])+' Hz\n')
#        
#        f.write( '% Invs : Nx = '+srcmodpar['Nx'])+' ')
#        f.write( 'Nz = '+srcmodpar['Nz'])+' \n')
#    return

def calcSynthwTrans(datasets,keys,filename,faults,trans,resdir,figdir, cd=None,ssh=False,**params):
    '''
    ssh: False, or path in ssh client to open result files
    if ssh is not False, please specify in **params:
    - sftp_client
    '''

    try:    
        if faults.__class__ is not list :
            lon0 = faults.lon0
            lat0 =  faults.lat0
        else:
            lon0 = faults[0].lon0
            lat0 = faults[0].lat0

        #---------------------------------------------------------------
        # get results from ALTAR

        if ssh is False:
            h5file =  h5py.File(resdir+filename+'.h5','r')
        else:
            sftp_client = params['sftp_client']
            sftp_client.get(ssh+'step_final.h5',resdir+filename+'.h5')
            h5file =  h5py.File(resdir+filename+'.h5','r')
            os.remove(resdir+filename+'.h5')
      

        sampss = []
        sampds = []
        samptrans = []
        for k in keys:
            if 'ss' in k or 'strike' in k:
                sampss.append(h5file['ParameterSets'][k][()])
            elif 'ds' in k or 'dip' in k:
                sampds.append(h5file['ParameterSets'][k][()])
            elif 'tr' in k or 'trans' in k:
                samptrans.append(h5file['ParameterSets'][k][()])
        
        stacks=[]
        if len(sampss) == len(sampds):
            for i in range(len(sampss)):
                stacks.append(np.hstack((sampss[i],sampds[i])))
            if len(stacks) > 1:
                samp = np.transpose( np.concatenate( stacks , axis = 1 ) )
            else:
                samp = np.transpose( stacks[0] )
        else:
            ss = np.concatenate([sampss[i] for i in range(len(sampss))], axis=1)
            ds = np.concatenate([sampds[i] for i in range(len(sampds))], axis=1)
            
            stacks.append( np.hstack((ss,ds)) )
            samp = np.transpose( stacks[0] )
        
        if faults.__class__ is not list :   
            slv = multiflt('multiflt', [faults]+[trans])
        else:
            slv = multiflt('multiflt', faults+[trans])
        slv.assembleGFs()
        

        if faults.__class__ is not list :
            std = [np.std(samp[i,:]) for i in range(np.shape(samp)[0]) ]
            sigma = np.array(std)
        else:
            std = [np.std(samp[i,:]) for i in range(np.shape(samp)[0]) ]
            sigma = np.array(std)

        moy = np.mean( samp, axis=1 )
        med = np.median( samp, axis=1 ) 
        # sigma = np.std( samp, axis=1 ) 
        
        trans_samp = np.mean(samptrans, axis=1)
        mpost = moy
        slv.mpost= np.hstack((mpost, trans_samp[0]))
        slv.writeMpost2File(outfile=resdir+filename+'_slip_csi.dat')
        slv.distributem(verbose=True)
        trans.distributem()
        trans.removePredictions(datasets)

        RMS=[]
        fi = open(resdir+filename+'_rms.dat', 'w')
        for data in datasets:
            if data.vertical == True:
                data.buildsynth(slv.faults, direction='sd')
            else:
                data.buildsynth(slv.faults, direction='sd',vertical=False)
            if data.dtype != 'opticorr':
                RMS.append(data.getRMS()[1])
                print('RMS '+data.name,data.getRMS()[1])
                fi.write("%s %.6f\n" % (data.name,data.getRMS()[1]))
        fi.close()
        print(RMS)
        rms_tot = np.sum(RMS)

    except Exception as err:
        print(type(err))
        print ("ERROR :",str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    if np.shape(faults)==() or np.shape(faults)[0]==1:
        return faults, sigma, samp
    else:
        return faults, sigma, samp



if __name__ == "__main__":
    
    print('Please define the files you want to use in a distinct python file')

