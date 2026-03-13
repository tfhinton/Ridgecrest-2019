#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:40:07 2022

@author: thea
"""

# ---- Import Python Libraries
import numpy as np
import sys
import os
import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import math
import seaborn as sns
import h5py
import pandas as pd
import netCDF4
import os
import subprocess
import scipy
import collections as col
import mpmath
import pdb
import scipy.stats as st
import scipy.interpolate as sciint
from sklearn.datasets import make_blobs
from scipy import interpolate
import pymap3d as pm
from PIL import ImageColor
import cartopy as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                        LatitudeLocator)
import cmocean.cm as cmo
import string
import rockhound as rh
from matplotlib.colors import LightSource
import pickle
import datetime

# Import CSI Libraries
import csi.fault3D as rectFault
import csi.TriangularPatches as triangleFault
import csi.TriangularTents as tentFault
import csi.gps as gr
import csi.multifaultsolve as multiflt
import csi.insar as ir
import csi.Fault as Fault
import csi.transformation as transformation

sub_zone_nm = 'south_america' # in from rockhound.slab2 import ZONES; print(ZONES)

# import my lib
altarfig = __import__('altar_fig')
altarsynth = __import__('altar_synth')
altarini = __import__('altar_ini')
fg = __import__('fault_geom')

## Figures config
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
plt.style.use('/home/thea/mycode/python/myfig.mplstyle')
myblue = '#244c77ff'
mycyan = '#3f7f93ff'
myred ='#c3553aff'
myorange ='#f4a40bff'


colorsco=[(250,250,250),(255,247,236),(254,232,200),(253,212,158),(253,187,132),(252,141,89),(239,101,72),(215,48,31),(179,0,0),(127,0,0)]
colorsco_rgba = [(y[0]/255., y[1]/255., y[2]/255.) for y in colorsco]
#sigma_grey_transp_hex = ['#ffffff00', '#d0d0d050', '#d0d0d091', '#d0d0d0c9', '#d0d0d0e1']
sigma_grey_transp_hex = ['#ffffff00', '#808080ff', '#808080ff', '#808080ff', '#808080ff']
sigma_grey_transp = [ImageColor.getcolor(i, "RGB") for i in sigma_grey_transp_hex]
sigma_grey_transp_rgba = [(y[0]/255., y[1]/255., y[2]/255.) for y in sigma_grey_transp]

above_hex = ['#840000', '#7c030c', '#740515', '#6c071c', '#630923', '#5b0a2a', '#520c30', '#480d37', '#3d0e3d']
above = [ImageColor.getcolor(i, "RGB") for i in above_hex]
above_rgba = [(y[0]/255., y[1]/255., y[2]/255.) for y in above]
colorsco_above_rgba = colorsco_rgba + above_rgba
colorsco_above = colorsco + above
        


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
def calcM0(fault):
    '''
    in dyne.m
    
    mu S D *nbr subfaults * 10**7 to convert to dyn.cm
    '''
    mu = 2.5*1.e10
    
    # convert area to m2 (slip already in m)
    if 'Tents' in str(type(fault)):
        area = np.array(fault.area_tent)*1.e6
    else:
        area = np.array(fault.area)*1.e6
    sumss = np.abs((fault.slip[:,0]*area).sum())
    sumds = np.abs((fault.slip[:,1]*area).sum())
    sumtot = sumss + sumds
    
    M0 = sumtot * mu * 1.e7
    Mw = (2./3.)*np.log10(M0)-10.7
    
    Mwss = (2./3.)*np.log10(sumss* mu * 1.e7)-10.7
    Mwds = (2./3.)*np.log10(sumds * mu * 1.e7)-10.7
    
    print('Equivalent magnitude: '+str(Mw))
    print('Mw SS: '+str(Mwss))
    print('Mw DS: '+str(Mwds))
    return M0, Mw

def readSlipPylith(pylithdir, fault):
    with h5py.File(pylithdir+'slip.h5', "r", driver="sec2") as data:
        p_slip_coord = data['geometry/vertices'][:]
        p_slip = data['vertex_fields/slip'][:][0]
    fault.patch = np.array(fault.patch)
    x0 = np.amin(fault.patch[:,:,0])
    ind = np.unravel_index(np.argmin(fault.patch[:,:,0], axis=None), fault.patch[:,:,0].shape)
    y0 = fault.patch[ind[0],ind[1],1]
    dis=[]
    dep=[]
    for i in range(np.shape(p_slip)[0]):
        x = p_slip_coord[i,0]/1000.
        y = p_slip_coord[i,1]/1000.
        d = np.sqrt((x-x0)**2 + (y-y0)**2)
        dis.append(d)
        dep.append(p_slip_coord[i,2]/1000.)
    sns.scatterplot(dis,dep,hue=p_slip[:,0])
    return 

def voronoiFnPoly2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
        
    from https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def closestNode(node, nodes, dist):
    dist2 = np.sum((nodes - node)**2, axis=1)
    return np.where(dist2 <= dist)[0]

#def getVoronoi(fault):
#    from scipy.spatial import Voronoi
#    import matplotlib.path as mpltPath
#    fault.tent = np.array(fault.tent)
#    x0 = np.amin(fault.tent[:,0])
#    ind = np.argmin(fault.tent[:,0], axis=None)
#    y0 = fault.tent[ind,1]
#    dis = []
#    dep = []
#    for i in range(np.shape(fault.tent)[0]):
#        x = fault.tent[i,0]
#        y = fault.tent[i,1]
##        d = np.sqrt( (y-y0)**2)
#        d = y
#        dis.append(d)
#        dep.append(-np.sqrt(fault.tent[i,2]**2 + (x-x0)**2))
#    
#    # Make voronoi and bound the cells
#    dis = np.vstack(dis)
#    dep = np.vstack(dep)
#    vertex = np.hstack((dis,dep))
#    vor = Voronoi(vertex)
#    regions, vertices = voronoiFnPoly2d(vor)
#    
#    rects = []
#    for i in range(len(regions)):
#        region = regions[i]
#        polygon = vertices[region]
#        rect = mpltPath.Path( polygon )
#        rects.append(rect)
#    return rects


def readFwdPylithGPS(pylithout, datadir, fault):
    with h5py.File(pylithout+'fwd_g'+str(1)+'_ss_points.h5', "r") as pts:
        coords = pts['geometry/vertices'][:]
    
    coordsll = fault.xy2ll(coords[:,0]/1.e3, coords[:,1]/1.e3)
    coordsll = np.array(coordsll).T
    disp = np.zeros((len(coordsll),6))
    fullgps = np.hstack((coordsll,disp))
    names = np.zeros((len(coordsll),1))
    fullgps = np.hstack((names,fullgps))
    np.savetxt(datadir+'gps_pylith.ll', fullgps)
    return

def readFwdPylith2Gf(pylithout, fault, geoData, coords_gps):
    '''
    XX_pyl means data indexed as output from pylith
    XX_gps means data indexed as in my data files
    
    - read pylith points coordinates
    - read GFs for each pylith point
    - for each pylith point, find closest GPS point
    - build Gassembled matrix for GPS points
    - distribute Gassembled for each dataset
    '''
    
    try:
        # read pylith points coordinates
        with h5py.File(pylithout+'fwd_g'+str(1)+'_ss_points.h5', "r") as pts:
            coords_pyl = pts['geometry/vertices'][:]/1.e3
        
        # read GFs for each pylith point
        GF_pyl = np.zeros((fault.N_slip*2, len(coords_pyl), 3))
        for i in range(fault.N_slip):
            try:
                with h5py.File(pylithout+'fwd_g'+str(i+1)+'_ss_points.h5', "r") as pts:
                    GF_pyl[i] = pts['vertex_fields/displacement'][0]
            except Exception as err:
                print(type(err))
                print("ERROR :", str(err))
                print("GF missing : SS ", str(i+1))
                GF_pyl[i] = np.zeros(np.shape(GF_pyl[i-1]))
            try:    
                with h5py.File(pylithout+'fwd_g'+str(i+1)+'_ds_points.h5', "r") as pts:
                    GF_pyl[i+fault.N_slip] = pts['vertex_fields/displacement'][0]
            except Exception as err:
                print(type(err))
                print("ERROR :", str(err))
                print("GF missing : DS ", str(i+1))
                GF_pyl[i+fault.N_slip] = np.zeros(np.shape(GF_pyl[i+fault.N_slip-1]))
        
        # for each pylith point, find closest GPS point
        from scipy import spatial
        tree = spatial.KDTree( list(map(tuple, coords_gps[:,0:2])) ) 
        
        GF_gps = np.zeros((fault.N_slip*2, len(coords_pyl), 3))
        for idx_pyl in range(len(coords_pyl)):
            idx_gps = tree.query([(coords_pyl[idx_pyl,0],coords_pyl[idx_pyl,1])])[1][0]   
            GF_gps[:,idx_gps,:] = GF_pyl[:,idx_pyl,:]
        
        # distribute Gassembled for each dataset    
        GF = {}
        for data in [geoData[0]]:   
            GF[data.name] = {}
            GF[data.name]['strikeslip'] = np.zeros((3*len(data.x),fault.N_slip))
            GF[data.name]['dipslip'] = np.zeros((3*len(data.x),fault.N_slip))
            
        k = 0
        for data in [geoData[0]]:   
            GF[data.name]['strikeslip'] = np.hstack((np.hstack( (GF_gps[:fault.N_slip,k:k+len(data.x),0],                        
                                                                 GF_gps[:fault.N_slip,k:k+len(data.x),1]) ),
                                                                 GF_gps[:fault.N_slip,k:k+len(data.x),2] )).T
            GF[data.name]['dipslip'] = np.hstack((np.hstack( (GF_gps[fault.N_slip:2*fault.N_slip,k:k+len(data.x),0],                        
                                                              GF_gps[fault.N_slip:2*fault.N_slip,k:k+len(data.x),1]) ),
                                                              GF_gps[fault.N_slip:2*fault.N_slip,k:k+len(data.x),2] )).T
            k += len(data.x)
        
        idx=[]
        for i in range(len(GF_pyl)):
            if np.amax(abs(GF_pyl[i]))==0:
                idx.append(i)
        
    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return GF, idx



def readGfPylith(pylithout, slp, fault):
    ## Read slip. If SS and DS, then 2 times more slip values than area!!
    ## IMP_coord: coordinates of fault vertices
    imp_coord = np.fromfile(pylithout+'output_info_vertices.dat', dtype=">f8")
    imp_coord = imp_coord.reshape((len(imp_coord)//3,3))
    imp = np.fromfile(pylithout+'output_slip.dat', dtype=">f8")
    imp = imp.reshape((len(imp)//np.shape(imp_coord)[0],np.shape(imp_coord)[0]))
    imp_inds = np.nonzero(np.abs(imp)>=1.0e-04)
    imp_coord = imp_coord[imp_inds[1]]
    imp = imp[np.abs(imp)>=1.0e-04]

    imp_area = np.fromfile(pylithout+'output_info_area.dat', dtype=">f8")
    imp_ampl = np.fromfile(pylithout+'output_info_impulse_amplitude.dat', dtype=">f8")
    imp_area = imp_area[imp_ampl != 0.0]
    imp_ampl = imp_ampl[imp_ampl != 0.0]
    
    ## p_gf : number of fault vertices x number of data x 3
    p_gf = np.fromfile(pylithout+'gfs_'+slp+'-cgps_sites_displacement.dat', dtype=">f8")
    p_gf_coords = np.fromfile(pylithout+'gfs_'+slp+'-cgps_sites_vertices.dat', dtype=">f8")
    p_gf_coords = p_gf_coords.reshape((len(p_gf_coords)//3,3))
    p_gf = p_gf.reshape( (len(imp_coord),len(p_gf_coords),3) )
    
    ## compute voronoi cells
    polys = getVoronoi(fault)
    
    ## find impulses who are within fault vertice Voronoi cell
    # need to transpose in 2D space first
    x0 = np.amin(fault.tent[:,0])
    dis = []
    dep = []
    for i in range(len(imp_coord)):
        x = imp_coord[i,0]
        y = imp_coord[i,1]
        dis.append(y)
        dep.append(-np.sqrt(imp_coord[i,2]**2 + (x-x0)**2))
    points = np.vstack((np.array(dis).T,np.array(dep).T)).T
    
#    pyl_gf = np.empty((len(fault.dassembled),fault.N_slip))
    pyl_gf = []
    for i in range(len(fault.tent)):
        inside = polys[i].contains_points(points/1.e3)
#        area = imp_area[inside[:len(inside)]]  # area inkm
        # GFs normalized by area
#        Gp = [p_gf[inside][i]/area[i] for i in range(len(p_gf[inside]))]
        Gp = [p_gf[inside][i] for i in range(len(p_gf[inside]))]
        Gfs = np.array(Gp).sum(axis=0)
        Gpss = np.hstack( ( np.hstack( (Gfs[:,0], Gfs[:,1]) ) , Gfs[:,2] ) )
        # multiply by area of tent (from km2 to m2!)
#        pyl_gf[:,i] = Gpss*fault.area_tent[i]*1.e6
        pyl_gf.append(Gfs)
        
    return pyl_gf
    
def Nodes2spatialDB(flt, slp, outdir, outname):
    
    hdr = """// -*- C++ -*- (syntax highlighting)
//
// This spatial database specifies the distribution of slip on the
// fault surface for the forward problem.
//
#SPATIAL.ascii 1
SimpleDB {
  num-values = 1
  value-names =  slip
  value-units =  m
  num-locs = 4
  data-dim = 1 // Data is specified along a line.
  space-dim = 2
  cs-data = cartesian {
    to-meters = 1.0e+3 // Specify coordinates in km for convenience.
    space-dim = 3
  } // cs-data
} // SimpleDB
// Columns are
// (1) x coordinate (km)
// (2) y coordinate (km)
// (3) z coordinate (km)
// (4) slip (m)"""
    with open(os.path.join(outdir, outname), 'w') as fout:
        fout.write('{}\n'.format(hdr))
        
        for tindex in range(len(flt.tent)):
            p = flt.tent[tindex]
            fout.write('{} {} {} {}\n'.format(p[0], p[1], p[2], slp))
    return

def Nodes2sDBGFs(flt, subidx, slpss, slpds, outdir, outname):
    
    hdr = """// -*- C++ -*- (syntax highlighting)
//
// This spatial database specifies the distribution of slip on the
// fault surface for the forward problem.
//
#SPATIAL.ascii 1
SimpleDB {
  num-values = 3
  value-names =  left-lateral-slip  reverse-slip  fault-opening
  value-units =  m  m  m
  num-locs = %d
  data-dim = 2 // Data is specified on a plane
  space-dim = 3
  cs-data = cartesian {
    to-meters = 1.0e+3 // Specify coordinates in km for convenience.
    space-dim = 3
  } // cs-data
} // SimpleDB
// Columns are
// (1) x coordinate (km)
// (2) y coordinate (km)
// (3) z coordinate (km)
// (4) left-lateral-slip (m)
// (5) reverse-slip (m)
// (6) fault-opening (m)"""  % len(flt.tent)
    with open(os.path.join(outdir, outname), 'w') as fout:
        fout.write('{}\n'.format(hdr))
        
        for tindex in range(len(flt.tent)):
            if tindex == subidx:
                p = flt.tent[tindex]
                fout.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], slpss, slpds, 0.))
            else:
                p = flt.tent[tindex]
                fout.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], 0., 0., 0.))
    return

def makeCorrNoise(data,nbrdata):
    
    white_noise = np.random.normal(0, 1, size=nbrdata) # 1 m noise
    
    ## Make correlated noise drawn from a multivariate gaussian distribution
    X, truth = make_blobs(n_samples=300, centers=3, 
                          cluster_std = [4,2,1], 
                          center_box = (-14,14),
                          random_state=42)
    xn = X[:, 0]
    yn = X[:, 1]
    deltaX = (max(xn) - min(xn))/10
    deltaY = (max(yn) - min(yn))/10
    xmin = min(xn) - deltaX
    xmax = max(xn) + deltaX
    ymin = min(yn) - deltaY
    ymax = max(yn) + deltaY
    xx, yy = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([xn, yn])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    def find_nearest(array, value):
        array = np.asarray(array)
        ids = (np.abs(array - value)).argmin()
        return ids
    idx = []
    idy = []
    for d in data:
        for i in range(len(d.x)):
            idx.append(find_nearest(np.unique(xx), d.x[i]))
            idy.append(find_nearest(np.unique(yy), d.y[i]))   
    corr = np.array([f[idx[i],idy[i]] for i in range(len(idx))])
    corr_noise = (corr / np.amax(corr)) # 1 m noise
    noise = 0.5*(white_noise + corr_noise)
    
    return noise

def polyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def in_box(towers, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0],
                                         towers[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= towers[:, 1],
                                         towers[:, 1] <= bounding_box[3]))

def voronoi(towers, bounding_box,eps = 1.):
    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)
    # Mirror points
    points_center = towers[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = scipy.spatial.Voronoi(points)
    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not(bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                       bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor

def getVoronoi(fault):
#    from scipy.spatial import Voronoi
#    import matplotlib.path as mpltPath
    fault.tent = np.array(fault.tent)
    x0 = np.amin(fault.tent[:,0])
    ind = np.argmin(fault.tent[:,0], axis=None)
    y0 = fault.tent[ind,1]
    dis = []
    dep = []
    for i in range(np.shape(fault.tent)[0]):
        x = fault.tent[i,0]
        y = fault.tent[i,1]
#        d = np.sqrt( (y-y0)**2)
        d = y
        dis.append(d)
        dep.append(-np.sqrt(fault.tent[i,2]**2 + (x-x0)**2))
    
    # Make voronoi and bound the cells
    dis = np.vstack(dis)
    dep = np.vstack(dep)
    vertex = np.hstack((dis,dep))
    boundingbox = [np.amin(dis)-0.5,np.amax(dis)+0.7,np.amin(dep)-0.5,np.amax(dep)+0.5]# [x_min, x_max, y_min, y_max]
    vor = voronoi(vertex, boundingbox)
    regions, vertices = vor.filtered_regions, vor.vertices

    rects = []
    for i in range(len(regions)):
        region = regions[i]
        polygon = vertices[region]
#        rect = mpltPath.Path( polygon )
        rects.append(polygon)
    return rects

def computeArea(fault):
    if fault.area is None:
        fault.computeArea()

    fault.area_tent = []

    areas = np.array(fault.area)
    # Loop over vertices
    for i in range(fault.numtent):
        vid = fault.tentid[i]

        # find the triangle neighbors for each vertex
        nbr_triangles = fault.adjacencyMap[vid]
        area = np.sum( (1/3) * areas[nbr_triangles] )
        fault.area_tent.append(area)
    return   
  
def computeAreaVoronoi(fault):
    
    vor = getVoronoi(fault)
    
    area = []
    for i in range(len(vor)):
        area.append(polyArea(vor[i][:,0], vor[i][:,1]))
    
    fault.area_tent = area
    return  


def plotGPSwSlipBivTent(name,fault, sigma, datalist, synths, xlim, ylim, slip=None, valmax=None, sigmamax=None, slipdir=None, savedir='./',epicenter=None, xystrides=[100, 100], vertbound=[-1.,1.], mask=''):
    '''
     
    '''
    try:    

        if slip in ('strikeslip','ss','strike-slip'):
            slp = np.abs(fault.slip[:,0].copy())
            sgm = sigma[0:len(sigma)//2]
        elif slip in ('dipslip','ds','dip-slip'):
            slp = np.abs(fault.slip[:,1].copy())
            sgm = sigma[len(sigma)//2:len(sigma)]
        elif slip in ('total','tot'):
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2)
            sigma = np.array(sigma)
            sgm = np.sqrt(sigma[0:len(sigma)//2]**2 + sigma[len(sigma)//2:len(sigma)]**2)
        elif slip=='sigma':
            slp = np.array(sigma[len(sigma)//2:len(sigma)])
            sgm = sigma[len(sigma)//2:len(sigma)]
        else:
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2)        
            sigma = np.array(sigma)
            sgm = np.sqrt(sigma[0:len(sigma)//2]**2 + sigma[len(sigma)//2:len(sigma)]**2)            
        if valmax is None:
            slipmax = np.amax(slp)
        else:
            slipmax = valmax
        if sigmamax is None:
            uncmax = 0.6*np.amax(sgm)
        else:
            uncmax = sigmamax
        
        ## colorbars
        cptslip = colors.LinearSegmentedColormap.from_list('cptslip',colorsco_above_rgba, N=256)
        cNorm = MidpointNormalize(vmin=0., vcenter=slipmax, vmax=slipmax+5.)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cptslip)
        scalarMap.set_array([])
        
        cptsig = colors.LinearSegmentedColormap.from_list('cptsig',sigma_grey_transp_rgba, N=256)
        cmap = cptsig
        mycmap = cmap(np.arange(cptsig.N))
        mycmap[:,-1] = np.hstack((np.zeros((cmap.N//4,)),np.linspace(0.,1.,cmap.N-cmap.N//4)))
        cptsig2 = colors.ListedColormap(mycmap)
        cNorm2  = colors.Normalize(vmin=0., vmax=uncmax)
        scalarMap2 = cmx.ScalarMappable(norm=cNorm2, cmap=cptsig)
        scalarMap2.set_array([])
        
        try:
            fault.tentll = np.array(fault.tentll)
        except:
            centers = np.array(fault.getcenters())
            llo, lla = fault.xy2ll(centers[:,0], centers[:,1])
            fault.tentll = np.concatenate((llo[:, None], lla[:, None], centers[:,2][:, None]), axis=1)

        # triangles
        vertices = fault.Vertices_ll.tolist()
        faces = fault.Faces
        zt = []
        dist = []
        for face in faces:
            verts = [vertices[f] for f in face]
            x = [v[0] for v in verts]
            y = [v[1] for v in verts]
            z = [v[2] for v in verts]
            zt.append(x); dist.append(y)
        
        fig = plt.figure(figsize=(6,6))
        proj = ccrs.crs.PlateCarree()
        ax =  plt.axes(projection=proj)
        ax.outline_patch.set_visible(False)
        ax.spines['left'].set_visible(True)  
        ax.spines['bottom'].set_visible(True) 
        
        ax.set_extent([xlim[0], # minimum lon
                        xlim[1], # max longitude
                        ylim[0], # min lat
                        ylim[1] # max lat
                        ])
        
        # Get topography
        grid = rh.fetch_etopo1(version="bedrock")
        subset = grid.sel(latitude=slice(min(ylim),max(ylim)), longitude=slice(xlim[0],xlim[1]))
        topo = subset.bedrock.values
        # subset.bedrock.plot.pcolormesh(cbar_kwargs=dict(pad=0.01, aspect=30), ax=ax, zorder=0)
        ls = LightSource(azdeg=225, altdeg=45)
        cmap = plt.cm.gist_earth
        rgb = ls.shade(topo, cmap=cmap, blend_mode='overlay',
                       vert_exag=.02)
        ax.pcolormesh(subset.bedrock.longitude.values,
                      subset.bedrock.latitude.values,
                      topo, zorder=0, transform=proj, cmap=cmo.diff,
                      vmin=-6500, vmax=13000.,rasterized = True)

        ax.pcolormesh(subset.bedrock.longitude.values,
                      subset.bedrock.latitude.values,
                      rgb[:,:,0], zorder=1, transform=proj, alpha=.25, cmap=cmo.diff,
                      linewidth=0, edgecolors=None,rasterized = True,
                      vmin=-0.1, vmax=2.5)
        
        ## Plot slip
        try:
            # if not hasattr(fault, 'plotSources'):
            #     print('--------------------------------------')
            #     print('Please precompute sources for plotting')
            Ids = fault.plotSources[0]
            X = fault.plotSources[1]
            Y = fault.plotSources[2]
            Z = fault.plotSources[3]
            llo, lla = fault.xy2ll(X,Y)
            lon2, lat2 = llo, lla
            Slip = fault._getSlipOnSubSources(Ids, X, Y, Z, slp)
            Sigma = fault._getSlipOnSubSources(Ids, X, Y, Z, sgm)
        except AttributeError:
            Z =  fault.tentll[:,2]
            llo = fault.tentll[:,0]
            lla = fault.tentll[:,1]
            lon2, lat2 = llo, lla
            Slip = slp
            Sigma = np.zeros(np.shape(slp))
        lat = np.linspace(np.nanmin(lat2), np.nanmax(lat2), 300)
        lon = np.linspace(np.nanmin(lon2), np.nanmax(lon2),300)      
        lon, lat = np.meshgrid(lon,lat)
        slip2 = sciint.griddata((lon2,lat2),Slip,(lon,lat),method='cubic')
        sigma2 = sciint.griddata((lon2,lat2),Sigma,(lon,lat),method='linear')
        
        cslip = colors.LinearSegmentedColormap.from_list('cslip',scalarMap.to_rgba(range(0,int(valmax)+5)), 256)
        
        # get outline of subfaults
        rects = []
        for i in range(len(dist)):
            vertex = np.vstack((np.array(zt[i]),dist[i]))
            rect = patches.Polygon( vertex.T )
            rects.append(rect)
        p = PatchCollection(rects, facecolors='None', edgecolor = 'white', lw=0.2,zorder=40)
        
        # # discard points outside of fault
        # if len(mask) > 0:
        #     maskin = np.loadtxt(mask)
        #     slip2 *= maskin
        #     sigma2 *= maskin
        # else:
        maskin = np.nan*np.ones(slip2.shape)
        tll = [(i,j) for i,j in zip(lon.flatten(),lat.flatten())]
        for rect in rects:
            cont = rect.contains_points(tll).reshape(slip2.shape)
            maskin[cont] = 1
        slip2 *= maskin
        sigma2 *= maskin
        
        # plot all
        if slip=='sigma':
            pcol = ax.pcolor(lon, lat, slip2, cmap=cmo.deep, vmin=0, vmax=slipmax,rasterized = True, zorder=40)
            ax.add_collection(p)
        else:
            ax.pcolor(lon, lat, slip2, cmap=cslip, vmin=0, vmax=slipmax,rasterized = True, zorder=40)
            ax.add_collection(p)
            ax.pcolor(lon, lat,sigma2, cmap=cptsig2, vmin=0, vmax=uncmax, edgecolors=None,rasterized = True, zorder=41)     

        if slipdir is not None:
            rake = np.arctan2(fault.slip[:,1], fault.slip[:,0])
            if fault.N_slip == len(fault.Vertices_ll):
                centers = fault.Vertices_ll
            else:    
                centers = np.array([fault.xy2ll(x,y) for x,y in zip(np.array(fault.getcenters())[:,0], np.array(fault.getcenters())[:,1])])
            for i in range(len(fault.slip)):
                if slp[i] > slipmax/4:
                    xc, yc, zc, width, length, strike, dip = fault.getpatchgeometry(i, center=True)
                    x = -(np.sin(strike)*np.cos(rake[i]) - np.cos(strike)*np.cos(dip)*np.sin(rake[i]))
                    y = -(np.cos(strike)*np.cos(rake[i]) + np.sin(strike)*np.cos(dip)*np.sin(rake[i]))
                    if 'Patches' in str(type(fault)):
                        x *= -1
                        y *= -1
                    ax.quiver(centers[i,0],
                              centers[i,1],
                              x,y,
                              units = 'width',
                               angles = 'xy',
                              width = 0.002,
            #                  scale = None, 
            #                  scale_units='inches',
                              scale = (1/slp[i])*5e1, 
                              scale_units = 'x', 
                              color='dimgrey',
                              zorder=41)                          
        
        # trench and shoreline                               
        ax.add_feature(ccrs.cartopy.feature.COASTLINE, linestyle='-', alpha=0.5, lw=0.6, edgecolor='gray', zorder=42)
        ax.add_feature(ccrs.cartopy.feature.BORDERS, linestyle='-', alpha=0.25, lw=0.5, edgecolor='gray', zorder=43)
        ax.add_feature(ccrs.cartopy.feature.LAND, color='#fafafa', zorder=5)

        # trench = np.array([ [ -76.006200, -45.658500 ], [ -75.821400, -44.849400 ], [ -75.782400, -44.050600 ], [ -75.631600, -43.648200 ], [ -75.558000, -42.972200 ], [ -75.401700, -42.284500 ], [ -75.396900, -41.825600 ], [ -75.392200, -41.366600 ], [ -75.188900, -40.862800 ], [ -75.194800, -40.358900 ], [ -75.127800, -39.813100 ], [ -75.149500, -39.564400 ], [ -74.988300, -39.068100 ], [ -74.835900, -38.542000 ], [ -74.654800, -38.137600 ], [ -74.595600, -37.563500 ], [ -74.441600, -36.724900 ], [ -74.261700, -36.165300 ], [ -73.897400, -35.577700 ], [ -73.675500, -34.887600 ], [ -73.245700, -34.290400 ], [ -72.937500, -33.836900 ], [ -72.861000, -33.358500 ], [ -72.889000, -32.741900 ], [ -72.762200, -32.318800 ], [ -72.678300, -31.531300 ], [ -72.618100, -30.965200 ], [ -72.599900, -30.190300 ], [ -72.554600, -29.535700 ], [ -72.433200, -28.901400 ], [ -72.169800, -28.277600 ], [ -71.957800, -27.754700 ], [ -71.847100, -27.247800 ], [ -71.752500, -26.652200 ], [ -71.628300, -25.839700 ], [ -71.524300, -25.092800 ], [ -71.465900, -24.262600 ], [ -71.389500, -23.522700 ], [ -71.347500, -22.758400 ], [ -71.306800, -21.965400 ] , [ -71.306800, -21.965400 ], [ -71.249500, -21.262200 ], [ -71.322800, -20.757700 ], [ -71.362300, -20.249100 ], [ -71.486500, -19.811200 ], [ -71.746500, -19.298100 ], [ -72.266400, -18.721500 ], [ -72.736600, -18.256700 ], [ -73.399700, -17.747000 ], [ -73.949300, -17.373500 ], [ -74.609500, -16.823000 ], [ -75.292300, -16.330200 ], [ -75.916300, -15.735500 ] ])
        # ax.plot(trench[:,0], trench[:,1], linestyle='-', alpha=0.5, lw=1, color='#003547', transform=proj , zorder=42) 
         
        # #epicenter
        plt.scatter(-73.641, -16.265, s=45, marker='*', c='black', lw=0, zorder=502)
        
        
        
        # slab
        def fmt(x):
            s = f"{x:.0f}"
            return rf"{s}"

        subzone = rh.fetch_slab2(sub_zone_nm)
        sub = subzone.sel(latitude=slice(min(ylim),max(ylim)))
        sub = sub.sel(longitude=slice(360.+xlim[0],360.+xlim[1]))
        data = sub.to_dataframe().dropna(how='all')
        coords = np.array(list(data.index.values))
        depths = [10., 20., 30., 50., 70., 90., 110]
        slab = -sub.depth.values*1e-3
        cs = plt.contour(-(360-sub.longitude.values), sub.latitude.values, 
                    slab, depths,
                    zorder=43,colors='#637b8c',
                    linestyles='--', linewidths=.5)
        # cs2 = plt.contour(-(360-sub.longitude.values), sub.latitude.values, 
        #             slab, depths,
        #             zorder=44,colors='#637b8c',
        #             linestyles='solid', linewidths=.5, alpha=.5)
        ax.clabel(cs, cs.levels,inline=True, fmt=fmt, fontsize='x-small')
        # for i in range(3):
        #     ax.clabel(cs, cs.levels[:-4],inline=True, fmt=fmt, fontsize='x-small')
        
        # surface defo
        for data in datalist:
            lond, latd = fault.xy2ll(data.x, data.y)
            plt.quiver(lond, latd, 
                       data.vel_enu[:,0], 
                       data.vel_enu[:,1],
    #                   units = 'width', width = 0.2,
        #                  scale = None, 
        #                  scale_units='inches',
    #                   scale = 2.5, 
                       scale_units = 'xy',
                       scale=1.e-0,
                       width = 0.003,
                       color='dimgrey',zorder=44)
            plt.quiver(lond,latd, 
                       synths[data.name][:,0], 
                       synths[data.name][:,1],
    #                   units = 'width', width = 0.2,
        #                  scale = None, 
        #                  scale_units='inches',
    #                   scale = 2.5, 
                       scale_units = 'xy',
                       scale=1.e-0,
                       width = 0.004,
                       color='#1a5665' , zorder=44)
            
            sc = plt.scatter(lond, latd, c=data.vel_enu[:,2], s=80, 
                             cmap='RdBu_r', vmin=vertbound[0], vmax=vertbound[1],
                             zorder = 44)
            sc = plt.scatter(lond, latd, c=synths[data.name][:,2], s=25,
                             lw=0.1, edgecolors='white', cmap='RdBu_r',
                             vmin=vertbound[0], vmax=vertbound[1],
                             zorder = 44)
        
       ## GPS colorbars and legend
        localeg = [xlim[1]-0.5, ylim[0]-.3]
        plt.quiver(localeg[0]+0.3, localeg[1]-.7,
                  [-.5], 
                   [0.],
#                   units = 'width', width = 0.2,
    #                  scale = None, 
    #                  scale_units='inches',
#                   scale = 2.5, 
                   scale_units = 'xy',
                   scale=1.e-0,
                   width = 0.004,
                   color='dimgrey', zorder=44)
        plt.text(localeg[0]-0.18, localeg[1]-.9,
                 '50 cm', color='dimgrey',weight='light',fontsize= 'small', zorder=44)
        plt.text(localeg[0]-0.33, localeg[1]-1.15,
                 'observations', color='dimgrey',weight='light',fontsize= 'small', zorder=44)
        plt.text(localeg[0]-0.33, localeg[1]-1.3,
                 'predictions', color='#1a5665',weight='light',fontsize='small', zorder=44)
        plt.scatter(localeg[0]-0.5, localeg[1]-1.22,
                    s=80, c='dimgrey', 
                    vmin=vertbound[0], vmax=vertbound[1],
                    zorder = 500)
        plt.scatter(localeg[0]-0.5, localeg[1]-1.22,
                    c='#62a9bc', s=25,
                    lw=0.5, edgecolors='white',
                    vmin=vertbound[0], vmax=vertbound[1],
                    zorder = 501)
        
        cbaxes = ax.inset_axes([0.94, 0.05, 0.02, 0.2], transform=ax.transAxes,zorder=501)      
        clb = plt.colorbar(sc, cax=cbaxes,orientation='vertical')    
        clb.ax.xaxis.set_ticks_position('bottom')
        clb.ax.yaxis.set_label_position("left")
        clb.ax.set_ylabel('Vertical disp. (m)')
        rect = patches.Rectangle((-70.17,-19.42), 0.64,1.2, linewidth=0, facecolor='#fafafa', zorder=500)
        ax.add_patch(rect)

        gl = ax.gridlines(draw_labels=True, crs=proj)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlines = False
        gl.ylines = False
#        gl.ylocator = LatitudeLocator()
        gl.xlocator = mticker.MaxNLocator(5)
        gl.ylocator = mticker.MaxNLocator(4)
        gl.xlabel_style = {'size': 'small', 'color': 'gray'}
        gl.ylabel_style = {'size': 'small', 'color': 'gray', 'rotation':90}
        
        
    ## bivariate colorbar
        if slip != 'sigma':
            cax = fig.add_axes([0.62, 0.7, 0.25, 0.05])
            xx, yy = np.mgrid[0:uncmax:30j,0:slipmax+5:100j]
            C_map = scalarMap.to_rgba(yy)
            cax.imshow(C_map)
            xx_plot = np.array(255*(xx-xx.min())/(xx.max()-xx.min()), dtype=np.int)
            C_map2  = cptsig2(xx_plot)
            cax.imshow(C_map2)
            cax.set_xlim((-0.,95)  )   
            cax.set_ylim((-0.,29)  )  
            # if uncmax < 2:
            cax.locator_params(axis='y', nbins=2)
            cax.yticks = [0., uncmax//2]
            y_label_list = ['0',"{:1.0f}".format(uncmax//2.)]            
            # else:
            #     cax.locator_params(axis='y', nbins=3)
            #     y_label_list = ['0',"{:1.0f}".format(uncmax/3.),"{:1.0f}".format(2*uncmax/3.)]
            cax.locator_params(axis='x', nbins=4)
            x_label_list = ['0',"{:1.0f}".format(slipmax/4.),"{:1.0f}".format(2*slipmax/4.),"{:1.0f}".format(3*slipmax/4.)]
            cax.set_xticklabels(x_label_list)
            cax.set_yticklabels(y_label_list)
            cax.yaxis.set_label_position("right")
            cax.yaxis.tick_right()
            cax.set_title('Slip (m)')
            cax.set_ylabel('σ (m)')
        else:
            cax = fig.add_axes([0.55, 0.8, 0.25, 0.02])
            plt.colorbar(pcol, cax=cax,orientation='horizontal')        
            cax.set_title('Posterior σ (m)', fontweight='normal',fontsize='small')

            
        plt.savefig(savedir+name+'.pdf', format='pdf',bbox_inches="tight")
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
 
def plotFault(name,fault, savedir='./', xystrides=[100, 100], vertbound=[-1.,1.]):
    '''
     
    '''
    try:    
        fault.tentll = np.array(fault.tentll)
        
        # triangles
        vertices = fault.Vertices_ll.tolist()
        faces = fault.Faces
        zt = []
        dist = []
        for face in faces:
            verts = [vertices[f] for f in face]
            x = [v[0] for v in verts]
            y = [v[1] for v in verts]
            z = [v[2] for v in verts]
            zt.append(x); dist.append(y)
        
        fig = plt.figure(figsize=(6,6))
        proj = ccrs.crs.PlateCarree()
        ax =  plt.axes(projection=proj)
        ax.outline_patch.set_visible(False)
        ax.spines['left'].set_visible(True)  
        ax.spines['bottom'].set_visible(True) 
        
        # ax.set_extent([xlim[0], # minimum lon
        #                 xlim[1], # max longitude
        #                 ylim[0], # min lat
        #                 ylim[1] # max lat
        #                 ])
        
        ## Plot slip
        from csi.EDKSmp import dropSourcesInPatches as Patches2Sources
        fault.sourceNumber = 10
        fault.plotSources = Patches2Sources(fault, verbose=False)
        if not hasattr(fault, 'plotSources'):
            print('--------------------------------------')
            print('Please precompute sources for plotting')
        Ids = fault.plotSources[0]
        X = fault.plotSources[1]
        Y = fault.plotSources[2]
        Z = fault.plotSources[3]
        
        llo, lla = fault.xy2ll(X,Y)
        lon2, lat2 = llo, lla
        Slip = fault._getSlipOnSubSources(Ids, X, Y, Z, fault.slip[:,1])
        lat = np.linspace(np.nanmin(lat2), np.nanmax(lat2), 300)
        lon = np.linspace(np.nanmin(lon2), np.nanmax(lon2),300)      
        lon, lat = np.meshgrid(lon,lat)
        slip22 = sciint.griddata((lon2,lat2),Slip,(lon,lat),method='linear')
        import scipy.ndimage as ndimage
        slip2 = ndimage.median_filter(slip22,size=(1,1))
        
        slp = ax.pcolor(lon, lat, slip2, cmap=cmo.matter_r, vmin=int(np.amin(fault.slip[:,1])), vmax=int(np.amax(fault.slip[:,1])),rasterized = True)
#        import pdb; pdb.set_trace()
        ## Plot triangles
        rects = []
        for i in range(len(dist)):
            vertex = np.vstack((np.array(zt[i]),dist[i]))
            rect = patches.Polygon( vertex.T )
            rects.append(rect)
        p = PatchCollection(rects, facecolors='None', edgecolor = 'white', lw=0.2)
        ax.add_collection(p)
        
        # trench and shoreline                               
        ax.add_feature(ccrs.cartopy.feature.COASTLINE, linestyle='-', alpha=0.5, lw=0.6, edgecolor='gray')
        ax.add_feature(ccrs.cartopy.feature.BORDERS, linestyle='-', alpha=0.25, lw=0.5, edgecolor='gray')
        ax.add_feature(ccrs.cartopy.feature.LAND, color='#fafafa')
#        df = pd.read_json('/home/thea/projet/PB2002_boundaries.json')
#        for i in range(len(df.features)):
#            if df.features[i]['properties']['Name'] == 'NZ\\SA':
#                trench = df.features[i]['geometry']['coordinates']
        # trench = np.array([ [ -76.006200, -45.658500 ], [ -75.821400, -44.849400 ], [ -75.782400, -44.050600 ], [ -75.631600, -43.648200 ], [ -75.558000, -42.972200 ], [ -75.401700, -42.284500 ], [ -75.396900, -41.825600 ], [ -75.392200, -41.366600 ], [ -75.188900, -40.862800 ], [ -75.194800, -40.358900 ], [ -75.127800, -39.813100 ], [ -75.149500, -39.564400 ], [ -74.988300, -39.068100 ], [ -74.835900, -38.542000 ], [ -74.654800, -38.137600 ], [ -74.595600, -37.563500 ], [ -74.441600, -36.724900 ], [ -74.261700, -36.165300 ], [ -73.897400, -35.577700 ], [ -73.675500, -34.887600 ], [ -73.245700, -34.290400 ], [ -72.937500, -33.836900 ], [ -72.861000, -33.358500 ], [ -72.889000, -32.741900 ], [ -72.762200, -32.318800 ], [ -72.678300, -31.531300 ], [ -72.618100, -30.965200 ], [ -72.599900, -30.190300 ], [ -72.554600, -29.535700 ], [ -72.433200, -28.901400 ], [ -72.169800, -28.277600 ], [ -71.957800, -27.754700 ], [ -71.847100, -27.247800 ], [ -71.752500, -26.652200 ], [ -71.628300, -25.839700 ], [ -71.524300, -25.092800 ], [ -71.465900, -24.262600 ], [ -71.389500, -23.522700 ], [ -71.347500, -22.758400 ], [ -71.306800, -21.965400 ] ])
        # ax.plot(trench[:,0], trench[:,1], linestyle='-', alpha=0.5, lw=1, color='#003547', transform=proj)  
         
        # #epicenter
        # plt.scatter(-72.898, -36.122, s=25, marker='*', c='black', lw=0, zorder=502)
                
        cbaxes = ax.inset_axes([0.55, 0.95, 0.4, 0.02], transform=ax.transAxes)      
        plt.colorbar(slp, cax=cbaxes, label='Slip (m)',orientation='horizontal')        
        
        gl = ax.gridlines(draw_labels=True, crs=proj)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlines = False
        gl.ylines = False
#        gl.ylocator = LatitudeLocator()
        gl.xlocator = mticker.MaxNLocator(5)
        gl.ylocator = mticker.MaxNLocator(4)
        gl.xlabel_style = {'size': 6, 'color': 'gray'}
        gl.ylabel_style = {'size': 6, 'color': 'gray'}
    
        plt.savefig(savedir+name+'_slipgps.pdf', format='pdf',bbox_inches="tight")
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
        
        
def drawbasemap(lonlim, latlim, figsize = (7,7)):
    fig = plt.figure(figsize=figsize)
    proj = ccrs.crs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.outline_patch.set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_extent([lonlim[0],  # minimum lon
                   lonlim[1],  # max longitude
                   latlim[0],  # min lat
                   latlim[1]  # max lat
                   ])
    ax.add_feature(ccrs.cartopy.feature.COASTLINE, linestyle='-', alpha=0.5, lw=0.6, edgecolor='gray')
    ax.add_feature(ccrs.cartopy.feature.BORDERS, linestyle='-', alpha=0.25, lw=0.5, edgecolor='gray')
    ax.add_feature(ccrs.cartopy.feature.LAND, color='#fafafa')

    gl = ax.gridlines(draw_labels=True, crs=proj)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    gl.xlocator = mticker.MaxNLocator(5)
    gl.ylocator = mticker.MaxNLocator(4)
    gl.xlabel_style = {'size':  'small'}
    gl.ylabel_style = {'size':  'small'}

    return fig, ax, proj

def drawbasemap_multiax(lonlim, latlim, nrows=1, ncols=2, figsize=(7,7)):
    
    proj = ccrs.crs.PlateCarree()
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,
                        subplot_kw={'projection': proj},
                        figsize=figsize,
                        sharex=True, sharey=True)
    for ax in axes.reshape(-1):
        ax.spines['geo'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_extent([lonlim[0],  # minimum lon
                       lonlim[1],  # max longitude
                       latlim[0],  # min lat
                       latlim[1]  # max lat
                       ])
        ax.add_feature(ccrs.cartopy.feature.COASTLINE, linestyle='-', alpha=0.5, lw=0.6, edgecolor='gray')
        ax.add_feature(ccrs.cartopy.feature.BORDERS, linestyle='-', alpha=0.25, lw=0.5, edgecolor='gray')
        ax.add_feature(ccrs.cartopy.feature.LAND, color='#fafafa')
        
        sbs = ax.get_subplotspec()
        if sbs.is_first_col():
            gl = ax.gridlines(draw_labels=True, crs=proj)
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlabels_bottom = False
            gl.xlines = False
            gl.ylines = False
            # gl.xlocator = mticker.MaxNLocator(5)
            gl.ylocator = mticker.MaxNLocator(4)
            gl.xlabel_style = {'size':  'small'}
            gl.ylabel_style = {'size':  'small'}
        elif sbs.is_last_row():
            gl = ax.gridlines(draw_labels=True, crs=proj)
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.ylabels_left = False
            gl.xlines = False
            gl.ylines = False
            gl.xlocator = mticker.MaxNLocator(4)
            gl.ylocator = mticker.MaxNLocator(4)
            gl.xlabel_style = {'size':  'small'}
            gl.ylabel_style = {'size':  'small'}
        elif sbs.is_first_col() and sbs.is_last_row():
            gl = ax.gridlines(draw_labels=True, crs=proj)
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlines = False
            gl.ylines = False
            gl.xlocator = mticker.MaxNLocator(4)
            gl.ylocator = mticker.MaxNLocator(4)
            gl.xlabel_style = {'size':  'small'}
            gl.ylabel_style = {'size':  'small'}

    fig.tight_layout()
        
    return fig, axes

def drawinsar(sar, ax, vmin=-10., vmax=10., data='data', label=''):
    '''
    data in data, res, synth
    '''
    if data in ('data', 'd', 'dat', 'Data'):
        values = sar.vel
        plt.text(0.9, 0.9, 'data', ha='right',
                 size='small', transform = ax.transAxes)
    elif data in ('synth', 's', 'synt', 'Synth'):
        values = sar.synth
        plt.text(0.9, 0.9, 'predictions', ha='right',
                 size='small', transform = ax.transAxes)
    elif data in ('res', 'resid', 'residuals', 'r'):
        values = sar.vel - sar.synth
        plt.text(0.9, 0.9, 'residuals', ha='right',
                 size='small', transform = ax.transAxes)
    
    if len(label) > 0:
        plt.text(0.05, 0.90, label, weight='bold',
                bbox=dict(boxstyle='square', facecolor='none', edgecolor='dimgrey', pad=.2),
                transform = ax.transAxes)

    cNorm  = colors.Normalize(vmin, vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmo.curl)
    
    rects = []
    for i in range(len(sar.corner)):
        pts = sar.corner[i] 
        tl = pts[0:1+1]
        br = pts[2:3+1]
        tr = [pts[2], pts[1]]
        bl = [pts[0], pts[3]]
        rect = patches.Polygon( [tl, tr, br, bl] )
        colorval = scalarMap.to_rgba(values[i]*1e2)  # convert to cm
        rect.set_color(colorval)
        # rect.set_edgecolor('white')
        # rect.set_linewidth(0.1)
        rects.append(rect)
    p = PatchCollection(rects, match_original=True)
    ax.add_collection(p)
    
    cbaxes = ax.inset_axes([0.1, 0.1, 0.35, 0.05], transform=ax.transAxes)
    clb = plt.colorbar(scalarMap, cax=cbaxes,orientation='horizontal')
    clb.ax.set_title('LOS disp. (cm)')
    # clb.set_ticks([0,vmax/2,vmax])  
   
    return

def savesynths(datasets, filename, resdir = './', cd=None):
    
    for data in datasets:
        if data.dtype == 'gps':
            if cd is not None:  ## if Cp
                data.err_enu = np.reshape(cd[0:len(data.x)*3],(len(data.x),3))
            data.write2file(filename+'_'+data.name+'_data.dat', outDir=resdir, data='data')
            memory = data.err_enu
            # data.err_enu=errpost[data.name]
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
                
                
    return

def plotSlip2D(name,fault,slip=None, valmax=None, slipdir=None, sigma=False, savedir='./', legend='Coseismic slip (cm)',colorbar=colorsco_above_rgba,savename='slip2d',epicenter=None,index=False):
    '''
    '''
    if sigma.__class__ is list or sigma.__class__ is np.ndarray:
        sigma = np.array(sigma)
        if slip in ('strikeslip','ss','strike-slip'):
            slp = sigma[0:len(sigma)//2]
        elif slip in ('dipslip','ds','dip-slip'):
            slp = sigma[len(sigma)//2:len(sigma)]
        elif slip in ('SNR','snr'):
            slp = fault.slip[:,0].copy() / sigma[0:len(sigma)//2]
        else:
            slp = np.sqrt(sigma[0:len(sigma)//2]**2 + sigma[len(sigma)//2:len(sigma)]**2)
    else:
        if slip in ('strikeslip','ss','strike-slip'):
            slp = fault.slip[:,0].copy()
        elif slip in ('dipslip','ds','dip-slip'):
            slp = np.abs(fault.slip[:,1].copy())
        elif slip in ['tensile']:
            slp = fault.slip[:,2].copy()
        elif slip in ('total','tot'):
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
#            for i in range(len(fault.slip[:,0])):
#                if fault.slip[:,1][i] <= 70:
#                    slp[i] = fault.slip[:,0][i]
        elif slip in ('strikeslip_mle','ss_mle','strike-slip_mle'):
            slp = fault.mle[:,0].copy()
        elif slip in ('dipslip_mle','ds_mle','dip-slip_mle'):
            slp = np.abs(fault.mle[:,1].copy())
        else:
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
    
        
    if colorbar == 'sigma':
#        cm = sns.cubehelix_palette(rot=-.32,light=0.99,dark=0.15,as_cmap=True)
        cm = cmo.tempo
        if valmax is not None:
            cNorm  = colors.Normalize(vmin=0., vmax=valmax)
        elif legend == 'Entropy':
            cNorm  = colors.Normalize(vmin=0.9, vmax=1)
        else:
            cNorm  = colors.Normalize(vmin=np.amin(slp), vmax=np.amax(slp))
    elif colorbar == 'slipres':
        cm = sns.cubehelix_palette(rot=-.4,light=0.98,dark=0.2,as_cmap=True)
    else:
        cm = colors.LinearSegmentedColormap.from_list('cptslip',colorbar, N=256)
        if valmax is not None:
            cNorm = MidpointNormalize(vmin=0., vcenter=valmax, vmax=valmax+5.)
        else:
            cNorm = MidpointNormalize(vmin=np.amin(slp), vcenter=np.amax(slp), vmax=np.amax(slp)+5.)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    scalarMap.set_array([])

    fault.patch = np.array(fault.patch)
    x0 = np.amin(fault.patch[:,:,0])
#    ind = np.unravel_index(np.argmin(fault.patch[:,:,0], axis=None), fault.patch[:,:,0].shape)
#    y0 = fault.patch[ind[0],ind[1],1]
    y0 = np.amin(fault.patch[:,:,1])
    dis = []
    dep = []
    for i in range(np.shape(fault.patch)[0]):
        x = fault.patch[i,:,0]
        y = fault.patch[i,:,1]
        d = np.sqrt((x-x0)**2)
        # d = np.sqrt( (y-y0)**2)
        dis.append(d)
#        dep.append(-fault.patch[i,:,2] )
        dep.append(-np.sqrt(fault.patch[i,:,2]**2 ))
    
    fig, ax = plt.subplots(1,figsize=(7,3))
    ax.set_xlabel('Distance along strike (km)')
    ax.set_ylabel('Distance along dip (km)')
    
    rects = []
    for i in range(len(dis)):
        dis[i] = np.vstack(dis[i])
        dep[i] = np.vstack(dep[i])
        vertex = np.hstack((dis[i],dep[i]))
        rect = patches.Polygon( vertex )
        if slip is None:
            rect.set_color('gray')
        else:
            colorval = scalarMap.to_rgba(slp[i])
            rect.set_color(colorval)
        rect.set_edgecolor('white')
        rect.set_linewidth(0.1)
        rects.append(rect)

    p = PatchCollection(rects, match_original=True)
    ax.add_collection(p)

    if index is True:
        if 'Tents' in str(type(fault)):
            centers= np.array(fault.tent)
            x = centers[:,0]
            y = centers[:,1]
            z = centers[:,2]
            for i in range(len(x)):
    #            d = np.sqrt((x[i]-x0)**2 + (y[i]-y0)**2)
                d = np.sqrt((y[i]-y0)**2)
                dep = -np.sqrt(z[i]**2 + (x[i]-x0)**2)
                plt.text(d,dep,str(i),color='k')
        else:
            centers= np.array(fault.getcenters())
            x = centers[:,0]
            y = centers[:,1]
            z = centers[:,2]
            for i in range(len(x)):
    #            d = np.sqrt((x[i]-x0)**2 + (y[i]-y0)**2)
                d = np.sqrt((y[i]-y0)**2)
                dep = -np.sqrt(z[i]**2 + (x[i]-x0)**2)
                plt.text(d,dep,str(int(fault.index_parameter[i,0])),color='k')

    if slipdir is not None:
        rake = np.loadtxt(slipdir, comments='>')
        xc = rake[:,0]
        yc = rake[:,1]
        cent = []
        dep3=[]
        for i in range(len(xc)):
#            cent.append( np.sqrt((yc[i]-y0)**2) )
            cent.append(yc[i]-y0 )
            dep3.append( -np.sqrt(rake[i,4]**2+(xc[i]-x0)**2) )
        cent = np.array(cent)
        dep3 = np.array(dep3)
#            import pdb
#            pdb.set_trace()
        cent= cent[slp>np.amax(slp)/5]
        dep3 = dep3[slp>np.amax(slp)/5]
        rake= rake[slp>np.amax(slp)/5]
        slp = slp[slp>np.amax(slp)/5]
        ax.quiver(cent,dep3,   
                  2.5*slp[:], 2.5*slp[:],
                  units = 'width',
                  angles = [rake[:,7]],
                  width = 0.002,
#                  scale = None, 
#                  scale_units='inches',
                  scale = 4.5**1.5, 
                  scale_units = 'x', 
                  color='dimgrey')
    
    if epicenter is not None:
        xe, ye = fault.ll2xy(epicenter[0], epicenter[1])
        de = np.sqrt((xe-x0)**2 + (ye-y0)**2)
        plt.scatter(de, epicenter[2], s=100, c='white', edgecolors='dimgrey', marker=(5, 1))
    
    plt.xlim(np.amin(dis),np.amax(dis))
#    plt.ylim(np.amin(dep),np.amax(dep))
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=3)
    
    xold = np.linspace(0,np.amax(np.abs(np.array(dep))),10)
    depths = [np.abs(fault.patch[i][0,2]) for i in range(len(fault.patch))]
#    import pdb; pdb.set_trace()
    xnew = np.linspace(0,max(depths),10)
    def forward(x):
        return -np.interp(np.abs(x), xold, xnew)
    def inverse(x):
        return -np.interp(np.abs(x), xnew, xold)
    ax2 = ax.secondary_yaxis(-0.1, functions=(forward, inverse))
    ax2.set_ylabel('Depth (km)')
    ax2.set_yticks([0,-10,-20,-40,-60,-80])
        

    ## bivariate colorbar
    cax = fig.add_axes([0.93, 0.23, 0.05, 0.5])
    xx, yy = np.mgrid[0:valmax+5:100j,0:1.:30j]
    C_map = scalarMap.to_rgba(xx)
    cax.imshow(C_map)
#    yy_plot = np.array(255*(yy-yy.min())/(yy.max()-yy.min()), dtype=np.int)
    cax.set_ylim((-0.,95)  )   
    cax.set_xlim((-0.,9)  )  
    cax.tick_params(axis='x',
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False)
    cax.locator_params(axis='y', nbins=4)
    if valmax > 2.:
        y_label_list = ['0',"{:1.0f}".format(valmax/4.),"{:1.0f}".format(2*valmax/4.),"{:1.0f}".format(4*valmax/4.)]
    else:
        y_label_list = ['0',"{:.1f}".format(valmax/4.),"{:.1f}".format(2*valmax/4.),"{:.1f}".format(4*valmax/4.)]        
#    cax.set_xticklabels([-30.,30.])
    cax.set_yticklabels(y_label_list)
    cax.set_ylabel('Slip (m)')
#    cax.set_xlabel('Uncertainty (m)')
    
    ## simple colorbar 
#    colo=plt.colorbar(scalarMap,orientation='vertical',fraction=0.0085,pad=0.02)
#    if valmax is not None:
#        colo.set_ticks([0,valmax/2,valmax])  
#    elif legend=='Entropy':
#        colo.set_ticks([0.9,0.95,1.])  
#    else:
#        colo.set_ticks([np.amin(slp),(np.amax(slp)-np.amin(slp))//2,np.amax(slp)])  
#    ax=colo.ax
#    ax.text(-0.9,0.5, legend,rotation='vertical',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    
    
    plt.savefig(savedir+name+'_'+fault.name+'_'+savename+'.pdf', format='pdf',bbox_inches="tight")
    plt.show()              
     
    return

def drawscalebar(ax, proj, length, location=(0.5, 0.05), linewidth=3,
              units='km', m_per_unit=1000):
    """

    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    proj is the projection the axes are in
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    units is the name of the unit
    m_per_unit is the number of meters in a unit
    """
    # find lat/lon center to find best UTM zone
    x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
    # Projection in metres
    utm = ccrs.UTM(utm_from_lon((x0+x1)/2))
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
    # Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * m_per_unit/2, sbcx + length * m_per_unit/2]
    # buffer for scalebar
    buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
    # Plot the scalebar with buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, path_effects=buffer)
    # buffer for text
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    # Plot the scalebar label
    t0 = ax.text(sbcx, sbcy, str(length) + ' ' + units, transform=utm,
        horizontalalignment='center', verticalalignment='bottom',
        path_effects=buffer, zorder=2)
    left = x0+(x1-x0)*0.05
    # Plot the N arrow
    t1 = ax.text(left, sbcy, u'\u25B2\nN', transform=utm,
        horizontalalignment='center', verticalalignment='bottom',
        path_effects=buffer, zorder=2)
    # Plot the scalebar without buffer, in case covered by text buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, zorder=3)

    return

def drawglobe(center, dark=False, save=True):
    ax = plt.axes(projection=ccrs.crs.Orthographic(center[0], center[1]))

    if dark is False:
        ax.add_feature(ccrs.feature.OCEAN, zorder=0, facecolor='white')
        ax.add_feature(ccrs.feature.LAND, zorder=0, edgecolor='#cdcdcd', facecolor='#cdcdcd')
        ax.set_global()
        ax.gridlines(color='#b9b9b9', linewidth=.4)
        # ax.outline_patch.set_visible(False)
        ax.background_patch.set_visible(False)
    else:
        plt.style.use('dark_background')
        ax.add_feature(ccrs.feature.OCEAN, zorder=0, facecolor='#131e3aff')
        ax.add_feature(ccrs.feature.LAND, zorder=0, edgecolor='#6f83b9ff', facecolor='#6f83b9ff')
        ax.set_global()
        ax.gridlines(color='#6f83b9ff', linewidth=.4)
        
    plt.scatter(center[0], center[1], s=45, marker='*', c='black', lw=0, zorder=502)
    plt.show()
    
    if save==True:
        if dark is False:
            plt.savefig('globe.pdf', transparent=True)
        else:
            plt.savefig('globe_dark.png', transparent=True)

    return ax


def plotGIFslip(name,fault,samp,  datalist, synths, xlim, ylim, valmax=None, sigmamax=None, slipdir=None, savedir='./',epicenter=None, xystrides=[100, 100], vertbound=[-1.,1.], mask=''):
    '''
     
    '''
    try:
        ## create dir
        from pathlib import Path
        Path(savedir).mkdir(parents=True, exist_ok=True)
        
        
        ## MAKE SLIP FAMILIES
        Np = samp.shape[0]//2
        perms = np.random.permutation(samp.shape[1])
        rdm = [perms[i:i + 50] for i in range(0, len(perms), 50)]
        
        fams_slip = []
        fams_sigma = []
        for i in range(20):
            fams_slip.append( np.mean(samp[:,rdm[i]], axis=1))
            fams_sigma.append(np.std(samp[Np:2*Np,rdm[i]], axis=1) )
        
        uncmax = sigmamax
        slipmax = valmax
        
        ## colorbars
        cptslip = colors.LinearSegmentedColormap.from_list('cptslip',colorsco_above_rgba, N=256)
        cNorm = MidpointNormalize(vmin=0., vcenter=slipmax, vmax=slipmax+5.)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cptslip)
        scalarMap.set_array([])
        
        cptsig = colors.LinearSegmentedColormap.from_list('cptsig',sigma_grey_transp_rgba, N=256)
        cmap = cptsig
        mycmap = cmap(np.arange(cptsig.N))
        mycmap[:,-1] = np.hstack((np.zeros((cmap.N//4,)),np.linspace(0.,1.,cmap.N-cmap.N//4)))
        cptsig2 = colors.ListedColormap(mycmap)
        cNorm2  = colors.Normalize(vmin=0., vmax=uncmax)
        scalarMap2 = cmx.ScalarMappable(norm=cNorm2, cmap=cptsig)
        scalarMap2.set_array([])
        
        try:
            fault.tentll = np.array(fault.tentll)
        except:
            centers = np.array(fault.getcenters())
            llo, lla = fault.xy2ll(centers[:,0], centers[:,1])
            fault.tentll = np.concatenate((llo[:, None], lla[:, None], centers[:,2][:, None]), axis=1)

        # triangles
        vertices = fault.Vertices_ll.tolist()
        faces = fault.Faces
        zt = []
        dist = []
        for face in faces:
            verts = [vertices[f] for f in face]
            x = [v[0] for v in verts]
            y = [v[1] for v in verts]
            z = [v[2] for v in verts]
            zt.append(x); dist.append(y)
        
        # Get topography
        grid = rh.fetch_etopo1(version="bedrock")
        subset = grid.sel(latitude=slice(min(ylim),max(ylim)), longitude=slice(xlim[0],xlim[1]))
        topo = subset.bedrock.values
        # subset.bedrock.plot.pcolormesh(cbar_kwargs=dict(pad=0.01, aspect=30), ax=ax, zorder=0)
        
        # Get slab
        def fmt(x):
            s = f"{x:.0f}"
            return rf"{s}"

        subzone = rh.fetch_slab2(sub_zone_nm)
        sub = subzone.sel(latitude=slice(min(ylim),max(ylim)))
        sub = sub.sel(longitude=slice(360.+xlim[0],360.+xlim[1]))
        data = sub.to_dataframe().dropna(how='all')
        coords = np.array(list(data.index.values))
        depths = [10., 20., 30., 50., 70., 90., 110]
        slab = -sub.depth.values*1e-3

        for rr in range(len(rdm)):    
            slp = np.sqrt(fams_slip[rr][0:Np]**2 + fams_slip[rr][Np:2*Np]**2)
            
            for i in range(len(slp)):
                if slp[i] > slipmax/4:
                    slp[i] += np.random.uniform(-2.,2.)

            sgm = fams_sigma[rr] *3.
            
            ## Get slip
            try:
                if not hasattr(fault, 'plotSources'):
                    print('--------------------------------------')
                    print('Please precompute sources for plotting')
                Ids = fault.plotSources[0]
                X = fault.plotSources[1]
                Y = fault.plotSources[2]
                Z = fault.plotSources[3]
                fault.slip[:,0] = fams_slip[rr][0:Np]
                fault.slip[:,1] = fams_slip[rr][Np:2*Np]
                llo, lla = fault.xy2ll(X,Y)
                lon2, lat2 = llo, lla
                Slip = fault._getSlipOnSubSources(Ids, X, Y, Z, slp)
                Sigma = fault._getSlipOnSubSources(Ids, X, Y, Z, sgm)
            except AttributeError:
                Z =  fault.tentll[:,2]
                llo = fault.tentll[:,0]
                lla = fault.tentll[:,1]
                lon2, lat2 = llo, lla
                Slip = slp
                Sigma = np.zeros(np.shape(slp))
            lat = np.linspace(np.nanmin(lat2), np.nanmax(lat2), 300)
            lon = np.linspace(np.nanmin(lon2), np.nanmax(lon2),300)      
            lon, lat = np.meshgrid(lon,lat)
            slip2 = sciint.griddata((lon2,lat2),Slip,(lon,lat),method='cubic')
            sigma2 = sciint.griddata((lon2,lat2),Sigma,(lon,lat),method='linear')
            
            cslip = colors.LinearSegmentedColormap.from_list('cslip',scalarMap.to_rgba(range(0,int(valmax)+5)), 256)
            
            # get outline of subfaults
            rects = []
            for i in range(len(dist)):
                vertex = np.vstack((np.array(zt[i]),dist[i]))
                rect = patches.Polygon( vertex.T )
                rects.append(rect)
            p = PatchCollection(rects, facecolors='None', edgecolor = 'white', lw=0.2,zorder=40)
            
            # # discard points outside of fault
            # if len(mask) > 0:
            #     maskin = np.loadtxt(mask)
            #     slip2 *= maskin
            #     sigma2 *= maskin
            # else:
            maskin = np.nan*np.ones(slip2.shape)
            tll = [(i,j) for i,j in zip(lon.flatten(),lat.flatten())]
            for rect in rects:
                cont = rect.contains_points(tll).reshape(slip2.shape)
                maskin[cont] = 1
            slip2 *= maskin
            sigma2 *= maskin
            
            
            ## Figure
            
            fig = plt.figure(figsize=(6,6))
            proj = ccrs.crs.PlateCarree()
            ax =  plt.axes(projection=proj)
            ax.outline_patch.set_visible(False)
            ax.spines['left'].set_visible(True)  
            ax.spines['bottom'].set_visible(True) 
            
            ax.set_extent([xlim[0], # minimum lon
                            xlim[1], # max longitude
                            ylim[0], # min lat
                            ylim[1] # max lat
                            ])
            
            ls = LightSource(azdeg=225, altdeg=45)
            cmap = plt.cm.gist_earth
            rgb = ls.shade(topo, cmap=cmap, blend_mode='overlay',
                           vert_exag=.02)
            ax.pcolormesh(subset.bedrock.longitude.values,
                          subset.bedrock.latitude.values,
                          topo, zorder=0, transform=proj, cmap=cmo.diff,
                          vmin=-6500, vmax=13000.,rasterized = True)
    
            ax.pcolormesh(subset.bedrock.longitude.values,
                          subset.bedrock.latitude.values,
                          rgb[:,:,0], zorder=1, transform=proj, alpha=.25, cmap=cmo.diff,
                          linewidth=0, edgecolors=None,rasterized = True,
                          vmin=-0.1, vmax=2.5)
            
    
            # plot Slip
            ax.pcolor(lon, lat, slip2, cmap=cslip, vmin=0, vmax=slipmax,rasterized = True, zorder=40)
            ax.add_collection(p)
            ax.pcolor(lon, lat,sigma2, cmap=cptsig2, vmin=0, vmax=uncmax, edgecolors=None,rasterized = True, zorder=41)     
    
            if slipdir is not None:
                rake = np.arctan2(fault.slip[:,1], fault.slip[:,0])
                if fault.N_slip == len(fault.Vertices_ll):
                    centers = fault.Vertices_ll
                else:
                    centers = np.array([fault.xy2ll(x,y) for x,y in zip(np.array(fault.getcenters())[:,0], np.array(fault.getcenters())[:,1])])
                for i in range(len(fault.slip)):
                    if slp[i] > slipmax/4:
                        xc, yc, zc, width, length, strike, dip = fault.getpatchgeometry(i, center=True)
                        x = -(np.sin(strike)*np.cos(rake[i]) - np.cos(strike)*np.cos(dip)*np.sin(rake[i]))
                        y = -(np.cos(strike)*np.cos(rake[i]) + np.sin(strike)*np.cos(dip)*np.sin(rake[i]))
                        if 'Patches' in str(type(fault)):
                            x *= -1
                            y *= -1
                        ax.quiver(centers[i,0],
                                  centers[i,1],
                                  x,y,
                                  units = 'width',
                                   angles = 'xy',
                                  width = 0.002,
                #                  scale = None, 
                #                  scale_units='inches',
                                  scale = (1/slp[i])*5e1, 
                                  scale_units = 'x', 
                                  color='dimgrey',
                                  zorder=41)                          
            
            # trench and shoreline                               
            ax.add_feature(ccrs.cartopy.feature.COASTLINE, linestyle='-', alpha=0.5, lw=0.6, edgecolor='gray', zorder=42)
            ax.add_feature(ccrs.cartopy.feature.BORDERS, linestyle='-', alpha=0.25, lw=0.5, edgecolor='gray', zorder=43)
            ax.add_feature(ccrs.cartopy.feature.LAND, color='#fafafa', zorder=5)
    
            # #epicenter
            plt.scatter(-73.641, -16.265, s=45, marker='*', c='black', lw=0, zorder=502)
            
            # slab
            cs = plt.contour(-(360-sub.longitude.values), sub.latitude.values, 
                        slab, depths,
                        zorder=43,colors='#637b8c',
                        linestyles='--', linewidths=.5)
            ax.clabel(cs, cs.levels,inline=True, fmt=fmt, fontsize='x-small')
            
            # surface defo
            for data in datalist:
                lond, latd = fault.xy2ll(data.x, data.y)
                plt.quiver(lond, latd, 
                           data.vel_enu[:,0], 
                           data.vel_enu[:,1],
        #                   units = 'width', width = 0.2,
            #                  scale = None, 
            #                  scale_units='inches',
        #                   scale = 2.5, 
                           scale_units = 'xy',
                           scale=1.e-0,
                           width = 0.003,
                           color='dimgrey',zorder=44)
                plt.quiver(lond,latd, 
                           synths[data.name][:,0], 
                           synths[data.name][:,1],
        #                   units = 'width', width = 0.2,
            #                  scale = None, 
            #                  scale_units='inches',
        #                   scale = 2.5, 
                           scale_units = 'xy',
                           scale=1.e-0,
                           width = 0.004,
                           color='#1a5665' , zorder=44)
                
                sc = plt.scatter(lond, latd, c=data.vel_enu[:,2], s=80, 
                                 cmap='RdBu_r', vmin=vertbound[0], vmax=vertbound[1],
                                 zorder = 44)
                sc = plt.scatter(lond, latd, c=synths[data.name][:,2], s=25,
                                 lw=0.1, edgecolors='white', cmap='RdBu_r',
                                 vmin=vertbound[0], vmax=vertbound[1],
                                 zorder = 44)
            
           ## GPS colorbars and legend
            localeg = [xlim[1]-0.5, ylim[0]-.3]
            plt.quiver(localeg[0]+0.3, localeg[1]-.7,
                      [-.5], 
                       [0.],
    #                   units = 'width', width = 0.2,
        #                  scale = None, 
        #                  scale_units='inches',
    #                   scale = 2.5, 
                       scale_units = 'xy',
                       scale=1.e-0,
                       width = 0.004,
                       color='dimgrey', zorder=44)
            plt.text(localeg[0]-0.18, localeg[1]-.9,
                     '50 cm', color='dimgrey',weight='light',fontsize= 'small', zorder=44)
            plt.text(localeg[0]-0.33, localeg[1]-1.15,
                     'observations', color='dimgrey',weight='light',fontsize= 'small', zorder=44)
            plt.text(localeg[0]-0.33, localeg[1]-1.3,
                     'predictions', color='#1a5665',weight='light',fontsize='small', zorder=44)
            plt.scatter(localeg[0]-0.5, localeg[1]-1.22,
                        s=80, c='dimgrey', 
                        vmin=vertbound[0], vmax=vertbound[1],
                        zorder = 500)
            plt.scatter(localeg[0]-0.5, localeg[1]-1.22,
                        c='#62a9bc', s=25,
                        lw=0.5, edgecolors='white',
                        vmin=vertbound[0], vmax=vertbound[1],
                        zorder = 501)
            
            cbaxes = ax.inset_axes([0.94, 0.05, 0.02, 0.2], transform=ax.transAxes,zorder=501)      
            clb = plt.colorbar(sc, cax=cbaxes,orientation='vertical')    
            clb.ax.xaxis.set_ticks_position('bottom')
            clb.ax.yaxis.set_label_position("left")
            clb.ax.set_ylabel('Vertical disp. (m)')
            rect = patches.Rectangle((-70.17,-19.42), 0.64,1.2, linewidth=0, facecolor='#fafafa', zorder=500)
            ax.add_patch(rect)
    
            gl = ax.gridlines(draw_labels=True, crs=proj)
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlines = False
            gl.ylines = False
    #        gl.ylocator = LatitudeLocator()
            gl.xlocator = mticker.MaxNLocator(5)
            gl.ylocator = mticker.MaxNLocator(4)
            gl.xlabel_style = {'size': 'small', 'color': 'gray'}
            gl.ylabel_style = {'size': 'small', 'color': 'gray', 'rotation':90}
            
            
        ## bivariate colorbar
            cax = fig.add_axes([0.62, 0.7, 0.25, 0.05])
            xx, yy = np.mgrid[0:uncmax:30j,0:slipmax+5:100j]
            C_map = scalarMap.to_rgba(yy)
            cax.imshow(C_map)
            xx_plot = np.array(255*(xx-xx.min())/(xx.max()-xx.min()), dtype=np.int)
            C_map2  = cptsig2(xx_plot)
            cax.imshow(C_map2)
            cax.set_xlim((-0.,95)  )   
            cax.set_ylim((-0.,29)  )  
            # if uncmax < 2:
            cax.locator_params(axis='y', nbins=2)
            cax.yticks = [0., uncmax/2]
            y_label_list = ['0',"{:1.1f}".format(uncmax/2.)]            
            # else:
            #     cax.locator_params(axis='y', nbins=3)
            #     y_label_list = ['0',"{:1.0f}".format(uncmax/3.),"{:1.0f}".format(2*uncmax/3.)]
            cax.locator_params(axis='x', nbins=4)
            x_label_list = ['0',"{:1.0f}".format(slipmax/4.),"{:1.0f}".format(2*slipmax/4.),"{:1.0f}".format(4*slipmax/4.)]
            cax.set_xticklabels(x_label_list)
            cax.set_yticklabels(y_label_list)
            cax.yaxis.set_label_position("right")
            cax.yaxis.tick_right()
            cax.set_title('Slip (m)')
            cax.set_ylabel('σ (m)')
    
                
            plt.savefig(savedir+name+'_'+str(rr)+'.png', dpi=300, bbox_inches="tight")
            plt.close()
            # plt.show()  

        ## MAKE GIF
        import imageio as io      
        images = []
        for rr in range(20):      
            images.append(io.imread(savedir+name+'_'+str(rr)+'.png'))
        kargs = { 'duration': .2 }
        io.mimsave(savedir+name+'_slip_io.gif', images, 'GIF', **kargs)
        
        script1 = '''
        #!/bin/bash
        cd {di}
        convert -delay 25 -loop 1 *.png slip.mp4        
        '''.format(di=savedir)
        subprocess.call(script1, shell=True)

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
def plotSlipBivariate(name,fault,sigma,slip=None, valmax=None, sigmamax=None, slipdir=None, savedir='./',epicenter=None):
    '''
     
    '''
    try:
        # Bivariate colors defined from base to top and from left to right
        bivcolors=[ (208, 208, 208),
                    (232, 232, 232),  (164, 128, 128),
                    (250, 250, 250), (237,204,187), (214, 137, 127), (149, 75, 75),
                    (254, 248, 241), (254, 227, 190), (253, 173, 119), (248, 130, 84), (233, 87, 61), (210, 41, 27), (173, 0, 0), (127, 0, 0)]
        bivcolors_rgba= [(x[0]/255,x[1]/255,x[2]/255) for x in bivcolors]
        
#        bivcolors=[ (208, 208, 208),
#                    (221, 221, 221),  (199, 173, 173),
#                    (232, 232, 232),  (228, 178, 152), (214, 137, 127), (174, 111, 111),
#                    (250, 250, 250), (253, 214, 163), (253, 173, 119), (248, 130, 84), (233, 87, 61), (210, 41, 27), (173, 0, 0), (127, 0, 0)]
                    
#        bivcolors=[ (208, 208, 208),
#                    (221, 221, 221),  (164, 128, 128),                    
#                    (232, 232, 232), (228, 203, 176), (214,137,127), (149, 75, 75),
#                    (250, 250, 250), (255, 243, 226), (253, 214, 163), (253, 173, 119), (250, 137, 87), (232,87,61), (194,20,13), (127, 0, 0)]
        
        cmap.create_cmap(bivcolors, 'biv')
        cmape = plt.cm.get_cmap('biv',15)
        cNorm  = colors.Normalize(0,15)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='biv')
        scalarMap.set_array([])
                
        if slip in ('strikeslip','ss','strike-slip'):
            slp = fault.slip[:,0].copy()
            sgm = sigma[0:len(sigma)//2]
        elif slip in ('dipslip','ds','dip-slip'):
            slp = fault.slip[:,1].copy()
            sgm = sigma[len(sigma)//2:len(sigma)]
        elif slip in ('total','tot'):
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
            sgm = np.sqrt(sigma[0:len(sigma)//2]**2 + sigma[len(sigma)//2:len(sigma)]**2)
        else:
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)        
            sgm = np.sqrt(sigma[0:len(sigma)//2]**2 + sigma[len(sigma)//2:len(sigma)]**2)
            
        if valmax is None:
            slipmax = np.amax(slp)
        else:
            slipmax = valmax
        if sigmamax is None:
            uncmax = np.amax(sgm)
        else:
            uncmax = sigmamax
        colval = ColValSup(slp,sgm,slipmax,uncmax)
        
        fault.patch = np.array(fault.patch)
        x0 = fault.patch[0,0,0]
        y0 = fault.patch[0,0,1]
        dis = []
        dep = []
        for i in range(np.shape(fault.patch)[0]):
            x = fault.patch[i,:,0]
            y = fault.patch[i,:,1]
            d = np.sqrt((x-x0)**2 + (y-y0)**2)
            dis.append(d)
            dep.append(-fault.patch[i,:,2] )
        
        fig, ax = plt.subplots(1,figsize=(8,4))
        ax.set_xlabel('\n Distance along strike (km)')
        ax.set_ylabel('\n Depth (km)')
        
        rects = []
        for i in range(len(dis)):
            dis[i] = np.vstack(dis[i])
            dep[i] = np.vstack(dep[i])
            vertex = np.hstack((dis[i],dep[i]))
            rect = patches.Polygon( vertex )
            if slip is None:
                rect.set_color('gray')
            else:
                colorval = scalarMap.to_rgba(colval[i])
                rect.set_color(colorval)
            rect.set_edgecolor('white')
            rect.set_linewidth(0.1)
            rects.append(rect)
        p = PatchCollection(rects, match_original=True)
        ax.add_collection(p)
        
        if slipdir is not None:
            rake = np.loadtxt(slipdir, comments='>')
            xc = rake[:,0]
            yc = rake[:,1]
            cent = []
            for i in range(len(xc)):
                d = np.sqrt((xc-x0)**2 + (yc-y0)**2)
                cent.append(d)
            cent = np.array(cent)
            ax.quiver(cent, -rake[:,4],
                      0.5*slp[:], 0.5*slp[:],
                      units = 'width',
                      angles = [-rake[:,7]],
                      width = 0.002,
    #                  scale = None, 
    #                  scale_units='inches',
                      scale = 10**2.1, 
                      scale_units = 'x', 
                      color='dimgrey')
        
        if epicenter is not None:
            xe, ye = fault.ll2xy(epicenter[0], epicenter[1])
            de = np.sqrt((xe-x0)**2 + (ye-y0)**2)
            plt.scatter(de, epicenter[2], s=250, c='white', edgecolors='dimgrey', marker=(5, 1))
            
        
        # plot colorscale 
        center = [np.amax(dep)-0.1*np.amax(dep),np.amax(dis)+0.05*np.amax(dis) ]
        L = np.abs(np.amax(dep))/3
        wdgs = []
        #1
        wdgs.append(patches.Wedge(center, L/4, -30+90, 30+90, width=None,ec='white',fc=bivcolors_rgba[0],lw=0.1))
        #2
        wdgs.append(patches.Wedge(center, 2*L/4, -30+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[2],lw=0.1))
        wdgs.append(patches.Wedge(center, 2*L/4, 0+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[1],lw=0.1))
        #3
        wdgs.append(patches.Wedge(center, 3*L/4, -30+90, -15+90, width=L/4,ec='white',fc=bivcolors_rgba[6],lw=0.1))
        wdgs.append(patches.Wedge(center, 3*L/4, -15+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[5],lw=0.1))
        wdgs.append(patches.Wedge(center, 3*L/4, 0+90, 15+90, width=L/4,ec='white',fc=bivcolors_rgba[4],lw=0.1))
        wdgs.append(patches.Wedge(center, 3*L/4, 15+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[3],lw=0.1))
        #4
        wdgs.append(patches.Wedge(center, 4*L/4, -30+90, -22.5+90, width=L/4,ec='white',fc=bivcolors_rgba[14],lw=0.1))
        wdgs.append(patches.Wedge(center, 4*L/4, -22.5+90, -15+90, width=L/4,ec='white',fc=bivcolors_rgba[13],lw=0.1))
        wdgs.append(patches.Wedge(center, 4*L/4, -15+90, -7.5+90, width=L/4,ec='white',fc=bivcolors_rgba[12],lw=0.1))
        wdgs.append(patches.Wedge(center, 4*L/4, -7.5+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[11],lw=0.1))
        wdgs.append(patches.Wedge(center, 4*L/4, 0+90, 7.5+90, width=L/4,ec='white',fc=bivcolors_rgba[10],lw=0.1))
        wdgs.append(patches.Wedge(center, 4*L/4, 7.5+90, 15+90, width=L/4,ec='white',fc=bivcolors_rgba[9],lw=0.1))
        wdgs.append(patches.Wedge(center, 4*L/4, 15+90, 22.5+90, width=L/4,ec='white',fc=bivcolors_rgba[8],lw=0.1))
        wdgs.append(patches.Wedge(center, 4*L/4, 22.5+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[7],lw=0.1))
        w = PatchCollection(wdgs, match_original=True)
        ax.add_collection(w)
        
        #legend
        coords = []
        for i in [14,12,10,8]:
            coords.append(wdgs[i].get_path().vertices[6])
        for i in [7,3,1,0]:
            coords.append(wdgs[i].get_path().vertices[0])
        labels = [str(i) for i in np.arange(0,valmax,valmax/4)]+[str(valmax)]+[str(sigmamax//2)]+[str(3*sigmamax//4)]+[str(sigmamax)]     
        rot = [30,15,0,-15,-30,60,60,60]
        vas = ['center']*5+['top']*3
        offset = [[0,0.05*L]]*5+[[0.05*L,0]]*3
        for i in range(len(labels)):
            ax.text(coords[i][0]+offset[i][0],coords[i][1]+offset[i][1],labels[i],rotation=rot[i],rotation_mode='anchor',ha='center',va=vas[i])
        
        # legend titles
        wdg = patches.Wedge(center, L, -30+90, 30+90, width=None)
        x = wdg.get_path().vertices[:,0]
        y = wdg.get_path().vertices[:,1]
        text = CurvedText(
            x = x[::-1][3:-1],
            y = y[::-1][3:-1]+0.13*L,
            text='Slip amplitude (m)',
            va = 'bottom',
            fontweight='regular',
            axes = ax)
        text2 = CurvedText(
            x = x[::-1][0:2]+0.13*L,
            y = y[::-1][0:2],
            text='Slip uncertainty (m)',
            va = 'top',
            fontweight='regular',
            axes = ax)
        
        plt.xlim(np.amin(dis),np.amax(dis)+0.1*np.amax(dis) )
        plt.ylim(np.amin(dep),np.amax(dep))
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.savefig(savedir+name+'_'+fault.name+'_slipbiv.pdf', format='pdf',bbox_inches="tight")
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    
    return

def plotSlipBivCont(name,fault,sigma, slip=None, valmax=None, sigmamax = None, slipdir=None, savedir='./', colorbar=colorsco,epicenter=None):
    '''
    '''
    if slip in ('strikeslip','ss','strike-slip'):
        slp = fault.slip[:,0].copy()
        sgm = sigma[0:len(sigma)//2]
    elif slip in ('dipslip','ds','dip-slip'):
        slp = np.abs(fault.slip[:,1].copy())
        sgm = sigma[len(sigma)//2:len(sigma)]
    elif slip in ('total','tot'):
        slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
        sigma = np.array(sigma)
        sgm = np.sqrt(sigma[0:len(sigma)//2]**2 + sigma[len(sigma)//2:len(sigma)]**2)
    else:
        slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)        
        sigma = np.array(sigma)
        sgm = np.sqrt(sigma[0:len(sigma)//2]**2 + sigma[len(sigma)//2:len(sigma)]**2)  
    if valmax is None:
        slipmax = np.amax(slp)
    else:
        slipmax = valmax
    if sigmamax is None:
        uncmax = np.amax(sgm)
    else:
        uncmax = sigmamax    
        
    cptslip = colors.LinearSegmentedColormap.from_list('cptslip',colorsco_above_rgba, N=256)
    if valmax is not None:
#        cNorm  = colors.Normalize(vmin=0., vmax=valmax)
        cNorm = MidpointNormalize(vmin=0., vcenter=valmax, vmax=valmax+5.)
    else:
#        cNorm  = colors.Normalize(vmin=np.amin(slp), vmax=np.amax(slp))
        cNorm = MidpointNormalize(vmin=np.amin(slp), vcenter=np.amax(slp), vmax=np.amax(slp)+5.)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cptslip)
    scalarMap.set_array([])
    
    cptsig = colors.LinearSegmentedColormap.from_list('cptsig',sigma_grey_transp_rgba, N=256)
    cmap = cptsig
    mycmap = cmap(np.arange(cptsig.N))
    mycmap[:,-1] = np.hstack((np.zeros((cmap.N//4,)),np.linspace(0.,1.,cmap.N-cmap.N//4)))
    cptsig2 = colors.ListedColormap(mycmap)
    cNorm2  = colors.Normalize(vmin=0., vmax=uncmax)
    scalarMap2 = cmx.ScalarMappable(norm=cNorm2, cmap=cptsig)
    scalarMap2.set_array([])

    fault.patch = np.array(fault.patch)
    x0 = np.amin(fault.patch[:,:,0])
#    ind = np.unravel_index(np.argmin(fault.patch[:,:,0], axis=None), fault.patch[:,:,0].shape)
#    y0 = fault.patch[ind[0],ind[1],1]
    y0 = np.amin(fault.patch[:,:,1])
    dis = []
    dep = []
    for i in range(np.shape(fault.patch)[0]):
        x = fault.patch[i,:,0]
        y = fault.patch[i,:,1]
#        d = np.sqrt((x-x0)**2 + (y-y0)**2)
        d = np.sqrt( (y-y0)**2)
        dis.append(d)
        dep.append(-np.sqrt(fault.patch[i,:,2]**2 + (x-x0)**2))
    
    fig, ax = plt.subplots(1,figsize=(9,3))
    ax.set_xlabel('\n Along-strike distance (km)')
    ax.set_ylabel('\n Along-dip distance (km)')
    
    rects = []
    for i in range(len(dis)):
        dis[i] = np.vstack(dis[i])
        dep[i] = np.vstack(dep[i])
        vertex = np.hstack((dis[i],dep[i]))
        rect = patches.Polygon( vertex )
        if slip is None:
            rect.set_color('gray')
        else:
            colorval = scalarMap.to_rgba(slp[i])
            rect.set_color(colorval)
        rect.set_edgecolor('white')
        rect.set_linewidth(0.1)
        rects.append(rect)

    p = PatchCollection(rects, match_original=True)
    ax.add_collection(p)
    
    ## discrete sigma
    rects_sig = []
    for i in range(len(dis)):
        dis[i] = np.vstack(dis[i])
        dep[i] = np.vstack(dep[i])
        vertex = np.hstack((dis[i],dep[i]))
        rect = patches.Polygon( vertex )
        colorval = scalarMap2.to_rgba(sgm[i], alpha=sgm[i]/uncmax)
        rect.set_color(colorval)
        rect.set_edgecolor('white')
        rect.set_linewidth(0.1)
        rects_sig.append(rect)

    p_sig = PatchCollection(rects_sig, match_original=True)
    ax.add_collection(p_sig)
       
    if slipdir is not None:
        rake = np.loadtxt(slipdir, comments='>')
        xc = rake[:,0]
        yc = rake[:,1]
        cent = []
        ad = []
        for i in range(len(xc)):
#            d = np.sqrt((xc-x0)**2 + (yc-y0)**2)
            d = np.sqrt((yc-y0)**2)
            cent.append(d)
            ad.append(-np.sqrt(rake[i,4]**2+(xc-x0)**2))
        cent = np.array(cent)
        ax.quiver(cent, ad,
                  0.5*slp[:], 0.5*slp[:],
                  units = 'width',
                  angles = [-rake[:,7]],
                  width = 0.002,
#                  scale = None, 
#                  scale_units='inches',
                  scale = 10**0.1, 
                  scale_units = 'x', 
                  color='dimgrey')
#        for i in range(len(lonc)):
#            ax.quiver(cent[i], -rake[i,2],
#                      slp[i]*cent[i], -slp[i]*rake[i,3],
#                      units = 'width',
#                      angles = [rake[i,5]],
#                      width = 0.002,
##                      scale = None, 
#                      scale = 10**6.1*(1/slp[i]), 
#                      scale_units = 'x', 
#                      color='dimgrey')
##                      lw=10**-6*slp[i//2]) 
##                      arrow_length_ratio = 0.15,zorder=1000)
    
    ## Continuous sigma
#    diss2 = np.linspace(np.nanmin(cent), np.nanmax(cent), 700)
#    z2 = np.linspace(np.nanmin(-rake[:,2]), np.nanmax(-rake[:,2]),700)     
#    diss2, z2 = np.meshgrid(diss2,z2)
#    slip22 = sciint.griddata((cent,-rake[:,2]),slp,(diss2,z2),method='linear')
#    sigma22 = sciint.griddata((cent,-rake[:,2]),sgm,(diss2,z2),method='linear')
#    import scipy.ndimage as ndimage
#    slip2 = ndimage.median_filter(slip22,size=(1,1))
#    sigma2 = ndimage.median_filter(sigma22,size=(1,1))
#    ax.pcolor(diss2, z2, sigma2, cmap=cptsig2, vmin=0, vmax=uncmax, edgecolors=None,rasterized = True)
    
    
#    import pdb
#    pdb.set_trace()
    if epicenter is not None:
        xe, ye = fault.ll2xy(epicenter[0], epicenter[1])
        de = np.sqrt((xe-x0)**2 + (ye-y0)**2)
        plt.scatter(de, epicenter[2], s=100, c='white', edgecolors='dimgrey', marker=(5, 1))
    
    plt.xlim(np.amin(dis),np.amax(dis))
    plt.ylim(np.amin(dep),np.amax(dep))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=3)
    
    # Plot depth axis
    xold = np.linspace(np.amin(-np.array(dep)),np.amax(-np.array(dep)),10)
    depths = [-fault.patch[i][0,2] for i in range(len(fault.patch))]
#    import pdb; pdb.set_trace()
    xnew = np.linspace(min(depths),max(depths),10)
    def forward(x):
        return -np.interp(np.abs(x), xold, xnew)
    def inverse(x):
        return -np.interp(np.abs(x), xnew, xold)
    ax2 = ax.secondary_yaxis(-0.1, functions=(forward, inverse))
    ax2.set_ylabel('Depth (km)')
    ax2.set_yticks([0,-10,-20,-40,-60])
        
    ## simple colorbar    
#    colo=plt.colorbar(scalarMap,orientation='vertical',fraction=0.0085,pad=0.05)
#    colo.set_ticks([0,slipmax/2,slipmax])  
#    ax=colo.ax
#    ax.text(-1.,0.5, 'Coseismic slip (m)',rotation='vertical',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    
    ## bivariate colorbar
    cax = fig.add_axes([0.9, 0.23, 0.2, 0.5])
    xx, yy = np.mgrid[0:slipmax+5:100j,0:uncmax:30j]
    C_map = scalarMap.to_rgba(xx)
    cax.imshow(C_map)
    yy_plot = np.array(255*(yy-yy.min())/(yy.max()-yy.min()), dtype=np.int)
    C_map2  = cptsig2(yy_plot)
    cax.imshow(C_map2)
    cax.set_ylim((-0.,95)  )   
    cax.set_xlim((-0.,29)  )  
    cax.locator_params(axis='x', nbins=3)
    cax.locator_params(axis='y', nbins=4)
    x_label_list = ['0',"{:2.1f}".format(uncmax/3.),"{:2.1f}".format(2*uncmax/3.)]
    y_label_list = ['0',"{:1.0f}".format(slipmax/4.),"{:1.0f}".format(2*slipmax/4.),"{:1.0f}".format(4*slipmax/4.)]
    cax.set_xticklabels(x_label_list)
    cax.set_yticklabels(y_label_list)
    cax.set_ylabel('Slip (m)')
    cax.set_xlabel('Uncertainty (m)')
    
#    plt.tight_layout()
    plt.savefig(savedir+name+'_'+fault.name+'_'+'slipbivcont'+'.png', format='png',bbox_inches="tight",dpi=300)
    plt.savefig(savedir+name+'_'+fault.name+'_'+'slipbivcont'+'.pdf', format='pdf',bbox_inches="tight")
    plt.show()              
     
    return

def writeSlipCenters(nstrike, ndip, length, width, slip, filename, savedir):
    
    fout = open(os.path.join(savedir,filename), 'w')
    
    Np=slip.shape[0]
    pstk = []
    pdip=[]
    # position of each subpatch (small) in dip and strike
    for i in range(nstrike*4+1):
        pstk.append( i*length/(nstrike*4) )
    for i in range(ndip*4+1):
        pdip.append( i*width/(ndip*4) )
        
    l=1
    d=0
    p=0
    i=0
    while l <= 2:
        s=0
        while s < len(pstk)-1:
            # I want the centers: pstk+ pstk/2
            x= pstk[s]+pstk[1]/2
            y= pdip[d]+pdip[1]/2
            fout.write('{} {} {} \n'.format(x, -y, slip[i]))
            i=i+1
            if p < nstrike*4*2:           
                p=p+1
            s=s+1
        d=d+1
        l=l+1
    while 3 <= l <= 5:
        s=0
        while s < len(pstk)-1:
            x= pstk[s]+pstk[1]/2
            y= pdip[d]+pdip[1]/2
            fout.write('{} {} {} \n'.format(x, -y, slip[i]))
            i=i+1
            if p < nstrike*3*2+nstrike*4*2:           
                p=p+1
            s=s+2
        d=d+2
        l=l+1
    while 6 <= l <= ndip-2+5:
        s=0
        while s < len(pstk)-1:
            x= pstk[s]+pstk[1]/2
            y= pdip[d]+pdip[1]/2
            fout.write('{} {} {} \n'.format(x, -y, slip[i]))
            i=i+1
            if p <= Np:
                p=p+1
            s=s+4
        d=d+4
        l=l+1
    
    fout.close()
    
    return

def plotGPScompa(name, data, synths, savedir='./', valmin=-1., valmax=1., shoreline=120., ylim=[-210.,210.],xlim=[-10.,500.]):
    '''
     
    '''
    try:
#        colors_topo_hex = ['#003547', '#286477', '#5595a9', '#8ac9de', '#f5f5f5', '#ebebeb', '#e1e1e1', '#d7d7d7', '#cdcdcd']
        colors_topo_hex = ['#3f7f93', '#5893a2', '#72a8b3', '#8ebcc4', '#acd0d6', '#ededed', '#e5e5e5', '#dddddd', '#d5d5d5', '#cdcdcd']                  
        colors_topo = [ImageColor.getcolor(i, "RGB") for i in colors_topo_hex]  
        colors_topo = [(y[0]/255., y[1]/255., y[2]/255.) for y in colors_topo]
        topocmap = colors.LinearSegmentedColormap.from_list('topo',colors_topo, N=256)
        cNorm = MidpointNormalize(vmin=-6., vcenter=0.1, vmax=3.)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=topocmap)
        scalarMap.set_array([])
        topocmap = colors.LinearSegmentedColormap.from_list('ctopo',scalarMap.to_rgba(range(-6,3)), 256)
                  
        fig, ax = plt.subplots(1,figsize=(3.7,7))
        ax.set_xlabel('Distance from trench (km)')
        ax.set_ylabel('Distance along strike (km)')
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        
        #background
        ax.add_patch(patches.Rectangle((-100, -300), 220, 600, color='#c4dade'))
#        ax.pcolor(X,Y,topo, cmap=topocmap)                                       
        ax.add_patch(patches.Rectangle((shoreline, -300), 850, 600, color='#f1f3f4'))             
        ax.plot([shoreline,shoreline],[-300.,300], lw=1, c='#cdcdcd')
        ax.plot([0,0],[-300.,300], lw=2, c='#003547')
                 
        # surface defo
        plt.quiver(data.x, data.y, 
                   data.vel_enu[:,0], 
                   data.vel_enu[:,1],
#                   units = 'width', width = 0.2,
    #                  scale = None, 
    #                  scale_units='inches',
#                   scale = 2.5, 
                   scale_units = 'xy',
                   scale=3.e-2,
                   width = 0.002,
                   color='dimgrey' , zorder=5)
        plt.quiver(data.x, data.y, 
                   synths[data.name][:,0], 
                   synths[data.name][:,1],
#                   units = 'width', width = 0.2,
    #                  scale = None, 
    #                  scale_units='inches',
#                   scale = 2.5, 
                   scale_units = 'xy',
                   scale=3.e-2,
                   width = 0.003,
                   color='#1a5665' , zorder=6)
        
        sc = plt.scatter(data.x, data.y, c=data.vel_enu[:,2], s=40, cmap=cmo.curl, vmin=valmin, vmax=valmax, zorder=6)
        sc = plt.scatter(data.x, data.y, c=synths[data.name][:,2], s=15, lw=0.1, edgecolors='white', cmap=cmo.curl, vmin=valmin, vmax=valmax, zorder=6)

        plt.quiver([xlim[0]+100.], [ylim[1]-20.],
                  [-2.], 
                   [0.],
#                   units = 'width', width = 0.2,
    #                  scale = None, 
    #                  scale_units='inches',
#                   scale = 2.5, 
                   scale_units = 'xy',
                   scale=3.e-2,
                   width = 0.003,
                   color='dimgrey' , zorder=5)
        plt.text(xlim[0]+60., ylim[1]-30., '2 m', color='dimgrey',weight='light',fontsize=7, zorder=5) 
        cbaxes = ax.inset_axes([0.55, 0.95, 0.4, 0.02], transform=ax.transAxes)      
        plt.colorbar(sc, cax=cbaxes, label='Vertical displacement (m)',orientation='horizontal')
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])
        plt.savefig(savedir+name+'_surfacedef.pdf', format='pdf',bbox_inches="tight")
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
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
