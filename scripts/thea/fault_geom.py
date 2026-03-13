# -*- coding: utf-8 -*-

'''
Created on Wed May 24 14:37:22 2016

@author: Théa Ragon

Creates fault and rectangle fault patches
'''

import numpy as np
from math import sqrt
from pdb import set_trace 

#np.set_printoptions(precision=6)

def d(lon1, lat1, lon2, lat2):
    """
    
    calculates distance (in meters) between two points (lon lat)

    """

    def to_radians(theta):
        return np.divide(np.dot(theta, np.pi), 180.0)

    def to_degrees(theta):
        return np.divide(np.dot(theta, 180.0), np.pi)

    delta_lat = to_radians(lat2 - lat1)
    delta_lon = to_radians(lon2 - lon1)
    lat1 = to_radians(lat1)
    lat2 = to_radians(lat2)
    
    a = np.sin(delta_lat/2) * np.sin(delta_lat/2) \
     + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2) * np.sin(delta_lon/2)
    c = 2 * np.arctan2( sqrt(a), sqrt(1-a) )
    d = c * 6371000

    return d


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

    #lon2 = (lon2 + 3 * np.pi) % (2 * np.pi) - np.pi
    
    return (to_degrees(lon2), to_degrees(lat2), z)
    
def fh_xy(x,y, z, strike, d):
    """
    Given a start point (lon lat), bearing (degrees), and distance (m),
    calculates the destination point (lon lat)
    """
    strike = strike+90
    def to_radians(theta):
        return theta * np.pi / 180.
    y2 = y + d*np.sin(to_radians(strike))
    x2 = x + d*np.cos(to_radians(strike))
    return (x2, y2, z)

def fd_xy(x,y, z, strike, dip, ddip):
    """
    
    Given a start point (lon lat z), dip of the fault (degrees),
    and distance along dip (m), calculates the destination point (lon lat z)

    """
    theta = strike + 90
    def to_radians(theta):
        return np.divide(np.dot(theta, np.pi), 180.0)
    z2 = z + ddip * np.sin( to_radians(dip) )
    D = np.cos( to_radians(dip) ) * ddip
    return fh_xy(x,y, z2, theta, D)

def fz(lon, lat, z, strike, dip, depth):
    """
    
    Given a start point (lon lat z), dip of the fault (degrees),
    and vertical depth (m), calculates the destination point (lon lat z)

    """
    z2 = depth
    theta = strike + 90
    
    def to_radians(theta):
        return np.divide(np.dot(theta, np.pi), 180.0)

    def to_degrees(theta):
        return np.divide(np.dot(theta, 180.0), np.pi)
    
    D = depth / np.tan( to_radians(dip) )
   
    return fh(lon, lat, z2, theta, D)
    

def fd(lon, lat, z, strike, dip, ddip):
    """
    
    Given a start point (lon lat z), dip of the fault (degrees),
    and distance along dip (m), calculates the destination point (lon lat z)

    """
   
    theta = strike + 90

    def to_radians(theta):
        return np.divide(np.dot(theta, np.pi), 180.0)

    def to_degrees(theta):
        return np.divide(np.dot(theta, 180.0), np.pi)
        
    z2 = z + ddip * np.sin( to_radians(dip) )
    D = np.cos( to_radians(dip) ) * ddip
   
    return fh(lon, lat, z2, theta, D)


def inter_segment(fault1,fault2,nd,ns1,ns2,filename):
    # Read fault file
    fgfile = open(fault1, 'r') 
    L = fgfile.readlines()
    i = 0
    f1=[]
    while i<len(L):
        pll=[]
        # get the values
        for j in range(1,4+1):
            lon, lat, z = L[i+j].split()
            # Pass as floating point
            lon = float(lon); lat = float(lat); z = float(z)
            pll.append([lon, lat, z])
        pll = np.array(pll)
        # Store these in the lists
        f1.append(pll)
        # increase i
        i += 5
        
    fgfile = open(fault2, 'r') 
    L = fgfile.readlines()
    i = 0
    f2=[]
    while i<len(L):
        pll=[]
        # get the values
        for j in range(1,4+1):
            lon, lat, z = L[i+j].split()
            # Pass as floating point
            lon = float(lon); lat = float(lat); z = float(z)
            pll.append([lon, lat, z])
        pll = np.array(pll)
        # Store these in the lists
        f2.append(pll)
        # increase i
        i += 5
    
    l = 0
    f = open(filename, 'w')
    while l < nd:
        f.write("%s\n" % (">       #") )
        f.write("%.6f %.6f %.6f\n" % (f1[ns1-1+l*ns1][1][0],f1[ns1-1+l*ns1][1][1],f1[ns1-1+l*ns1][1][2]) )
        f.write("%.6f %.6f %.6f\n" % (f2[0+l*ns2][0][0],f2[0+l*ns2][0][1],f2[0+l*ns2][0][2]) )
        f.write("%.6f %.6f %.6f\n" % (f2[0+l*ns2][3][0],f2[0+l*ns2][3][1],f2[0+l*ns2][3][2]) )
        f.write("%.6f %.6f %.6f\n" % (f1[ns1-1+l*ns1][2][0],f1[ns1-1+l*ns1][2][1],f1[ns1-1+l*ns1][2][2]) )
        l = l+1  
    f.close()    
    
    fi2 = open(filename+'_area', 'w')
    fi2.write("[")
    for l in range(0, nd):
        d1 = d(f1[ns1-1+l*ns1][1][0],f1[ns1-1+l*ns1][1][1], f2[0+l*ns2][0][0], f2[0+l*ns2][0][1])
        d2 = d(f2[0+l*ns2][3][0],f2[0+l*ns2][3][1], f2[0+l*ns2][0][0], f2[0+l*ns2][0][1]) 
        area =  d1*d2
        if l<nd-1:
            fi2.write("%.6f, " % (area) )
        else:
            fi2.write("%.6f" % (area) )
    fi2.write("]")  
    fi2.close()
        
    return

def patches_xy(x0,y0, z0, length, strike, dip, width, nstrike, ndip, filename):
    
    """
    Given a start point (lon lat z) (z = depth of top fault), a fault length
    (m), strike (degrees), dip (degrees) and width (m), and a number of 
    patches along strike and along dip
    
    Return a file of rectangle fault patches which can be read by CSI
    Return a file of patches area which can be read by ALTAR
    
    ex: 
    in python console:
    import fault_geom
    fault_geom.patches(13.41, 42.41, 0, 17000, 139, 49, 15000, 10, 10, 'fault_patches.rectangles') 
    fault_geom.patches(13.2508, 42.8575, 0, 26000, 166, 45, 16000, 26, 16, 'amatrice/fault_patches_166_45.rectangles')
    fault_geom.patches(13.386, 42.445, 0, 25000, 142, 54, 20000, 25, 20, 'laquila/fault_patches_best.rectangles')
    fault_geom.patches(86.0408, 27.0215, 0, 180000, 285, 7, 120000, 18, 12, 'gorkha/fault_patches_285_7.rectangles')
    fault_geom.patches(-72.77492, -33.0, -9600., -400000., 184.76, 180+24.5, 112000., 20, 5., 'illapel_fault_plan1.rectangles')
    """        

    depth = width * np.sin(dip)
    A = (x0,y0,z0)
    ddip = depth / (np.sin(dip) * ndip) # distance btw each point along dip
    dstrike = length /  nstrike # distance btw each point along strike
    N_pts = (ndip+1)*(nstrike+1) #points for patches
    F = np.full( shape=(N_pts,3),fill_value=0, dtype='Float64') 
    
    Y = A
    i = 0
    j = 0
    while i < N_pts:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        while j <= i+nstrike :
            Y = fh_xy(Y[0], Y[1], Y[2], strike, dstrike)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
        i = i + nstrike +1
        Y = X
        Y = fd_xy(Y[0], Y[1], Y[2], strike, dip, ddip)

    k = 0
    l = 0
    N = (nstrike + 1)*(ndip + 1) - nstrike - 2
    with open(filename, 'w') as f:
        while l < N:
            k = l
            while k < l+nstrike:
                f.write("%s\n" % (">       #") )
                f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]) )
                f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]) )
                f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+2,0], F[k+nstrike+2,1], F[k+nstrike+2,2]) )
                f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+1,0], F[k+nstrike+1,1], F[k+nstrike+1,2]) )
                k = k+1
            l = l+nstrike+1
    with open(filename+'_area', 'w')as f2:
        f2.write("[")
        for i in range(1, nstrike*ndip+1):
            if i<nstrike*ndip:
                area = length/nstrike * width/ndip
                f2.write("%.6f, " % (area) )
            else:
                area = length/nstrike * width/ndip
                f2.write("%.6f" % (area) )
                
        f2.write("]")  
    
    return
 
def patches(lon, lat, z, length, strike, dip, width, nstrike, ndip, filename):
    
    """
    Given a start point (lon lat z) (z = depth of top fault), a fault length
    (m), strike (degrees), dip (degrees) and width (m), and a number of 
    patches along strike and along dip
    
    Return a file of rectangle fault patches which can be read by CSI
    Return a file of patches area which can be read by ALTAR
    
    ex: 
    in python console:
    import fault_geom
    fault_geom.patches(13.41, 42.41, 0, 17000, 139, 49, 15000, 10, 10, 'fault_patches.rectangles') 
    fault_geom.patches(13.2508, 42.8575, 0, 26000, 166, 45, 16000, 26, 16, 'amatrice/fault_patches_166_45.rectangles')
    fault_geom.patches(13.386, 42.445, 0, 25000, 142, 54, 20000, 25, 20, 'laquila/fault_patches_best.rectangles')
    fault_geom.patches(86.0408, 27.0215, 0, 180000, 285, 7, 120000, 18, 12, 'gorkha/fault_patches_285_7.rectangles')
    fault_geom.patches(-72.77492, -33.0, -9600., -400000., 184.76, 180+24.5, 112000., 20, 5., 'illapel_fault_plan1.rectangles')
    """        
    
    ################ INPUT ####################
#        lon = float(sys.argv[1])
#        lat = float(sys.argv[2])
#        length = float(sys.argv[3])
#        strike = float(sys.argv[4])
#        dip = float(sys.argv[5])
#        width = float(sys.argv[6])
#        nstrike = float(sys.argv[7]) # number of rectangles along dip
#        ndip = float(sys.argv[8]) # number of rectangles along strike
#        
    depth = width * np.sin(dip)
    
    
    ################ FAULT BUILDING ####################
    A = (lon, lat, z)
    #B = fh(A[0], A[1], A[2], strike, length)
    #print A
    #print B
    #C = fz(A[0], A[1], A[2], strike, dip, depth)
    #D = fz(B[0], B[1], B[2], strike, dip, depth)
    #print C
    #print D
    
    
    ################ FAULT SAMPLING ####################
    ddip = depth / (np.sin(dip) * ndip) # distance btw each point along dip
    dstrike = length /  nstrike # distance btw each point along strike
    
    
    
    ######write all points -> NO PATCHES
    #Y = A
    #f = open('fault.txt', 'w')
    #while Y[2] <= depth:   
    #    X = Y
    #    f.write("%.6f %.6f %.6f\n" % (Y[0], Y[1], Y[2]) )
    #    
    #    while d(X[0],X[1],Y[0],Y[1]) < length - 1 :
    #        Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
    #        f.write("%.6f %.6f %.6f\n" % (Y[0], Y[1], Y[2]) )
    #        
    #    Y = X
    #    Y = fd(Y[0], Y[1], Y[2], strike, dip, Y[2] + ddip * np.cos(dip))
    #
    #f.close()
        

        
        
        
    ######write PATCHES
    N_pts = (ndip+1)*(nstrike+1) #points for patches
    F = np.full( shape=(N_pts,3),fill_value=0, dtype='Float64') 
    
    Y = A
    i = 0
    j = 0
    
    while i < N_pts:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i+nstrike :
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i + nstrike +1
        Y = X
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip)
    
#    print F
    
    k = 0
    l = 0
    N = (nstrike + 1)*(ndip + 1) - nstrike - 2
    f = open(filename, 'w')
    
    #CSI wants depth in kilometers -> z divided by 1000
    
    while l < N:
        k = l
        while k < l+nstrike:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+2,0], F[k+nstrike+2,1], F[k+nstrike+2,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+1,0], F[k+nstrike+1,1], F[k+nstrike+1,2]/1000) )
            k = k+1
        l = l+nstrike+1
        
    f.close()
    
    f2 = open(filename+'_area', 'w')
    f2.write("[")
    for i in range(1, nstrike*ndip+1):
        if i<nstrike*ndip:
            area = length/nstrike * width/ndip
            f2.write("%.6f, " % (area) )
        else:
            area = length/nstrike * width/ndip
            f2.write("%.6f" % (area) )
            
    f2.write("]")  
    f2.close()
    
    return
    
def patchesCurv(lon, lat, z, length, strike, width, dip_sur, dip_dep, depth, nstrike, ndip, filename,diprg=None):
    
    """
    Given a start point (lon lat z) (z = depth of top fault), a fault length
    (m), strike (degrees), dip at surface (degrees), dip at maximum depth (degrees),
    and a number of patches along strike and along dip
    
    Return a file of rectangle fault patches which can be read by CSI
    Return a file of patches area which can be read by ALTAR
    
    fault is curved with beginning dip of dip at surface value, termining dip of dip at max depth
    value
    
    ex: 
    in python console:
    import fault_geom
    fault_geom.patches_curv(-72.77492, -33.0, -9600., -400000., 184.76, 112000. , 180+7.3, 180+24.5, 40000., 20, 6., 'illapel_fault_curv1.rectangles')#,diprg=np.array([ 187.8, 190.66666667, 193.53333333, 196.4 ,199.26666667, 202.13333333, 205.])) 
    """
   
    ################ FAULT BUILDING ####################
    A = (lon, lat, z)
    
    ################ FAULT SAMPLING ####################
    ddip = width / ndip # distance btw each point along dip
    dstrike = length /  nstrike # distance btw each point along strike
            
    ###### calc PATCHES
    N_pts = int((ndip+1)*(nstrike+1)) #points for patches
    N_pts_2 = int((nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3 + N_pts - (nstrike + 1)*(2+1))
    F = np.empty( (N_pts_2,3) ) 
    
    Y = A
    i = 0
    j = 0
    if diprg is None:         
        dip = np.linspace(dip_sur,dip_dep,ndip)
    else:
        dip = diprg
    while i < N_pts:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i+nstrike :
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i + int(nstrike) +1
        m = i // nstrike -1
        if m >= len(dip):
            m = len(dip)-1
        Y = X
        Y = fd(Y[0], Y[1], Y[2], strike, dip[m], ddip)
        
#    set_trace()
    ###### write PATCHES
    k = 0
    l = 0
    N = int((nstrike + 1)*(ndip + 1) - nstrike - 2)
    f = open(filename, 'w')
    
    #CSI wants depth in kilometers -> z divided by 1000
    
    while l < N:
        k = l
        while k < l+nstrike:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+2,0], F[k+nstrike+2,1], F[k+nstrike+2,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+1,0], F[k+nstrike+1,1], F[k+nstrike+1,2]/1000) )
            k = k+1
        l = l+nstrike+1
        
    f.close()
    
    ###### write AREA
#    f2 = open(filename+'_area', 'w')
#    f2.write("[")
#    for i in range(1, nstrike*ndip+1):
#        if i<nstrike*ndip:
#            area = length/nstrike * width/ndip
#            f2.write("%.6f, " % (area) )
#        else:
#            area = length/nstrike * width/ndip
#            f2.write("%.6f" % (area) )
#            
#    f2.write("]")  
#    f2.close()
    
    return

def patchesCurv_xy(x0, y0, z, length, strike, width, dip_sur, dip_dep, depth, nstrike, ndip, filename,diprg=None):
    
    """
    Given a start point (x0 y0 z) (z = depth of top fault), a fault length
    (m), strike (degrees), dip at surface (degrees), dip at maximum depth (degrees),
    and a number of patches along strike and along dip
    
    Return a file of rectangle fault patches which can be read by CSI
    Return a file of patches area which can be read by ALTAR
    
    fault is curved with beginning dip of dip at surface value, termining dip of dip at max depth
    value
    
    ex: 
    in python console:
    import fault_geom
    fault_geom.patches_curv(-72.77492, -33.0, -9600., -400000., 184.76, 112000. , 180+7.3, 180+24.5, 40000., 20, 6., 'illapel_fault_curv1.rectangles')#,diprg=np.array([ 187.8, 190.66666667, 193.53333333, 196.4 ,199.26666667, 202.13333333, 205.])) 
    """
   
    ################ FAULT BUILDING ####################
    A = (x0, y0, z)
    
    ################ FAULT SAMPLING ####################
    ddip = width / ndip # distance btw each point along dip
    dstrike = length /  nstrike # distance btw each point along strike
            
    ###### calc PATCHES
    N_pts = int((ndip+1)*(nstrike+1)) #points for patches
    N_pts_2 = int((nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3 + N_pts - (nstrike + 1)*(2+1))
    F = np.empty( (N_pts_2,3) ) 
    
    Y = A
    i = 0
    j = 0
    if diprg is None:         
        dip = np.linspace(dip_sur,dip_dep,ndip)
    else:
        dip = diprg
    while i < N_pts:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i+nstrike :
            Y = fh_xy(Y[0], Y[1], Y[2], strike, dstrike)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i + int(nstrike) +1
        m = i // nstrike -1
        if m >= len(dip):
            m = len(dip)-1
        Y = X
        Y = fd_xy(Y[0], Y[1], Y[2], strike, dip[m], ddip)
        
#    set_trace()
    ###### write PATCHES
    k = 0
    l = 0
    N = int((nstrike + 1)*(ndip + 1) - nstrike - 2)
    f = open(filename, 'w')
    
    #CSI wants depth in kilometers -> z divided by 1000
    
    while l < N:
        k = l
        while k < l+nstrike:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]) )
            f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+2,0], F[k+nstrike+2,1], F[k+nstrike+2,2]) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+1,0], F[k+nstrike+1,1], F[k+nstrike+1,2]) )
            k = k+1
        l = l+nstrike+1
        
    f.close()
    return

def profileCurv_xy(x0, y0, z, width, dip_sur, dip_dep, depth, ndip, filename,diprg=None):
    
    """
    Given a start point (x0 y0 z) (z = depth of top fault), a fault length
    (m), strike (degrees), dip at surface (degrees), dip at maximum depth (degrees),
    and a number of patches along strike and along dip
    
    Return a file of rectangle fault patches which can be read by CSI
    Return a file of patches area which can be read by ALTAR
    
    fault is curved with beginning dip of dip at surface value, termining dip of dip at max depth
    """
   
    ################ FAULT BUILDING ####################
    
    ################ FAULT SAMPLING ####################
    ddip = width / ndip # distance btw each point along dip
            
    ###### calc PATCHES
    N_pts = int(ndip+1) #points for patches
    F = np.empty( (N_pts,3) ) 
    
    if diprg is None:         
        dip = np.linspace(dip_sur,dip_dep,ndip)
    else:
        dip = diprg
    F[0,:] = [x0, y0, z]
    
    for i in range(1,len(F)):
        F[i,:] = fd_xy(F[i-1,0], F[i-1,1], F[i-1,2], 0., dip[i-1], ddip)
        
    with open(filename, 'w') as f:
        for k in range(len(F)):
            f.write("%.6f %.6f %.6f\n" % (-F[k,0], F[k,1], -F[k,2]) )
    return

def patches_rough(lon, lat, z, length, strike, dip, width, nstrike, ndip, rough_rg, filename):
    
    """
    Given a start point (lon lat z) (z = depth of top fault), a fault length
    (m), strike (degrees), dip (degrees) and width (m), and a number of 
    patches along strike and along dip
    
    Return a file of rectangle fault patches which can be read by CSI
    Return a file of patches area which can be read by ALTAR
    
    ex: 
    in python console:
    import fault_geom
    fault_geom.patches(13.41, 42.41, 0, 17000, 139, 49, 15000, 10, 10, 'fault_patches.rectangles') 
    fault_geom.patches(13.2508, 42.8575, 0, 26000, 166, 45, 16000, 26, 16, 'amatrice/fault_patches_166_45.rectangles')
    fault_geom.patches(13.386, 42.445, 0, 25000, 142, 54, 20000, 25, 20, 'laquila/fault_patches_best.rectangles')
    """        
    
 
    depth = width * np.sin(dip)
    A = (lon, lat, z)
    
    ddip = depth / (np.sin(dip) * ndip) 
    dstrike = length /  nstrike


    N_pts = (ndip+1)*(nstrike+1) #points for patches
    F = np.full( shape=(N_pts,3),fill_value=0, dtype='Float64') 
    
    Y = A
    i = 0
    j = 0
    k = 0
    
    roughstk = np.random.normal(rough_rg[0],rough_rg[1], N_pts)
    roughdip = np.random.normal(rough_rg[0],rough_rg[1], N_pts)
    
    while i < N_pts:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        while j <= i+nstrike :
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            k = k+1  
        i = i + nstrike +1
        Y = X
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip)
        k = k+1
    
    
    i=0
    k=0
    R=np.full( shape=(ndip*nstrike*4,3),fill_value=0, dtype='Float64') 
    while i < ndip*nstrike*4:
        R[i,0]=F[k,0]
        R[i,1]=F[k,1]
        R[i,2]=F[k,2]
        R[i+1,0],R[i+1,1],R[i+1,2] = fh(R[i,0], R[i,1], R[i,2], strike+roughstk[k], dstrike)
        R[i+2,0],R[i+2,1],R[i+2,2] = fd(R[i,0], R[i,1], R[i,2], strike+roughstk[k], dip+roughdip[k], ddip)
        R[i+3,0],R[i+3,1],R[i+3,2] =fh(R[i+2,0], R[i+2,1], R[i+2,2], strike+roughstk[k], dstrike)
        if k%nstrike==0 and k !=0:
            k = k+2
            i=i+4
        else:
            i=i+4
            k=k+1
    
    f = open(filename, 'w')
    i=0
    while i < ndip*nstrike*4:
        f.write("%s\n" % (">       #") )
        f.write("%.6f %.6f %.6f\n" % (R[i,0], R[i,1], R[i,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (R[i+1,0],R[i+1,1],R[i+1,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (R[i+3,0],R[i+3,1],R[i+3,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (R[i+2,0],R[i+2,1],R[i+2,2]/1000) )
        i = i+4
    f.close()
    
    f2 = open(filename+'_area', 'w')
    f2.write("[")
    for i in range(1, nstrike*ndip+1):
        if i<nstrike*ndip:
            area = length/nstrike * width/ndip
            f2.write("%.6f, " % (area) )
        else:
            area = length/nstrike * width/ndip
            f2.write("%.6f" % (area) )
            
    f2.write("]")  
    f2.close()
    
    return
    
def patchesopti2(lon, lat, z, length, strike, dip, width, nstrike, ndip, filename):
    
    """
    Given a start point (lon lat z) (z = depth of top fault), a fault length
    (m), strike (degrees), dip (degrees) and width (m), and a number of 
    patches along strike and along dip
    
    Return a file of rectangle fault patches which can be read by CSI with optimized fault resolution:
    
    3 first patches lines: patches are divided by 2

    
    ex: 
    in python console:
    import fault_geom
    
    AMATRICE
    fault_geom.patchesopti(13.233, 42.8365, 0, 25000, 160, 40, 12000, 6, 4, 'amatrice/fault_patches_opti_best.rectangles')
    """        
    
    ################ INPUT ####################
#        lon = float(sys.argv[1])
#        lat = float(sys.argv[2])
#        length = float(sys.argv[3])
#        strike = float(sys.argv[4])
#        dip = float(sys.argv[5])
#        width = float(sys.argv[6])
#        nstrike = float(sys.argv[7]) # number of rectangles along dip
#        ndip = float(sys.argv[8]) # number of rectangles along strike
#        
    depth = width * np.sin(dip)
    
    
    ################ FAULT BUILDING ####################
    A = (lon, lat, z)
    #B = fh(A[0], A[1], A[2], strike, length)
    #print A
    #print B
    #C = fz(A[0], A[1], A[2], strike, dip, depth)
    #D = fz(B[0], B[1], B[2], strike, dip, depth)
    #print C
    #print D
    
    
    ################ FAULT SAMPLING ####################
    ddip = depth / (np.sin(dip) * ndip) # distance btw each point along dip
    dstrike = length /  nstrike # distance btw each point along strike
    
    
    
    ######write all points -> NO PATCHES
    #Y = A
    #f = open('fault.txt', 'w')
    #while Y[2] <= depth:   
    #    X = Y
    #    f.write("%.6f %.6f %.6f\n" % (Y[0], Y[1], Y[2]) )
    #    
    #    while d(X[0],X[1],Y[0],Y[1]) < length - 1 :
    #        Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
    #        f.write("%.6f %.6f %.6f\n" % (Y[0], Y[1], Y[2]) )
    #        
    #    Y = X
    #    Y = fd(Y[0], Y[1], Y[2], strike, dip, Y[2] + ddip * np.cos(dip))
    #
    #f.close()
        

        
        
        
    ######write PATCHES
    N_pts = (ndip+1)*(nstrike+1) #points for patches
    N_pts_2 = (nstrike+1+nstrike*(2-1))*7 + N_pts - (nstrike + 1)*(3+1)
    F = np.full( shape=(N_pts_2,3),fill_value=0, dtype='Float64') 
    
    Y = A
    i = 0
    j = 0
    
    
    while i < (nstrike+1+nstrike*(2-1))*7 -1:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i+nstrike+1+nstrike*(2-1) -1 :
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike/2)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i +nstrike+1+nstrike*(2-1)
        Y = X
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/2)
    
    i = (nstrike+1+nstrike*(2-1))*7 
    Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/2)
    
    while i < (nstrike+1+nstrike*(2-1))*7 + N_pts - (nstrike + 1)*(3+1):   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i+nstrike :
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i + nstrike +1
        Y = X
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip)

    
    k = 0
    l = 0
    N = N_pts_2 - nstrike - 2
    f = open(filename, 'w')
    N2 = nstrike+1+nstrike*(2-1)
    
    #CSI wants depth in kilometers -> z divided by 1000
    
    while l < N2*6 -1:
        k = l
        while k < l+N2-1:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000.) )
            f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]/1000.) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2+1,0], F[k+N2+1,1], F[k+N2+1,2]/1000.) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2,0], F[k+N2,1], F[k+N2,2]/1000.) )
            k = k+1
        l = l+N2
    
    l = N2*6
    while l < N2*7 -1:
        k = l
        n = 0
        while k < l+N2-2:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000.) )
            f.write("%.6f %.6f %.6f\n" % (F[k+2,0], F[k+2,1], F[k+2,2]/1000.) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2-n+1,0], F[k+N2-n+1,1], F[k+N2-n+1,2]/1000.) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2-n,0], F[k+N2-n,1], F[k+N2-n,2]/1000.) )
            k = k+2
            n = n+1
        l = l+N2  
    
    while l < N:
        k = l
        while k < l+nstrike:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000.) )
            f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]/1000.) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+2,0], F[k+nstrike+2,1], F[k+nstrike+2,2]/1000.) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+1,0], F[k+nstrike+1,1], F[k+nstrike+1,2]/1000.) )
            k = k+1
        l = l+nstrike+1
        
    f.close()
        
    f2 = open(filename+'_area', 'w')
    f2.write("[")
    Nbr = ndip*nstrike + 2*6*nstrike - 3*nstrike # Nbr of patches
    for i in range(1,Nbr+1):
        if i <= N2*6:
            area = (length/(2*nstrike)) * (width/(2*ndip))
            f2.write("%.6f, " % (area) )
        elif N2*6 +1 <= i <= Nbr-1 :
            area = length/nstrike * width/ndip
            f2.write("%.6f, " % (area) )
        else:
            area = length/nstrike * width/ndip
            f2.write("%.6f" % (area) )
    f2.write("]")
    f2.close()
    
    return
    
def patchesopti(lon, lat, z, length, strike, dip, width, nstrike, ndip, filename):
    
    """
    Given a start point (lon lat z) (z = depth of top fault), a fault length
    (m), strike (degrees), dip (degrees) and width (m), and a number of 
    patches along strike and along dip
    
    Return a file of rectangle fault patches which can be read by CSI with optimized fault resolution:
    
    two first patches lines: patches are divided by 4
    first line of resulting patches: patches are divided by 4
    
    ex: 
    in python console:
    import fault_geom
    fault_geom.patchesopti(13.41, 42.41, 0, 17000, 139, 49, 15000, 4, 4, 'fault_patches_opti.rectangles') 
    faille cheloni14:    
    fault_geom.patchesopti(13.386, 42.447, 0, 25000, 139, 49, 20000, 6, 5, 'fault_patches_opti.rectangles')
    trace surface faille cheloni 14
    13.4173 42.4214
    
    best fault moy rms:
    fault_geom.patchesopti(13.386, 42.445, 0, 25000, 142, 54, 20000, 6, 5, 'laquila/fault_patches_opti_best.rectangles') 
    
    AMATRICE
    fault_geom.patchesopti(13.2301, 42.8372, 0, 25000, 157, 45, 20000, 6, 5, 'amatrice/fault_patches_opti.rectangles')
    fault_geom.patchesopti(13.2139, 42.8396, 0, 25000, 157, 45, 20000, 6, 5, 'amatrice/fault_patches_opti.rectangles')
    fault_geom.patchesopti(13.2459, 42.8441, 0, 25000, 162, 45, 12000, 6, 4, 'amatrice/fault_patches_opti.rectangles')
fault_geom.patchesopti(13.233, 42.8365, 0, 25000, 160, 40, 12000, 6, 4, 'amatrice/fault_patches_opti_best.rectangles')
    fault_geom.patchesopti(13.233, 42.8365, 0, 25000, 160, 40, 12000, 6, 4, 'amatrice/fault_patches_opti_best.rectangles')
    fault_geom.patchesopti(13.215550, 42.870277, 0, 28000, 160, 40, 16000, 7, 4, 'amatrice/fault_patches_opti.rectangles')
    fault_geom.patchesopti(13.23948, 42.875095, 0, 28000, 164, 45, 16000, 7, 4, 'amatrice/fault_patches_opti.rectangles')
    fault_geom.patchesopti(13.2508, 42.8575, 0, 26000, 166, 45, 16000, 7, 4, 'amatrice/fault_patches_opti_166_45.rectangles')
    fault_geom.patchesopti(13.2294, 42.8757, 0, 30000, 160.7, 45, 16000, 7, 4, 'amatrice/fault_patches_opti_160_45.rectangles')

    """        
    
    ################ INPUT ####################
#        lon = float(sys.argv[1])
#        lat = float(sys.argv[2])
#        length = float(sys.argv[3])
#        strike = float(sys.argv[4])
#        dip = float(sys.argv[5])
#        width = float(sys.argv[6])
#        nstrike = float(sys.argv[7]) # number of rectangles along dip
#        ndip = float(sys.argv[8]) # number of rectangles along strike
#        
    depth = width * np.sin(dip)
    print( depth)
    
    ################ FAULT BUILDING ####################
    A = (lon, lat, z)
    #B = fh(A[0], A[1], A[2], strike, length)
    #print A
    #print B
    #C = fz(A[0], A[1], A[2], strike, dip, depth)
    #D = fz(B[0], B[1], B[2], strike, dip, depth)
    #print C
    #print D
    
    
    ################ FAULT SAMPLING ####################
    ddip = depth / (np.sin(dip) * ndip) # distance btw each point along dip
    dstrike = length /  nstrike # distance btw each point along strike
    print( dstrike)
    
    
    ######write all points -> NO PATCHES
    #Y = A
    #f = open('fault.txt', 'w')
    #while Y[2] <= depth:   
    #    X = Y
    #    f.write("%.6f %.6f %.6f\n" % (Y[0], Y[1], Y[2]) )
    #    
    #    while d(X[0],X[1],Y[0],Y[1]) < length - 1 :
    #        Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
    #        f.write("%.6f %.6f %.6f\n" % (Y[0], Y[1], Y[2]) )
    #        
    #    Y = X
    #    Y = fd(Y[0], Y[1], Y[2], strike, dip, Y[2] + ddip * np.cos(dip))
    #
    #f.close()
        

        
        
        
    ######write PATCHES
    N_pts = (ndip+1)*(nstrike+1) #points for patches
    N_pts_2 = (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3 + N_pts - (nstrike + 1)*(2+1)
    F = np.full( shape=(N_pts_2,3),fill_value=0., dtype='Float64') 
    
    Y = A
    i = 0
    j = 0
    
    while i < (nstrike+1+nstrike*(4-1))*3 -1:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i + nstrike+1+nstrike*(4-1)-1:
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike/4)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i + nstrike+1+nstrike*(4-1)
        Y = X
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/4)
        
    i = (nstrike+1+nstrike*(4-1))*3
    Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/4)
    
    while i < (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3 -1:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i+nstrike+1+nstrike*(2-1) -1 :
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike/2)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i +nstrike+1+nstrike*(2-1)
        Y = X
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/2)
    
    i = (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3
    Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/2)
    
    while i < (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3 + N_pts - (nstrike + 1)*(2+1):   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i+nstrike :
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i + nstrike +1
        Y = X
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip)

    print(F)
    
    k = 0
    l = 0
    N = N_pts_2 - nstrike - 2
    f = open(filename, 'w')
    N4 = nstrike+1+nstrike*(4-1)
    N2 = nstrike+1+nstrike*(2-1)
    
    #CSI wants depth in kilometers -> z divided by 1000
    
    while l < N4*2 -1:
        k = l
        while k < l +N4 -1 :
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N4+1,0], F[k+N4+1,1], F[k+N4+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N4,0], F[k+N4,1], F[k+N4,2]/1000) )
            k = k+1
        l = l+N4
        
    l = N4*2
    while l < N4*2 + N4 -1:
        k = l
        n = 0
        while k < l+N4-2:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+2,0], F[k+2,1], F[k+2,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N4-n+1,0], F[k+N4-n+1,1], F[k+N4-n+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N4-n,0], F[k+N4-n,1], F[k+N4-n,2]/1000) )
            k = k+2
            n = n+1
        l = l+N4
        
    l = N4*3     
    while l < N4*3 + N2*2 -1:
        k = l
        while k < l+N2-1:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2+1,0], F[k+N2+1,1], F[k+N2+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2,0], F[k+N2,1], F[k+N2,2]/1000) )
            k = k+1
        l = l+N2
    
    l = N4*3 + N2*2
    while l < N4*3 + N2*3 -1:
        k = l
        n = 0
        while k < l+N2-2:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+2,0], F[k+2,1], F[k+2,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2-n+1,0], F[k+N2-n+1,1], F[k+N2-n+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2-n,0], F[k+N2-n,1], F[k+N2-n,2]/1000) )
            k = k+2
            n = n+1
        l = l+N2  
    
    l = N4*3 + N2*3
    while l < N:
        k = l
        while k < l+nstrike:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+2,0], F[k+nstrike+2,1], F[k+nstrike+2,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+1,0], F[k+nstrike+1,1], F[k+nstrike+1,2]/1000) )
            k = k+1
        l = l+nstrike+1
        
    f.close()
        
    f2 = open(filename+'_area', 'w')
    f2.write("[")
    Nbr = ndip*nstrike + 12*nstrike # Nbr of patches
    for i in range(1,Nbr+1):
        if i <= nstrike*8:
            area = (length/(4*nstrike)) * (width/(4*ndip))
            f2.write("%.6f, " % (area) )
        elif nstrike *8 <= i <= nstrike*8 + nstrike*6:
            area = (length/(2*nstrike)) * (width/(2*ndip))
            f2.write("%.6f, " % (area) )
        elif nstrike*8 + nstrike*6 <= i <= Nbr-1 :
            area = length/nstrike * width/ndip
            f2.write("%.6f, " % (area) )
        else:
            area = length/nstrike * width/ndip
            f2.write("%.6f" % (area) )
    f2.write("]")
    f2.close()
    
    print('---------------------------------')
    print('Done! Opitmized fault file has been created with {stk}° strike and {d}°dip'.format(stk=strike, d=dip))
    
    return
    
def patchesopti_1(lon, lat, z, length, strike, dip, width, nstrike, ndip, filename):
    
    """
    Idem patches opti but with only one patche in strike
    """        
    
    ################ INPUT ####################
#        lon = float(sys.argv[1])
#        lat = float(sys.argv[2])
#        length = float(sys.argv[3])
#        strike = float(sys.argv[4])
#        dip = float(sys.argv[5])
#        width = float(sys.argv[6])
#        nstrike = float(sys.argv[7]) # number of rectangles along dip
#        ndip = float(sys.argv[8]) # number of rectangles along strike
#        
    depth = width * np.sin(dip)
    print (depth)
    
    ################ FAULT BUILDING ####################
    A = (lon, lat, z)
    #B = fh(A[0], A[1], A[2], strike, length)
    #print A
    #print B
    #C = fz(A[0], A[1], A[2], strike, dip, depth)
    #D = fz(B[0], B[1], B[2], strike, dip, depth)
    #print C
    #print D
    
    
    ################ FAULT SAMPLING ####################
    ddip = depth / (np.sin(dip) * ndip) # distance btw each point along dip
    dstrike = length /  nstrike # distance btw each point along strike
    print (dstrike)
    
    
    ######write all points -> NO PATCHES
    #Y = A
    #f = open('fault.txt', 'w')
    #while Y[2] <= depth:   
    #    X = Y
    #    f.write("%.6f %.6f %.6f\n" % (Y[0], Y[1], Y[2]) )
    #    
    #    while d(X[0],X[1],Y[0],Y[1]) < length - 1 :
    #        Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
    #        f.write("%.6f %.6f %.6f\n" % (Y[0], Y[1], Y[2]) )
    #        
    #    Y = X
    #    Y = fd(Y[0], Y[1], Y[2], strike, dip, Y[2] + ddip * np.cos(dip))
    #
    #f.close()
        

        
        
        
    ######write PATCHES
    N_pts = (ndip+1)*(nstrike+1) #points for patches
    N_pts_2 = (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3 + N_pts - (nstrike + 1)*(2+1)
    F = np.full( shape=(N_pts_2,3),fill_value=0., dtype='Float64') 
    
    Y = A
    i = 0
    j = 0
    
    while i < (nstrike+1+nstrike*(4-1))*3 -1:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i + nstrike+1+nstrike*(4-1)-1:
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike/4)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i + nstrike+1+nstrike*(4-1)
        Y = X
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/4)
        
    i = (nstrike+1+nstrike*(4-1))*3
    Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/4)
    
    while i < (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3 -1:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i+nstrike+1+nstrike*(2-1) -1 :
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike/2)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i +nstrike+1+nstrike*(2-1)
        Y = X
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/2)
    
    i = (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3
    Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/2)
    
    while i < (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3 + N_pts - (nstrike + 1)*(2+1):   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i+nstrike :
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i + nstrike +1
        Y = X
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip)

    print(F)
    
    k = 0
    l = 0
    N = N_pts_2 - nstrike - 2
    f = open(filename, 'w')
    N4 = nstrike+1+nstrike*(4-1)
    N2 = nstrike+1+nstrike*(2-1)
    
    #CSI wants depth in kilometers -> z divided by 1000
    k = l
    while l < N4*2 -1:
        f.write("%s\n" % (">       #") )
        f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+N4-1,0], F[k+N4-1,1], F[k+N4-1,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+2*N4-1,0], F[k+2*N4-1,1], F[k+2*N4-1,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+N4,0], F[k+N4,1], F[k+N4,2]/1000) )
        l = l+N4
        k=l
        
    l = N4*2
    n = 0
    while l < N4*2 + N4 -1:
        f.write("%s\n" % (">       #") )
        f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+N4-1,0], F[k+N4-1,1], F[k+N4-1,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+2*N2-1,0], F[k+2*N2-1-1,1], F[k+2*N2-1,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+N4,0], F[k+N4,1], F[k+N4,2]/1000) )
        l = l+N4
        k = l
        
    l = N4*3     
    while l < N4*3 + N2*2 -1:
        f.write("%s\n" % (">       #") )
        f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+N2-1,0], F[k+N2-1,1], F[k+N2-1,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+2*N2-1,0], F[k+2*N2-1,1], F[k+2*N2-1,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+N2,0], F[k+N2,1], F[k+N2,2]/1000) )
        l = l+N2
        k = l
    
    l = N4*3 + N2*2
    n = 0
    while l < N4*3 + N2*3 -1:
        f.write("%s\n" % (">       #") )
        f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+N2-1,0], F[k+N2-1,1], F[k+N2-1,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+2,0], F[k+nstrike+2-1,1], F[k+nstrike+2,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+N2,0], F[k+N2,1], F[k+N2,2]/1000) )
        l = l+N2  
        k = l
    
    l = N4*3 + N2*3
    while l < N:
        f.write("%s\n" % (">       #") )
        f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+2,0], F[k+nstrike+2,1], F[k+nstrike+2,2]/1000) )
        f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+1,0], F[k+nstrike+1,1], F[k+nstrike+1,2]/1000) )
        l = l+nstrike+1
        k = l

    f.close()
        
    f2 = open(filename+'_area', 'w')
    f2.write("[")
    Nbr = ndip*nstrike + 12*nstrike # Nbr of patches
    for i in range(1,Nbr+1):
        if i <= nstrike*8:
            area = (length/(nstrike)) * (width/(4*ndip))
            f2.write("%.6f, " % (area) )
        elif nstrike *8 <= i <= nstrike*8 + nstrike*6:
            area = (length/(nstrike)) * (width/(2*ndip))
            f2.write("%.6f, " % (area) )
        elif nstrike*8 + nstrike*6 <= i <= Nbr-1 :
            area = length/nstrike * width/ndip
            f2.write("%.6f, " % (area) )
        else:
            area = length/nstrike * width/ndip
            f2.write("%.6f" % (area) )
    f2.write("]")
    f2.close()
    
    print('---------------------------------')
    print('Done! Opitmized fault file has been created with {stk}° strike and {d}°dip'.format(stk=strike, d=dip))
    
    return
    
def patchesopti_curv(lon, lat, z, length, strike, dip_sur, dip_dep, width_curv, width, nstrike, ndip, filename):
    
    """
    Given a start point (lon lat z) (z = depth of top fault), a fault length
    (m), strike (degrees), dip at surface (degrees), dip at maximum depth (degrees)
    and width to begin curvature (m, need to fit patches width boundaries), width (m), 
    and a number of patches along strike and along dip
    
    Return a file of rectangle fault patches which can be read by CSI with optimized fault resolution:
    
    two first patches lines: patches are divided by 4
    first line of resulting patches: patches are divided by 4
    fault is curved with beginning dip of dip at surface value, termining dip of dip at max depth
    value, and curvature is begining a a certain width
    
    ex: 
    in python console:
    import fault_geom as fg
    fg.patchesopti_curv

    """        
    
    ################ INPUT ####################
#        lon = float(sys.argv[1])
#        lat = float(sys.argv[2])
#        length = float(sys.argv[3])
#        strike = float(sys.argv[4])
#        dip = float(sys.argv[5])
#        width = float(sys.argv[6])
#        nstrike = float(sys.argv[7]) # number of rectangles along dip
#        ndip = float(sys.argv[8]) # number of rectangles along strike
#        
    depth = width * np.sin(dip_sur)
    depth_nocurv = width_curv * np.sin(dip_sur)
    
    ################ FAULT BUILDING ####################
    A = (lon, lat, z)
    #B = fh(A[0], A[1], A[2], strike, length)
    #print A
    #print B
    #C = fz(A[0], A[1], A[2], strike, dip, depth)
    #D = fz(B[0], B[1], B[2], strike, dip, depth)
    #print C
    #print D
    
    
    ################ FAULT SAMPLING ####################
    ddip = depth / (np.sin(dip_sur) * ndip) # distance btw each point along dip
    dstrike = length /  nstrike # distance btw each point along strike
    
    
    
    ######write all points -> NO PATCHES
    #Y = A
    #f = open('fault.txt', 'w')
    #while Y[2] <= depth:   
    #    X = Y
    #    f.write("%.6f %.6f %.6f\n" % (Y[0], Y[1], Y[2]) )
    #    
    #    while d(X[0],X[1],Y[0],Y[1]) < length - 1 :
    #        Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
    #        f.write("%.6f %.6f %.6f\n" % (Y[0], Y[1], Y[2]) )
    #        
    #    Y = X
    #    Y = fd(Y[0], Y[1], Y[2], strike, dip, Y[2] + ddip * np.cos(dip))
    #
    #f.close()
        

        
        
        
    ######write PATCHES
    N_pts = (ndip+1)*(nstrike+1) #points for patches
    N_pts_2 = (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3 + N_pts - (nstrike + 1)*(2+1)
    F = np.full( shape=(N_pts_2,3),fill_value=0 ) 
    
    Y = A
    i = 0
    j = 0
                   
    
    dip = dip_sur
    if width_curv < width/(ndip*4):
        np4_nocurve = 0
    elif width/(ndip*4) <= width_curv < (width/(ndip*4))*2 :
        np4_nocurve = 1
    elif (width/(ndip*4))*2 <= width_curv < (width/(ndip*4))*4 :
        np4_nocurve = 2
    elif (width/(ndip*4))*4 <= width_curv < (width/(ndip*4))*6 :
        np4_nocurve = 4
    elif (width/(ndip*4))*6 <= width_curv < (width/(ndip*4))*8 :
        np4_nocurve = 6
    elif (width/(ndip*4))*8 <= width_curv < (width/(ndip*4))*12 :
        np4_nocurve = 8
    else:
        np4_nocurve = 12 + ((width - 3* (width/ndip))%(width/ndip))*4
     
     
    while i < (nstrike+1+nstrike*(4-1))*3 -1:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i + nstrike+1+nstrike*(4-1)-1:
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike/4)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i + nstrike+1+nstrike*(4-1)
        Y = X
        if Y[2] >= depth_nocurv:
            dip = dip - (dip_sur-dip_dep)/(ndip*4 - np4_nocurve)
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/4)
        
    i = (nstrike+1+nstrike*(4-1))*3
    if Y[2] >= depth_nocurv:
            dip = dip - (dip_sur-dip_dep)/(ndip*4 - np4_nocurve)
    Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/4)
    
    while i < (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3 -1:   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i+nstrike+1+nstrike*(2-1) -1 :
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike/2)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i +nstrike+1+nstrike*(2-1)
        Y = X
        if Y[2] >= depth_nocurv:
            dip = dip - (dip_sur-dip_dep)/(ndip*4 - np4_nocurve)*2
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/2)
    
    i = (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3
    if Y[2] >= depth_nocurv:
            dip = dip - (dip_sur-dip_dep)/(ndip*4 - np4_nocurve)*2
    Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip/2)
    
    while i < (nstrike+1+nstrike*(4-1))*3 + (nstrike+1+nstrike*(2-1))*3 + N_pts - (nstrike + 1)*(2+1):   
        X = Y
        F[i,0] = Y[0]
        F[i,1] = Y[1]
        F[i,2] = Y[2]
        j = i + 1
        
        while j <= i+nstrike :
            Y = fh(Y[0], Y[1], Y[2], strike, dstrike)
            F[j,0] = Y[0]
            F[j,1] = Y[1]
            F[j,2] = Y[2]
            j = j + 1
            
        i = i + nstrike +1
        Y = X
        if Y[2] >= depth_nocurv:
            dip = dip - (dip_sur-dip_dep)/(ndip*4 - np4_nocurve)*4
        Y = fd(Y[0], Y[1], Y[2], strike, dip, ddip)
    
    k = 0
    l = 0
    N = N_pts_2 - nstrike - 2
    f = open(filename, 'w')
    N4 = nstrike+1+nstrike*(4-1)
    N2 = nstrike+1+nstrike*(2-1)
    
    #CSI wants depth in kilometers -> z divided by 1000
    
    while l < N4*2 -1:
        k = l
        while k < l +N4 -1 :
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N4+1,0], F[k+N4+1,1], F[k+N4+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N4,0], F[k+N4,1], F[k+N4,2]/1000) )
            k = k+1
        l = l+N4
        
    l = N4*2
    while l < N4*2 + N4 -1:
        k = l
        n = 0
        while k < l+N4-2:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+2,0], F[k+2,1], F[k+2,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N4-n+1,0], F[k+N4-n+1,1], F[k+N4-n+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N4-n,0], F[k+N4-n,1], F[k+N4-n,2]/1000) )
            k = k+2
            n = n+1
        l = l+N4
        
    l = N4*3     
    while l < N4*3 + N2*2 -1:
        k = l
        while k < l+N2-1:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2+1,0], F[k+N2+1,1], F[k+N2+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2,0], F[k+N2,1], F[k+N2,2]/1000) )
            k = k+1
        l = l+N2
    
    l = N4*3 + N2*2
    while l < N4*3 + N2*3 -1:
        k = l
        n = 0
        while k < l+N2-2:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+2,0], F[k+2,1], F[k+2,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2-n+1,0], F[k+N2-n+1,1], F[k+N2-n+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+N2-n,0], F[k+N2-n,1], F[k+N2-n,2]/1000) )
            k = k+2
            n = n+1
        l = l+N2  
    
    l = N4*3 + N2*3
    while l < N:
        k = l
        while k < l+nstrike:
            f.write("%s\n" % (">       #") )
            f.write("%.6f %.6f %.6f\n" % (F[k,0], F[k,1], F[k,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+1,0], F[k+1,1], F[k+1,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+2,0], F[k+nstrike+2,1], F[k+nstrike+2,2]/1000) )
            f.write("%.6f %.6f %.6f\n" % (F[k+nstrike+1,0], F[k+nstrike+1,1], F[k+nstrike+1,2]/1000) )
            k = k+1
        l = l+nstrike+1
        
    f.close()
        
    f2 = open(filename+'_area', 'w')
    f2.write("[")
    Nbr = ndip*nstrike + 12*nstrike # Nbr of patches
    for i in range(1,Nbr+1):
        if i <= nstrike*8:
            area = (length/(4*nstrike)) * (width/(4*ndip))
            f2.write("%.6f, " % (area) )
        elif nstrike *8 <= i <= nstrike*8 + nstrike*6:
            area = (length/(2*nstrike)) * (width/(2*ndip))
            f2.write("%.6f, " % (area) )
        elif nstrike*8 + nstrike*6 <= i <= Nbr-1 :
            area = length/nstrike * width/ndip
            f2.write("%.6f, " % (area) )
        else:
            area = length/nstrike * width/ndip
            f2.write("%.6f" % (area) )
    f2.write("]")
    f2.close()
    
    return
    
if __name__ == "__main__":
    patchesCurv(-72.77492, -33.0, -9600., -400000., 184.76, 112000. , 180+7.3, 180+24.5, 40000., 25, 7., 'illapel_fault_curv1.rectangles')#,diprg=np.array([ 187.8, 190.66666667, 193.53333333, 196.4 ,199.26666667, 202.13333333, 205.])) 
    patches(-71.629298, -33.079090, -40122.378, -400000., 184.76, 180+24.5, 96000., 25, 6, 'illapel_fault_plan1.rectangles')
    