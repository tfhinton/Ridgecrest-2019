# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:07:01 2016

@author: Théa Ragon

Plot PDFs from Altar
"""

# Import Python Libraries
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.cm as cmx
import matplotlib.ticker as mtick
import re
import subprocess
import os
import colormap as cmap
import sys
import h5py
import seaborn as sns
import scipy
#import cartopy 
from scipy.spatial import Voronoi
#from makeitpop import makeitpop
from matplotlib.ticker import MaxNLocator
import mpl_toolkits.mplot3d as a3
from pdb import set_trace
from PIL import ImageColor
from PIL import Image
# from pdf2image import convert_from_path, convert_from_bytes
import scipy.interpolate as sciint
import cmocean.cm as cmo

# local
from CurvedText import CurvedText

cm_parula = [[0.2422, 0.1504, 0.6603],
[0.2444, 0.1534, 0.6728],
[0.2464, 0.1569, 0.6847],
[0.2484, 0.1607, 0.6961],
[0.2503, 0.1648, 0.7071],
[0.2522, 0.1689, 0.7179],
[0.254, 0.1732, 0.7286],
[0.2558, 0.1773, 0.7393],
[0.2576, 0.1814, 0.7501],
[0.2594, 0.1854, 0.761],
[0.2611, 0.1893, 0.7719],
[0.2628, 0.1932, 0.7828],
[0.2645, 0.1972, 0.7937],
[0.2661, 0.2011, 0.8043],
[0.2676, 0.2052, 0.8148],
[0.2691, 0.2094, 0.8249],
[0.2704, 0.2138, 0.8346],
[0.2717, 0.2184, 0.8439],
[0.2729, 0.2231, 0.8528],
[0.274, 0.228, 0.8612],
[0.2749, 0.233, 0.8692],
[0.2758, 0.2382, 0.8767],
[0.2766, 0.2435, 0.884],
[0.2774, 0.2489, 0.8908],
[0.2781, 0.2543, 0.8973],
[0.2788, 0.2598, 0.9035],
[0.2794, 0.2653, 0.9094],
[0.2798, 0.2708, 0.915],
[0.2802, 0.2764, 0.9204],
[0.2806, 0.2819, 0.9255],
[0.2809, 0.2875, 0.9305],
[0.2811, 0.293, 0.9352],
[0.2813, 0.2985, 0.9397],
[0.2814, 0.304, 0.9441],
[0.2814, 0.3095, 0.9483],
[0.2813, 0.315, 0.9524],
[0.2811, 0.3204, 0.9563],
[0.2809, 0.3259, 0.96],
[0.2807, 0.3313, 0.9636],
[0.2803, 0.3367, 0.967],
[0.2798, 0.3421, 0.9702],
[0.2791, 0.3475, 0.9733],
[0.2784, 0.3529, 0.9763],
[0.2776, 0.3583, 0.9791],
[0.2766, 0.3638, 0.9817],
[0.2754, 0.3693, 0.984],
[0.2741, 0.3748, 0.9862],
[0.2726, 0.3804, 0.9881],
[0.271, 0.386, 0.9898],
[0.2691, 0.3916, 0.9912],
[0.267, 0.3973, 0.9924],
[0.2647, 0.403, 0.9935],
[0.2621, 0.4088, 0.9946],
[0.2591, 0.4145, 0.9955],
[0.2556, 0.4203, 0.9965],
[0.2517, 0.4261, 0.9974],
[0.2473, 0.4319, 0.9983],
[0.2424, 0.4378, 0.9991],
[0.2369, 0.4437, 0.9996],
[0.2311, 0.4497, 0.9995],
[0.225, 0.4559, 0.9985],
[0.2189, 0.462, 0.9968],
[0.2128, 0.4682, 0.9948],
[0.2066, 0.4743, 0.9926],
[0.2006, 0.4803, 0.9906],
[0.195, 0.4861, 0.9887],
[0.1903, 0.4919, 0.9867],
[0.1869, 0.4975, 0.9844],
[0.1847, 0.503, 0.9819],
[0.1831, 0.5084, 0.9793],
[0.1818, 0.5138, 0.9766],
[0.1806, 0.5191, 0.9738],
[0.1795, 0.5244, 0.9709],
[0.1785, 0.5296, 0.9677],
[0.1778, 0.5349, 0.9641],
[0.1773, 0.5401, 0.9602],
[0.1768, 0.5452, 0.956],
[0.1764, 0.5504, 0.9516],
[0.1755, 0.5554, 0.9473],
[0.174, 0.5605, 0.9432],
[0.1716, 0.5655, 0.9393],
[0.1686, 0.5705, 0.9357],
[0.1649, 0.5755, 0.9323],
[0.161, 0.5805, 0.9289],
[0.1573, 0.5854, 0.9254],
[0.154, 0.5902, 0.9218],
[0.1513, 0.595, 0.9182],
[0.1492, 0.5997, 0.9147],
[0.1475, 0.6043, 0.9113],
[0.1461, 0.6089, 0.908],
[0.1446, 0.6135, 0.905],
[0.1429, 0.618, 0.9022],
[0.1408, 0.6226, 0.8998],
[0.1383, 0.6272, 0.8975],
[0.1354, 0.6317, 0.8953],
[0.1321, 0.6363, 0.8932],
[0.1288, 0.6408, 0.891],
[0.1253, 0.6453, 0.8887],
[0.1219, 0.6497, 0.8862],
[0.1185, 0.6541, 0.8834],
[0.1152, 0.6584, 0.8804],
[0.1119, 0.6627, 0.877],
[0.1085, 0.6669, 0.8734],
[0.1048, 0.671, 0.8695],
[0.1009, 0.675, 0.8653],
[0.0964, 0.6789, 0.8609],
[0.0914, 0.6828, 0.8562],
[0.0855, 0.6865, 0.8513],
[0.0789, 0.6902, 0.8462],
[0.0713, 0.6938, 0.8409],
[0.0628, 0.6972, 0.8355],
[0.0535, 0.7006, 0.8299],
[0.0433, 0.7039, 0.8242],
[0.0328, 0.7071, 0.8183],
[0.0234, 0.7103, 0.8124],
[0.0155, 0.7133, 0.8064],
[0.0091, 0.7163, 0.8003],
[0.0046, 0.7192, 0.7941],
[0.0019, 0.722, 0.7878],
[0.0009, 0.7248, 0.7815],
[0.0018, 0.7275, 0.7752],
[0.0046, 0.7301, 0.7688],
[0.0094, 0.7327, 0.7623],
[0.0162, 0.7352, 0.7558],
[0.0253, 0.7376, 0.7492],
[0.0369, 0.74, 0.7426],
[0.0504, 0.7423, 0.7359],
[0.0638, 0.7446, 0.7292],
[0.077, 0.7468, 0.7224],
[0.0899, 0.7489, 0.7156],
[0.1023, 0.751, 0.7088],
[0.1141, 0.7531, 0.7019],
[0.1252, 0.7552, 0.695],
[0.1354, 0.7572, 0.6881],
[0.1448, 0.7593, 0.6812],
[0.1532, 0.7614, 0.6741],
[0.1609, 0.7635, 0.6671],
[0.1678, 0.7656, 0.6599],
[0.1741, 0.7678, 0.6527],
[0.1799, 0.7699, 0.6454],
[0.1853, 0.7721, 0.6379],
[0.1905, 0.7743, 0.6303],
[0.1954, 0.7765, 0.6225],
[0.2003, 0.7787, 0.6146],
[0.2061, 0.7808, 0.6065],
[0.2118, 0.7828, 0.5983],
[0.2178, 0.7849, 0.5899],
[0.2244, 0.7869, 0.5813],
[0.2318, 0.7887, 0.5725],
[0.2401, 0.7905, 0.5636],
[0.2491, 0.7922, 0.5546],
[0.2589, 0.7937, 0.5454],
[0.2695, 0.7951, 0.536],
[0.2809, 0.7964, 0.5266],
[0.2929, 0.7975, 0.517],
[0.3052, 0.7985, 0.5074],
[0.3176, 0.7994, 0.4975],
[0.3301, 0.8002, 0.4876],
[0.3424, 0.8009, 0.4774],
[0.3548, 0.8016, 0.4669],
[0.3671, 0.8021, 0.4563],
[0.3795, 0.8026, 0.4454],
[0.3921, 0.8029, 0.4344],
[0.405, 0.8031, 0.4233],
[0.4184, 0.803, 0.4122],
[0.4322, 0.8028, 0.4013],
[0.4463, 0.8024, 0.3904],
[0.4608, 0.8018, 0.3797],
[0.4753, 0.8011, 0.3691],
[0.4899, 0.8002, 0.3586],
[0.5044, 0.7993, 0.348],
[0.5187, 0.7982, 0.3374],
[0.5329, 0.797, 0.3267],
[0.547, 0.7957, 0.3159],
[0.5609, 0.7943, 0.305],
[0.5748, 0.7929, 0.2941],
[0.5886, 0.7913, 0.2833],
[0.6024, 0.7896, 0.2726],
[0.6161, 0.7878, 0.2622],
[0.6297, 0.7859, 0.2521],
[0.6433, 0.7839, 0.2423],
[0.6567, 0.7818, 0.2329],
[0.6701, 0.7796, 0.2239],
[0.6833, 0.7773, 0.2155],
[0.6963, 0.775, 0.2075],
[0.7091, 0.7727, 0.1998],
[0.7218, 0.7703, 0.1924],
[0.7344, 0.7679, 0.1852],
[0.7468, 0.7654, 0.1782],
[0.759, 0.7629, 0.1717],
[0.771, 0.7604, 0.1658],
[0.7829, 0.7579, 0.1608],
[0.7945, 0.7554, 0.157],
[0.806, 0.7529, 0.1546],
[0.8172, 0.7505, 0.1535],
[0.8281, 0.7481, 0.1536],
[0.8389, 0.7457, 0.1546],
[0.8495, 0.7435, 0.1564],
[0.86, 0.7413, 0.1587],
[0.8703, 0.7392, 0.1615],
[0.8804, 0.7372, 0.165],
[0.8903, 0.7353, 0.1695],
[0.9, 0.7336, 0.1749],
[0.9093, 0.7321, 0.1815],
[0.9184, 0.7308, 0.189],
[0.9272, 0.7298, 0.1973],
[0.9357, 0.729, 0.2061],
[0.944, 0.7285, 0.2151],
[0.9523, 0.7284, 0.2237],
[0.9606, 0.7285, 0.2312],
[0.9689, 0.7292, 0.2373],
[0.977, 0.7304, 0.2418],
[0.9842, 0.733, 0.2446],
[0.99, 0.7365, 0.2429],
[0.9946, 0.7407, 0.2394],
[0.9966, 0.7458, 0.2351],
[0.9971, 0.7513, 0.2309],
[0.9972, 0.7569, 0.2267],
[0.9971, 0.7626, 0.2224],
[0.9969, 0.7683, 0.2181],
[0.9966, 0.774, 0.2138],
[0.9962, 0.7798, 0.2095],
[0.9957, 0.7856, 0.2053],
[0.9949, 0.7915, 0.2012],
[0.9938, 0.7974, 0.1974],
[0.9923, 0.8034, 0.1939],
[0.9906, 0.8095, 0.1906],
[0.9885, 0.8156, 0.1875],
[0.9861, 0.8218, 0.1846],
[0.9835, 0.828, 0.1817],
[0.9807, 0.8342, 0.1787],
[0.9778, 0.8404, 0.1757],
[0.9748, 0.8467, 0.1726],
[0.972, 0.8529, 0.1695],
[0.9694, 0.8591, 0.1665],
[0.9671, 0.8654, 0.1636],
[0.9651, 0.8716, 0.1608],
[0.9634, 0.8778, 0.1582],
[0.9619, 0.884, 0.1557],
[0.9608, 0.8902, 0.1532],
[0.9601, 0.8963, 0.1507],
[0.9596, 0.9023, 0.148],
[0.9595, 0.9084, 0.145],
[0.9597, 0.9143, 0.1418],
[0.9601, 0.9203, 0.1382],
[0.9608, 0.9262, 0.1344],
[0.9618, 0.932, 0.1304],
[0.9629, 0.9379, 0.1261],
[0.9642, 0.9437, 0.1216],
[0.9657, 0.9494, 0.1168],
[0.9674, 0.9552, 0.1116],
[0.9692, 0.9609, 0.1061],
[0.9711, 0.9667, 0.1001],
[0.973, 0.9724, 0.0938],
[0.9749, 0.9782, 0.0872],
[0.9769, 0.9839, 0.0805]]
parula_map = colors.LinearSegmentedColormap.from_list('parula', cm_parula)

color1 = [(218,240,178),(163,219,184),(96,194,192),(47,163,194),(32,120,180),(36,73,158)]
color12 = [(218,240,178),(163,219,184),(96,194,192),(47,163,194),(32,120,180)]
color2 = [(255,247,188),(254,227,145),(254,196,79),(254,153,41),(236,112,20),(204,76,2),(153,52,4)]
color3 = [(222,235,247),(198,219,239),(158,202,225),(107,174,214),(66,146,198),(33,113,181),(8,81,156)]
color4= [(253,208,162),(253,174,107),(253,141,60),(241,105,19),(217,72,1),(166,54,3)]
#colornames = list(reversed(color4)) + color1 + list(reversed(color12)) + color4
colornames = list(reversed(color1)) + color4
# avec gris:
##colornames = list(reversed(color1)) + color4 + [(166,54,3),(166,54,3),(186,108,101),(184,165,164),(211,211,211)]
# avec violet: 
#colornames = list(reversed(color1)) + color4 + [(166,54,3),(130,60,89),(98,51,115)]

#colors2=list(reversed([(43,71,116),(32,110,142),(69,147,155),(125,181,165),(203,216,203),(252,252,247)]))
#colors2=list(reversed([(3,69,105),(11,107,136),(67,146,149),(132,181,179),(193,216,212),(252,252,247)]))
colorsco=[(250,250,250),(255,247,236),(254,232,200),(253,212,158),(253,187,132),(252,141,89),(239,101,72),(215,48,31),(179,0,0),(127,0,0)]
colorspo=[(250,250,250),(250,250,250),(247,252,240),(224,243,219),(204,235,197),(168,221,181),(123,204,196),(78,179,211),(43,140,190),(8,104,172),(8,64,129)]
colorsco_diverging=list(reversed(colorsco))+colorsco
diverging2_hex = ['#005a74', '#35818d', '#65a8a7', '#a0cec5', '#efefe9', '#fdb88a', '#f08345', '#d55015', '#b21100']
coupling_hex = ['#ebebeb','#ecc8c5','#dea9af','#c98ca0','#b07195','#96578c','#793e85','#59277f','#32107a']
coupling = [ImageColor.getcolor(i, "RGB") for i in coupling_hex]
diverging2 = [ImageColor.getcolor(i, "RGB") for i in diverging2_hex]
darklavend = '#cdcde8ff'
name = 'test'

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

colorsco_above_transparent_hex =  ['#fafafa00', '#fff7ec8d', '#fee8c89b', '#fdd49ed2', '#fdbb84e9', '#fc8d59ff', '#ef6548ff', '#d7301fff', '#b30000ff', '#7f0000ff'] + above_hex
colorsco_above_transparent = [ImageColor.getcolor(i, "RGB") for i in colorsco_above_transparent_hex]
colorsco_above_transparent_rgba = [(y[0]/255., y[1]/255., y[2]/255.) for y in colorsco_above_transparent]

# colorsco_above2_hex = ['#faf9f9', '#fbeddb', '#fce1bc', '#fdd39f', '#fdbe89', '#fca874', '#fa9361', '#f47e55', '#ee6949', '#e6533b', '#da3c2a', '#cd2619', '#bc160e', '#a60d08', '#900302', '#7b030c', '#66071d', '#510b2f', '#4c1355', '#491a7f', '#4622a9', '#4622a9']
colorsco_above2_hex = ['#faf9f9', '#fbeddb', '#fdd39f', '#fca874', '#f47e55', '#ee6949', '#e6533b', '#da3c2a', '#cd2619', '#bc160e', '#a60d08', '#990e1b', '#8b0e28', '#7d0e35', '#6e0e42', '#5d0f4f', '#49105d', '#29106f']
colorsco_above2 = [ImageColor.getcolor(i, "RGB") for i in colorsco_above2_hex]
colorsco_above2_rgba = [(y[0]/255., y[1]/255., y[2]/255.) for y in colorsco_above2]


# faf9f9,fdd6a2,fc9964,eb5d42,c81c12,880000

# font = {'family' : 'Source Sans Pro',
#         'weight' : 'light',
#         'size'   : 12}
#
# matplotlib.rc('font', **font)
# from matplotlib import font_manager
# font_manager._rebuild()

# plt.style.use('/home/thea/mycode/python/myfig.mplstyle')

myblue = '#244c77ff'
mycyan = '#3f7f93ff'
myred ='#c3553aff'
myorange ='#f4a40bff'

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.8, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def return_topo(X):
    from scipy.interpolate import interp1d
    from scipy import signal

    ## define parameters (distance in km)
    a = 2.0
    b = 4.0
    c = 180.0
    d = 2.0
    e = 200.0
    trench_width = 20.
    
    ## define rectangle function
    def rect(x, center, width, height):
        return np.where(abs(x-center)<=width/2., height, 0)
    def step(x, center, height):
        return np.where(x>=center, height, 0)
    def ramp(x, center, height,end):
        y = []
        for i in x:
            if i < center:
                y.append(0.)
            elif (i>=center) & (i<=end):
                y.append( (height/(end-center))*(i-center))
            else:
                y.append( height)
        return y
    
    x = np.linspace(-400,1000.,1000)
    trench = step(x,-trench_width/2,-a)
    prism = ramp(x,trench_width/2.,a+b,70.)
    dist_arc = trench_width/2. + c + e/2.
    arc = rect(x,dist_arc,e,d)    
    ## convolve rectangle function with window
    win_trench = signal.windows.blackmanharris(40)
    trench_smooth = signal.convolve(trench, win_trench, mode='same') / sum(win_trench)
    win_prism = signal.windows.blackmanharris(40)
    prism_smooth = signal.convolve(prism, win_prism, mode='same') / sum(win_prism)
    other = arc
    win_other = signal.windows.blackmanharris(80)
    other_smooth = signal.convolve(other, win_other, mode='same') / sum(win_other)
    profile = trench_smooth + prism_smooth + other_smooth -b
    
    ##interp
    f = interp1d(x, profile)
    return f(X)

def colorFromBivariateData(Z1,Z2,cmap1 = plt.cm.Blues, cmap2 = plt.cm.Reds, preset = False):
    '''
    from https://gist.github.com/wolfiex/64d2faa495f8f0e1b1a68cdbdf3817f1#file-bivariate-py
    '''

    if preset:
        z1mn = 0.
        z2mn = 0.
        z1mx = 1.
        z2mx = 1.
    else:
        z1mn = Z1.min()
        z2mn = Z2.min()
        z1mx = Z1.max()
        z2mx = Z2.max()        

    # Rescale values to fit into colormap range (0->255)
    Z1_plot = np.array(255*(Z1-z1mn)/(z1mx-z1mn), dtype=np.int)
    Z2_plot = np.array(255*(Z2-z2mn)/(z2mx-z2mn), dtype=np.int)

    Z1_color = cmap1(Z1_plot)
    Z2_color = cmap2(Z2_plot)

    # Color for each point
    Z_color = np.sum([Z1_color, Z2_color], axis=0)/2.0

    return Z_color

def colorFromData(Z1,cmap1 = plt.cm.Blues, preset = False):
    '''
    from https://gist.github.com/wolfiex/64d2faa495f8f0e1b1a68cdbdf3817f1#file-bivariate-py
    '''
    if preset:
        z1mn = 0.
        z1mx = 1.
    else:
        z1mn = Z1.min()
        z1mx = Z1.max()

    # Rescale values to fit into colormap range (0->255)
#    Z1_plot = np.array(255*(Z1-z1mn)/(z1mx-z1mn), dtype=np.int)
#    Z1_color = cmap1(Z1_plot)
    Z1_color = cmap1.to_rgba(Z1)

    return Z1_color

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

def voronoi_finite_polygons_2d(vor, radius=None):
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


def gmtcpt_open(cptf, name=None):
    ''' modified from pycpt
    '''
        # generate cmap name
    if name is None:
        name = '_'.join(os.path.basename(cptf.name).split('.')[:-1])

    # process file
    x = []
    r = []
    g = []
    b = []
    #lastls = None
    for l in cptf.readlines()[:-3]:
        ls = re.split('\t|[/]|\n', l)

        # skip empty lines
        if not ls:
            continue

        # parse header info
        if ls[0] in ["#", b"#"]:
            if ls[-1] in ["HSV", b"HSV"]:
                colorModel = "HSV"
            else:
                colorModel = "RGB"
                continue
        else:
            colorModel = "RGB"

        # skip BFN info
        if ls[0] in ["B", b"B", "F", b"F", "N", b"N"]:
            continue

        # parse color vectors
        x.append(float(ls[0]))
        r.append(float(ls[1]))
        g.append(float(ls[2]))
        b.append(float(ls[3]))
        
        x.append(float(ls[4]))
        r.append(float(ls[5]))
        g.append(float(ls[6]))
        b.append(float(ls[7]))

        # save last row
        #lastls = ls

    x = np.array(x)
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    if colorModel == "HSV":
        for i in range(r.shape[0]):
            # convert HSV to RGB
            rr,gg,bb = colorsys.hsv_to_rgb(r[i]/360., g[i], b[i])
            r[i] = rr ; g[i] = gg ; b[i] = bb
    elif colorModel == "RGB":
        r /= 255.
        g /= 255.
        b /= 255.

    red = []
    blue = []
    green = []
    xNorm = (x - x[0])/(x[-1] - x[0])
    for i in range(len(x)):
        red.append([xNorm[i],r[i],r[i]])
        green.append([xNorm[i],g[i],g[i]])
        blue.append([xNorm[i],b[i],b[i]])

    # return colormap
    cdict = dict(red=red,green=green,blue=blue)
    
    cpt = colors.LinearSegmentedColormap(name,cdict)
    
    plt.register_cmap(cmap=cpt)
    
    return colors.LinearSegmentedColormap(name=name,segmentdata=cdict)

def gmtcpt(cptf, name=None):
    ''' modified from pycpt
    '''
    
    with open(cptf, 'r') as cptf:
        return gmtcpt_open(cptf, name=name)



def slipres_gmt(path, step1, step2):
    """
    Created on Tue Oct 25 13:49:35 2016
    Calculates residuals between two slip models from altar
    
    /!\ Warning!! These residuals are normalized by the maximum slip on each patch, and are given in percentage
        The interpretation thus needs to be performed regarding the coseismic slip values for one model
        --> indeed, a high residual percentage on a very small slip patch is not very significant
    
    input:
        - path: ex path="/u/moana/user/ragon/code/altar/am_static/results/amstatic_224_cpdip"
        - step1 ex: "052"
        - step2 ex: "050"
    
    return:
        - 
    """
    try:
        fi1 = open(path+'/step_'+step1+'_slip.dat', 'r')
        fi2 = open(path+'/step_'+step2+'_slip.dat', 'r')
        
        slip1 = []
        slip2 = []
        
        li1 = fi1.readlines()
        for li in li1:
            l = li.split(' ')
            if l[0] == '>':
                slip1.append(float(l[1].replace('-Z','')))
        
        li2 = fi2.readlines()
        for li in li2:
            l = li.split(' ')
            if l[0] == '>':
                slip2.append(float(l[1].replace('-Z','')))
        
        res = [abs( slip2[i] - slip1[i] )/max(slip2[i],slip1[i])*100 for i in range(len(slip1))]
        res2 = [abs( slip2[i] - slip1[i] ) for i in range(len(slip1))]
        
        fi = open(path+'/slipres_'+step2+'-'+step1+'_100.dat', 'w')
        i = 0
        for li in li2:
            l = li.split(' ')
            if l[0] == '>':
                fi.write( "%s %s\n" %('>', '-Z'+str(res[i])) )
                i = i+1
            else:
                fi.write("%s" %(li))
        fi.close()
        fi = open(path+'/slipres_'+step2+'-'+step1+'.dat', 'w')
        i = 0
        for li in li2:
            l = li.split(' ')
            if l[0] == '>':
                fi.write( "%s %s\n" %('>', '-Z'+str(res2[i])) )
                i = i+1
            else:
                fi.write("%s" %(li))
        fi.close()
        #---------------------------------------------------------------
        #---------------------------------------------------------------
        # Plot slip and synthetics
        script1 = """
        #!/bin/bash
        cd {di}
        source /u/moana/user/ragon/fig/amatrice/gmt/func/plot_slip.sh
        plotslipres {di} {fi}
        plotslipres {di} {fi2}
        rm -f slipres*.ps
        """.format(di=path, fi='slipres_'+step2+'-'+step1+'_100.dat', fi2='slipres_'+step2+'-'+step1+'.dat')
        subprocess.call(script1, shell=True)
        
        print('Done! You can find slip residuals fig here  --->  /u/moana/user/ragon/fig/amatrice/gmt')
    except Exception as err:
        sys.stderr.write('ERROR: %sn' % str(err))
    return()
 
   
def slipres(path, filename1, filename2, nstrike, ndip, length, width, savedir,faulttype='classical'):
    
    fi1 = open(path+'/'+filename1+'_slip.dat', 'r')
    fi2 = open(path+'/'+filename2+'_slip.dat', 'r')
    
    slip1 = []
    slip2 = []
    
    li1 = fi1.readlines()
    for li in li1:
        l = li.split(' ')
        if l[0] == '>':
            slip1.append(float(l[1].replace('-Z','')))
    
    li2 = fi2.readlines()
    for li in li2:
        l = li.split(' ')
        if l[0] == '>':
            slip2.append(float(l[1].replace('-Z','')))
    
    res = [(abs( slip2[i] - slip1[i] )/max(slip2[i],slip1[i]))*100 for i in range(len(slip1))]
    res2 = [abs( slip2[i] - slip1[i] ) for i in range(len(slip1))]
    res3 = [(abs( slip2[i] - slip1[i] )/max(slip2[i],slip1[i]))*100*(np.maximum(slip1[i],slip2[i])/np.maximum(np.amax(slip1),np.amax(slip2))) for i in range(len(slip1))]

    print("mean residuals: {} % , {} ponderate % and {} cm".format(np.mean(res),np.mean(res3),np.mean(res2)))
    print("max residuals: {} % , {} ponderate % and {} cm".format(np.max(res),np.max(res3),np.max(res2)))

    f1=filename1+'_'+filename2+'_100'
    plotSlip(f1, nstrike, ndip, length, width, res, 100,savedir,legend = 'Residual slip (%)',colorbar='slipres', fault_type=faulttype)
    f2=filename1+'_'+filename2+'_100pond'
    plotSlip(f2, nstrike, ndip, length, width, res3, 100,savedir,legend = 'Residual slip (%)',colorbar='slipres', fault_type=faulttype)
    f3=filename1+'_'+filename2
    plotSlip(f3, nstrike, ndip, length, width, res2, 2000, savedir,legend='Residual slip (cm)',colorbar='slipres', fault_type=faulttype)
    
    return
        
        
def PDFlast(filename, ns, nd, resdir, savedir, boundss, boundds, valmax, entrel=None, color=colorsco_above_rgba,ssh=False):
    
    '''
    Plots PDFs of last iteration samples from Altar
    
    IN ARGUMENT:
    You need to specify at least 8 arguments:
     --> filename

     --> ns = number of patches in strike: total for classical type, 
                 number of large patches for optimized
          
     --> nd = number of patches in dip: total for classical type, 
                 number of large patches for optimized
    
     --> resdir= directory of sampfile
     --> savedir= where to save PDF's figures
     --> boundss= boudaries for strike slip
     --> boundds = boundaries for dip slip
     optional:
     *--> valmax: maximal value for the colormap
     *--> entrel = relative entropy vector (length: number of patches)
     *--> colormap used to color PDFs

    USAGE:
    >>> import altar_pdf
    >>> altar_pdf.PDFlast()

    '''
    try:
        
        #---------------------------------------------------------------
        #                   get results from ALTAR
        #---------------------------------------------------------------
        # Number of parameters
        Np = 2*ns*nd

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
        samp = np.hstack((ss,ds))
        Ns = float(samp.shape[0])
#        set_trace()
        if Np <= 1.5*float(samp.shape[1]):  ## only one fault
            Np = int(samp.shape[1])
            x00 = np.linspace(boundss[0],boundss[1],2000)  # steps in strike
            x90 = np.linspace(boundds[0],boundds[1],3000)  # steps in dip
            n = norm(loc=0.,scale=boundss[1]) #pdf('Normal',x00,0,1)/2;
            u = uniform(boundds[0]+np.abs(boundds[0]-boundds[1])/200,boundds[1]-np.abs(boundds[0]-boundds[1])/200) #pdf('Uniform',x90,-0.5,60)/2.0165;
            p00 = n.pdf(x00)
            p90 = u.pdf(x90)
            p002 = n.pdf(x00[:-1])/2
            p902 = u.pdf(x90[:-1])

            post00 = np.zeros((int(Np//2),int(len(x00)-1)))
            post90 = np.zeros((int(Np//2),int(len(x90)-1)))
            for i in range(Np//2): ## count number of values for each patch
                #print np.histogram(samp2[:,i],bins=x00)[0] / Ns
                post00[i,:] = np.histogram(samp[:,i],bins=x00)[0] / Ns
                post90[i,:] = np.histogram((samp[:,i+Np//2]),bins=x90)[0] / Ns

    #        kl00 = np.zeros((Np//2,1))
    #        kl90 = np.zeros((Np//2,1))
            #for i in range(Np//2):
                #a00 = p002[post00[i,:] > beta]
                #b00 = post00[i, post00[i,:] > beta]
                #kl00[i] = sum( b00*np.log(b00/a00) )
                #a90 = p902[post90[i,:] > beta]
                #b90 = post90[i, post90[i,:] > beta]
                #kl90[i] = sum( b90*np.log(b90/a90) )

    #        moyss = np.mean( np.array([ i for i in samp[:,0:Np//2]]), axis=0 ) # mean for colorscale
    #        moyds = np.mean( np.array([ i for i in samp[:,Np//2:Np]]), axis=0 ) # mean for colorscale
            moy = np.mean( samp, axis=0 )
            moyss = moy[0:Np//2]
            moyds = moy[Np//2:Np]
            
            mle = np.empty((Np//2,2))
            inds = [np.argmax(post00[i,:]) for i in range(Np//2)]
            mle[:,0] = [x00[inds[i]] for i in range(Np//2)]
            inds = [np.argmax(post90[i,:]) for i in range(Np//2)]
            mle[:,1] = [x90[inds[i]] for i in range(Np//2)]
    
            x00 = np.linspace(boundss[0],boundss[1],1999)
            x90 = np.linspace(boundds[0],boundds[1],2999)
    #        x00 = np.arange(boundss[0]+np.abs(boundss[0]-boundss[1])/400,boundss[1]-np.abs(boundss[0]-boundss[1])/400,np.abs(boundss[0]-boundss[1])/200)  # steps in strike
    #        x90 = np.arange(boundds[0]+np.abs(boundds[0]-boundds[1])/200,boundds[1],np.abs(boundds[0]-boundds[1])/200)  # steps in dip
            plotPDFss(filename, Np, ns, nd, x00, p002, post00, moyss, width = '',ent=entrel,savedir=savedir,color=color,valmax=valmax[0])
            plotPDFds(filename, Np, ns, nd, x90, p902, post90, moyds, width = '',ent=entrel,savedir=savedir,color=color,valmax=valmax[1])
        
        else: #### 2 faults (co+post)
            sampco = samp[:,0:Np]
            samppost = samp[:,Np:-1]
            x00 = np.linspace(boundss[0],boundss[1],2000)  # steps in strike
            x90 = np.linspace(boundds[0],boundds[1],3000)  # steps in dip
            n = norm(loc=0.,scale=2.) #pdf('Normal',x00,0,1)/2;
            u = uniform(boundds[0],boundds[1]) #pdf('Uniform',x90,-0.5,60)/2.0165;
            p00 = n.pdf(x00)
            p90 = u.pdf(x90)
            p002 = n.pdf(x00[:-1])/2
            p902 = u.pdf(x90[:-1])

            postco00 = np.zeros((Np//2,len(x00)-1))
            postco90 = np.zeros((Np//2,len(x90)-1))
            postpo00 = np.zeros(((np.shape(samp)[1]-Np)/2,len(x00)-1))
            postpo90 = np.zeros(((np.shape(samp)[1]-Np)/2,len(x90)-1))
            for i in range(Np//2): ## count number of values for each patch
                #print np.histogram(samp2[:,i],bins=x00)[0] / Ns
                postco00[i,:] = np.histogram(sampco[:,i],bins=x00)[0] / Ns
                postco90[i,:] = np.histogram((-sampco[:,i+Np//2]),bins=x90)[0] / Ns
            for i in range(np.shape(samppost)[1]/2):
                postpo00[i,:] = np.histogram(samppost[:,i],bins=x00)[0] / Ns
                postpo90[i,:] = np.histogram((-samppost[:,i+np.shape(samppost)[1]/2]),bins=x90)[0] / Ns
    #        kl00 = np.zeros((Np//2,1))
    #        kl90 = np.zeros((Np//2,1))
            #for i in range(Np//2):
                #a00 = p002[post00[i,:] > beta]
                #b00 = post00[i, post00[i,:] > beta]
                #kl00[i] = sum( b00*np.log(b00/a00) )
                #a90 = p902[post90[i,:] > beta]
                #b90 = post90[i, post90[i,:] > beta]
                #kl90[i] = sum( b90*np.log(b90/a90) )

            moyssco = np.mean( np.array([ i for i in sampco[:,0:Np//2]]), axis=0 ) # mean for colorscale
            moydsco = np.mean( np.array([ i for i in sampco[:,Np//2:Np]]), axis=0 ) # mean for colorscale
            moysspo = np.mean( np.array([ i for i in samppost[:,0:np.shape(samppost)[1]/2]]), axis=0 ) # mean for colorscale
            moydspo = np.mean( np.array([ i for i in samppost[:,np.shape(samppost)[1]/2:-1]]), axis=0 ) # mean for colorscale

            x00 = np.linspace(boundss[0],boundss[1],1999)
            x90 = np.linspace(boundds[0],boundds[1],2999)
            plotPDFss(filename+'_co', Np, ns, nd, fault_type, x00, p002, postco00, moyssco, width = '',ent=entrel,savedir=savedir)
            plotPDFds(filename+'_co', Np, ns, nd, fault_type, x90, p902, postco90, moydsco, width = '',ent=entrel,savedir=savedir,color=color[0],valmax=valmax[0])
            plotPDFss(filename+'_post', Np, ns, nd, fault_type, x00, p002, postpo00, moysspo, width = '',ent=entrel,savedir=savedir)
            plotPDFds(filename+'_post', Np, ns, nd, fault_type, x90, p902, postpo90, moydspo, width = '',ent=entrel,savedir=savedir,color=color[1],valmax=valmax[1])

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
    return mle, x00, post00, x90, post90, samp
    
def PDFmoy(step, nalpha, ns, nd, resdir, savedir, boundss, boundds, valmax=130, color=colorsco):
    
    '''
    Plots PDFs of last iteration samples from Altar
    
    IN ARGUMENT:
    You need to specify at least 8 arguments:
     --> sampfile = ascii samples file (string)
          ex: 'step_063_samp.dat'
     
     --> alpha = mean over ? subfaults
          ex: 2

     --> ns = number of patches in strike: total for classical type, 
                 number of large patches for optimized
          
     --> nd = number of patches in dip: total for classical type, 
                 number of large patches for optimized
    
     --> resdir= directory of sampfile
     --> savedir= where to save PDF's figures
     --> boundss= boudaries for strike slip
     --> boundds = boundaries for dip slip
     optional:
     *--> valmax: maximal value for the colormap
     *--> entrel = relative entropy vector (length: number of patches)
     *--> colormap used to color PDFs

    USAGE:
    >>> import altar_pdf
    >>> altar_pdf.PDFlast()

    '''
    try: 
        #---------------------------------------------------------------
        #                   get results from ALTAR
        #---------------------------------------------------------------
        # Number of parameters
        Np = 2*ns*nd
        
        #extract all samples
    #    samp = np.fromfile(resdir+sampfile, sep=",")
    #    mpost = moy
    #    dim = samp.shape[0]/Np
    #    samp2 = samp.reshape((dim,Np))
        h5file =  h5py.File(resdir+'step_'+step+'.h5','r')
        samp = np.array(list(h5file[u'Sample Set']))
        beta = 0 # cut-off for non-zero pdfs  
        Ns = float(samp.shape[0])
    
        if samp.shape[1] > Np:
            samp = samp[:,0:np.int(samp.shape[1]/2)]
        
        x00 = np.linspace(boundss[0],boundss[1],200)  # steps in strike
        x90 = np.linspace(boundds[0],boundds[1],3000)  # steps in dip
        n = norm(loc=0.,scale=boundss[1]) #pdf('Normal',x00,0,1)/2;
        u = uniform(boundds[0]+np.abs(boundds[0]-boundds[1])/200,boundds[1]-np.abs(boundds[0]-boundds[1])/200) #pdf('Uniform',x90,-0.5,60)/2.0165;
        p00 = n.pdf(x00)
        p90 = u.pdf(x90)
        p002 = n.pdf(x00[:-1])/2
        p902 = u.pdf(x90[:-1])
        
    #    post00 = np.zeros((Np//2,len(x00)-1))
    #    post90 = np.zeros((Np//2,len(x90)-1))
    #    for i in range(Np//2): ## count number of values for each patch
    #        #print np.histogram(samp2[:,i],bins=x00)[0] / Ns
    #        post00[i,:] = np.histogram(samp[:,i],bins=x00,density=True)[0]
    #        post90[i,:] = np.histogram((-samp[:,i+Np//2]),bins=x90,density=True)[0]
    #        post90[i,:] =  post90[i,:] / np.amax(post90[i,:])
        
        postf00 = []
        postf90 = []
        moyds = []
        moyss=[]
        medds=[]
        medss=[]
        if len(nalpha)==2:
            if nalpha[0] != 0:
                for l in range(nalpha[0]):
                    for i in range(l*ns,l*ns+ns,1):
                        postf90.append(np.histogram((-samp[:,i+Np//2]),bins=x90,density=True)[0])
                        postf00.append(np.histogram((-samp[:,i]),bins=x90,density=True)[0])
                        moyds.append(np.mean(-samp[:,i+Np//2]))
                        medds.append(np.median(-samp[:,i+Np//2]))
                        moyss.append(np.mean(samp[:,i]))
                        medss.append(np.median(samp[:,i]))
            for l in range(nalpha[0],nalpha[0]+nalpha[1]*2-1,2):
                alpha=2
                for i in range(l*ns,l*ns+ns-1,alpha):
                    aver = (samp[:,i+Np//2] + samp[:,i+Np//2+alpha-1] \
                            +samp[:,i+Np//2+ns] + samp[:,i+Np//2+ns+alpha-1] )/4
                    postf90.append(np.histogram((-aver[:]),bins=x90,density=True)[0])
                    aver0 = (samp[:,i] + samp[:,i+alpha-1] \
                            + samp[:,i+ns] + samp[:,i+ns+alpha-1] )/4
                    postf00.append(np.histogram((aver0[:]),bins=x90,density=True)[0])
                    moyds.append(np.mean(-aver[:]))
                    medds.append(np.median(-aver[:]))
                    moyss.append(np.mean(aver0[:]))
                    medss.append(np.median(aver0[:]))
        if len(nalpha)==3:
            if nalpha[0] != 0:
                for l in range(nalpha[0]):
                    for i in range(l*ns,l*ns+ns,1):
                        postf90.append(np.histogram((samp[:,i+Np//2]),bins=x90,density=True)[0])
                        postf00.append(np.histogram((samp[:,i]),bins=x90,density=True)[0])
                        moyds.append(np.mean(samp[:,i+Np//2]))
                        medds.append(np.median(samp[:,i+Np//2]))
                        moyss.append(np.mean(samp[:,i]))
                        medss.append(np.median(samp[:,i]))
            for l in range(nalpha[0],nalpha[0]+nalpha[1]*2-1,2):
                alpha=2
                for i in range(l*ns,l*ns+ns-1,alpha):
                    aver = (samp[:,i+Np//2] + samp[:,i+Np//2+alpha-1] \
                            +samp[:,i+Np//2+ns] + samp[:,i+Np//2+ns+alpha-1] )/4
                    postf90.append(np.histogram((aver[:]),bins=x90,density=True)[0])
                    aver0 = (samp[:,i] + samp[:,i+alpha-1] \
                            + samp[:,i+ns] + samp[:,i+ns+alpha-1] )/4
                    postf00.append(np.histogram((aver0[:]),bins=x90,density=True)[0])
                    moyds.append(np.mean(aver[:]))
                    medds.append(np.median(aver[:]))
                    moyss.append(np.mean(aver0[:]))
                    medss.append(np.median(aver0[:]))
            for l in range(nalpha[0]+nalpha[1]*2,nalpha[0]+nalpha[1]*2+nalpha[2]*3-2,3):
                alpha=3
    #            try:
                for i in range(l*ns,l*ns+ns-2,alpha):
                    aver = (samp[:,i+Np//2] + samp[:,i+Np//2+alpha-2] + samp[:,i+Np//2+alpha-1] \
                            +samp[:,i+Np//2+ns] +samp[:,i+Np//2+ns+alpha-2] + samp[:,i+Np//2+ns+alpha-1] \
                            +samp[:,i+Np//2+2*ns] + samp[:,i+Np//2+2*ns+alpha-2] + samp[:,i+Np//2+2*ns+alpha-1]) /9
                    postf90.append(np.histogram((aver[:]),bins=x90,density=True)[0])
                    aver0 = (samp[:,i] + samp[:,i+alpha-2] + samp[:,i+alpha-1] \
                            + samp[:,i+ns] + samp[:,i+ns+alpha-2] + samp[:,i+ns+alpha-1] \
                            + samp[:,i+2*ns] + samp[:,i+2*ns+alpha-2] + samp[:,i+2*ns+alpha-1])/9
                    postf00.append(np.histogram((aver0[:]),bins=x90,density=True)[0])
                    moyds.append(np.mean(aver[:]))
                    medds.append(np.median(aver[:]))
                    moyss.append(np.mean(aver0[:]))
                    medss.append(np.median(aver0[:]))
        moyds=np.array(moyds)
        medds=np.array(medds)
        moyss=np.array(moyss)
        medss=np.array(medss)
    #    if alpha==2:
    #        for l in range(nd/2):
    #            for i in range(l*alpha*ns,l*alpha*ns+ns-1,alpha):
    #                aver = (samp[:,i+Np//2] + samp[:,i+Np//2+alpha-1] \
    #                        +samp[:,i+Np//2+ns] + samp[:,i+Np//2+ns+alpha-1] )/4
    #                postf90.append(np.histogram((-aver[:]),bins=x90,density=True)[0])
    #                aver0 = (samp[:,i] + samp[:,i+alpha-1] \
    #                        + samp[:,i+ns] + samp[:,i+ns+alpha-1] )/4
    #                postf00.append(np.histogram((aver0[:]),bins=x90,density=True)[0])
    #    if alpha==3:
    #        for l in range(nd/3):
    #            for i in range(l*alpha*ns,l*alpha*ns+ns-1,alpha):
    #                aver = (samp[:,i+Np//2] + samp[:,i+Np//2+alpha-2] + samp[:,i+Np//2+alpha-1] \
    #                        +samp[:,i+Np//2+ns] +samp[:,i+Np//2+ns+alpha-2] + samp[:,i+Np//2+ns+alpha-1] \
    #                        +samp[:,i+Np//2+2*ns] + samp[:,i+Np//2+2*ns+alpha-2] + samp[:,i+Np//2+2*ns+alpha-1]) /9
    #                postf90.append(np.histogram((-aver[:]),bins=x90,density=True)[0])
    #                aver0 = (samp[:,i] + samp[:,i+alpha-2] + samp[:,i+alpha-1] \
    #                        + samp[:,i+ns] + samp[:,i+ns+alpha-2] + samp[:,i+ns+alpha-1] \
    #                        + samp[:,i+2*ns] + samp[:,i+2*ns+alpha-2] + samp[:,i+2*ns+alpha-1])/9
    #                postf00.append(np.histogram((aver0[:]),bins=x90,density=True)[0])
                   
    #    i=ns*(nd-6)+1
    #    somme = (-samp[:,i+Np//2] -samp[:,i+Np//2+alpha-2] -samp[:,i+Np//2+alpha-1] \
    #                        -samp[:,i+Np//2+ns] -samp[:,i+Np//2+ns+alpha-2] -samp[:,i+Np//2+ns+alpha-1] \
    #                        -samp[:,i+Np//2+2*ns-1] -samp[:,i+Np//2+2*ns-1+alpha-2] -samp[:,i+Np//2+2*ns-1+alpha-1])/9
    #    test=np.histogram(somme[:],bins=x90,density=True)[0]
    #    plt.figure(6)
    #    x90 = np.arange(-2+0.1,202,0.1)
    #    plt.plot(x90+0.5,test)
        
        postf90 = np.array(postf90)
        x00 = np.linspace(boundss[0],boundss[1],199)
        x90 = np.linspace(boundds[0],boundds[1],2999)
    #    plotPDFss(filename, Np, ns/alpha, nd/alpha, "classical", x00, p002, postf00, moyss, width = '',ent=entrel,savedir=savedir)
    #    plotPDFds(filename, Np, ns/alpha, nd/alpha, "classical", x90, p902, postf90, moyds, width = '',ent=entrel,savedir=savedir,color=color,valmax=valmax)
        plotPDFds_moy(filename, Np, ns, nd, nalpha, x90, postf90, moyds, savedir,color=color,valmax=valmax)

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
    return moyds, medds, moyss, medss
    
def PDFcomp2(step, fault_type, ns, nd, resdir, savedir, valmax=7000, entrel=None, color=colornames):
    
    '''
    Compare PDFs of two last iterations samples
    
    IN ARGUMENT:
    You need to specify at least 8 arguments:
     --> steps of inversions
         ex: ['055','056']
          
     --> fault_type = fault geometry type: 'classical' or 'optimized'

     --> ns = number of patches in strike: total for classical type, 
                 number of large patches for optimized
          
     --> nd = number of patches in dip: total for classical type, 
                 number of large patches for optimized
    
     --> resdir= directory of sampfile
     --> savedir= where to save PDF's figures
     optional:
     *--> valmax: maximal value for the colormap
     *--> entrel = relative entropy vector (length: number of patches)
     *--> colormap used to color PDFs

    USAGE:
    >>> import altar_pdf
    >>> altar_pdf.PDFlast()

    '''
    
    #---------------------------------------------------------------
    #                   get results from ALTAR
    #---------------------------------------------------------------
    # Number of parameters
    if fault_type=="optimized":
        nbr_fp = 8*ns + 6*ns + (nd-2)*ns
        Np = 2*nbr_fp
    elif fault_type=="classical":
        Np = 2*ns*nd
    
    h5file =  h5py.File(resdir+'step_'+step[0]+'.h5','r')
    samp = np.array(list(h5file[u'Sample Set']))
    h5file2 =  h5py.File(resdir+'step_'+step[1]+'.h5','r')
    samp2 = np.array(list(h5file2[u'Sample Set']))
    Ns = float(samp.shape[0])
    
    if Np == float(samp.shape[1]):  ## only one fault
        x00 = np.arange(-10,10.5,0.5)  # steps in strike
        x90 = np.arange(-2,202,1)  # steps in dip
        n = norm(loc=0.,scale=2.) #pdf('Normal',x00,0,1)/2;
        u = uniform(-1.,201.) #pdf('Uniform',x90,-0.5,60)/2.0165;
        p00 = n.pdf(x00)
        p90 = u.pdf(x90)
        p002 = n.pdf(x00[:-1])/2
        p902 = u.pdf(x90[:-1])
        
        post00 = np.zeros((Np//2,len(x00)-1))
        post90 = np.zeros((Np//2,len(x90)-1))
        post002 = np.zeros((Np//2,len(x00)-1))
        post902 = np.zeros((Np//2,len(x90)-1))
        for i in range(Np//2): ## count number of values for each patch
            post00[i,:] = np.histogram(samp[:,i],bins=x00)[0] / Ns
            post90[i,:] = np.histogram((-samp[:,i+Np//2]),bins=x90)[0] / Ns
            post002[i,:] = np.histogram(samp2[:,i],bins=x00)[0] / Ns
            post902[i,:] = np.histogram((-samp2[:,i+Np//2]),bins=x90)[0] / Ns
                
        moy = np.mean( samp, axis=0 )
        moyss = moy[0:Np//2]
        moyds = moy[Np//2:Np]
        
        moy2 = np.mean( samp2, axis=0 )
        moyss2 = moy2[0:Np//2]
        moyds2 = moy2[Np//2:Np]
        
        x00 = np.arange(-9.75,10.25,0.5) 
        x90 = np.arange(-1,202,1)
        plotPDFds_comp([step[0][0:3],step[1][0:3]], Np, ns, nd, fault_type, x90, p902, post90,post902, moyds,moyds2, width = '',savedir=savedir,color=colorsco,valmax=valmax)
    
    elif float(samp.shape[1]) > Np: #### 2 faults (co+post) 
        sampco = samp[:,0:Np]
        samppost = samp[:,Np:2*Np]
        sampco2 = samp2[:,0:Np]
        samppost2 = samp2[:,Np:2*Np]
        x00 = np.arange(-10,10.5,0.5)  # steps in strike
        x90 = np.arange(-2,202,1)  # steps in dip
        x90p = np.arange(-2,102,1)
        n = norm(loc=0.,scale=2.) #pdf('Normal',x00,0,1)/2;
        u = uniform(-1.,201.) #pdf('Uniform',x90,-0.5,60)/2.0165;
        up = uniform(-1.,101.)
        p00 = n.pdf(x00)
        p90 = u.pdf(x90)
        p002 = n.pdf(x00[:-1])/2
        p902 = u.pdf(x90[:-1])
        p902p = u.pdf(x90p[:-1])
        
        postco00 = np.zeros((Np//2,len(x00)-1))
        postco90 = np.zeros((Np//2,len(x90)-1))
        postpo00 = np.zeros((Np//2,len(x00)-1))
        postpo90 = np.zeros((Np//2,len(x90p)-1))
        postco002 = np.zeros((Np//2,len(x00)-1))
        postco902 = np.zeros((Np//2,len(x90)-1))
        postpo002 = np.zeros((Np//2,len(x00)-1))
        postpo902 = np.zeros((Np//2,len(x90p)-1))
        for i in range(Np//2): ## count number of values for each patch
            #print np.histogram(samp2[:,i],bins=x00)[0] / Ns
            postco00[i,:] = np.histogram(sampco[:,i],bins=x00)[0] / Ns
            postco90[i,:] = np.histogram((-sampco[:,i+Np//2]),bins=x90)[0] / Ns
            postpo00[i,:] = np.histogram(samppost[:,i],bins=x00)[0] / Ns
            postpo90[i,:] = np.histogram((-samppost[:,i+Np//2]),bins=x90p)[0] / Ns 
            postco002[i,:] = np.histogram(sampco2[:,i],bins=x00)[0] / Ns
            postco902[i,:] = np.histogram((-sampco2[:,i+Np//2]),bins=x90)[0] / Ns
            postpo002[i,:] = np.histogram(samppost2[:,i],bins=x00)[0] / Ns
            postpo902[i,:] = np.histogram((-samppost2[:,i+Np//2]),bins=x90p)[0] / Ns 
#        kl00 = np.zeros((Np//2,1))
#        kl90 = np.zeros((Np//2,1))
        #for i in range(Np//2):
            #a00 = p002[post00[i,:] > beta]
            #b00 = post00[i, post00[i,:] > beta]
            #kl00[i] = sum( b00*np.log(b00/a00) )
            #a90 = p902[post90[i,:] > beta]
            #b90 = post90[i, post90[i,:] > beta]
            #kl90[i] = sum( b90*np.log(b90/a90) )
        
        moyssco = np.mean( np.array([ i for i in sampco[:,0:Np//2]]), axis=0 ) # mean for colorscale
        moydsco = np.mean( np.array([ i for i in sampco[:,Np//2:Np]]), axis=0 ) # mean for colorscale
        moysspo = np.mean( np.array([ i for i in samppost[:,0:Np//2]]), axis=0 ) # mean for colorscale
        moydspo = np.mean( np.array([ i for i in samppost[:,Np//2:Np]]), axis=0 ) # mean for colorscale
        moyssco2 = np.mean( np.array([ i for i in sampco2[:,0:Np//2]]), axis=0 ) # mean for colorscale
        moydsco2 = np.mean( np.array([ i for i in sampco2[:,Np//2:Np]]), axis=0 ) # mean for colorscale
        moysspo2 = np.mean( np.array([ i for i in samppost2[:,0:Np//2]]), axis=0 ) # mean for colorscale
        moydspo2 = np.mean( np.array([ i for i in samppost2[:,Np//2:Np]]), axis=0 ) # mean for colorscale
        
        x00 = np.arange(-9.75,10.25,0.5) 
        x90 = np.arange(-1,202,1)
        x90p = np.arange(-1,102,1)
        plotPDFds_comp([step[0][0:3],step[1][0:3]+'_co'], Np, ns, nd, fault_type, x90, p902, postco90, postco902,moydsco,moydsco2, width = '',savedir=savedir,color=colorsco,valmax=valmax[0])
        plotPDFds_comp([step[0][0:3],step[1][0:3]+'_post'], Np, ns, nd, fault_type, x90p, p902p, postpo90,postpo902, moydspo, moydspo2, width = '',savedir=savedir,color=colorspo,valmax=valmax[1])
    return()
    
def PDFmoycomp(step, nalpha, ns, nd, boundss, boundds, valmax, resdir, savedir, entrel=False, color=colorsco):
    
    '''
    Compare PDFs of two last iterations samples
    
    IN ARGUMENT:
    You need to specify at least 8 arguments:
     --> steps of inversions
         ex: ['055','056']
    --> nalpha: number of subfaults
         ex: [4,4] or [3,1,1]
                
     --> fault_type = fault geometry type: 'classical' or 'optimized'

     --> ns = number of patches in strike: total for classical type, 
                 number of large patches for optimized
          
     --> nd = number of patches in dip: total for classical type, 
                 number of large patches for optimized
    
     --> resdir= directory of sampfile
     --> savedir= where to save PDF's figures
     optional:
     *--> valmax: maximal value for the colormap
     *--> entrel = relative entropy vector (length: number of patches)
     *--> colormap used to color PDFs

    USAGE:
    >>> import altar_pdf
    >>> altar_pdf.PDFlast()

    '''
    
    #---------------------------------------------------------------
    #                   get results from ALTAR
    #---------------------------------------------------------------
    try:
        # Number of parameters
        Np = 2*ns*nd
        
        h5file =  h5py.File(resdir+'step_'+step[0]+'.h5','r')
        samp = np.array(list(h5file[u'Sample Set']))
        h5file2 =  h5py.File(resdir+'step_'+step[1]+'.h5','r')
        samp2 = np.array(list(h5file2[u'Sample Set']))
        
        if Np == float(samp2.shape[1]):  ## only one fault
            x00 = np.linspace(boundss[0],boundss[1],200)  # steps in strike
            x90 = np.linspace(boundds[0],boundds[1],2000)  # steps in dip
            n = norm(loc=0.,scale=boundss[1]) #pdf('Normal',x00,0,1)/2;
            u = uniform(boundds[0]+np.abs(boundds[0]-boundds[1])/200,boundds[1]-np.abs(boundds[0]-boundds[1])/200) #pdf('Uniform',x90,-0.5,60)/2.0165;
        
            p00 = n.pdf(x00)
            p90 = u.pdf(x90)
            p002 = n.pdf(x00[:-1])/2
            p902 = u.pdf(x90[:-1])
            
            postf00,postf90,postf002,postf902 = [], [], [], []
            moyds, moyss, medds, medss = [], [], [], []
            moyds2, moyss2, medds2, medss2 = [], [], [], []
            for l in range(nalpha[0]):
                for i in range(l*ns,l*ns+ns,1):
                    try:       # useful when strange fault geometries
                        postf90.append(np.histogram((samp[:,i+samp2.shape[1]//2]),bins=x90,density=True)[0])
                        postf00.append(np.histogram((samp[:,i]),bins=x00,density=True)[0])
                        moyds.append(np.mean(samp[:,i+samp2.shape[1]//2]))
                        medds.append(np.median(samp[:,i+samp2.shape[1]//2]))
                        moyss.append(np.mean(samp[:,i]))
                        medss.append(np.median(samp[:,i]))
                        postf902.append(np.histogram((samp2[:,i+samp2.shape[1]//2]),bins=x90,density=True)[0])
                        postf002.append(np.histogram((samp2[:,i]),bins=x90,density=True)[0])
                        moyds2.append(np.mean(samp2[:,i+samp2.shape[1]//2]))
                        medds2.append(np.median(samp2[:,i+samp2.shape[1]//2]))
                        moyss2.append(np.mean(samp2[:,i]))
                        medss2.append(np.median(samp2[:,i]))
                    except:
                        pass
            for l in range(nalpha[0],nalpha[0]+nalpha[1]*2-1,2):
                alpha=2
                for i in range(l*ns,l*ns+ns-1,alpha):
                    try:
                        aver = (samp[:,i+samp2.shape[1]//2] + samp[:,i+samp2.shape[1]//2+alpha-1] \
                                +samp[:,i+samp2.shape[1]//2+ns] + samp[:,i+samp2.shape[1]//2+ns+alpha-1] )/4
                        postf90.append(np.histogram((aver[:]),bins=x90,density=True)[0])
                        aver0 = (samp[:,i] + samp[:,i+alpha-1] \
                                + samp[:,i+ns] + samp[:,i+ns+alpha-1] )/4
                        postf00.append(np.histogram((aver0[:]),bins=x90,density=True)[0])
                        moyds.append(np.mean(aver[:]))
                        medds.append(np.median(aver[:]))
                        moyss.append(np.mean(aver0[:]))
                        medss.append(np.median(aver0[:]))
                        aver2 = (samp2[:,i+samp2.shape[1]//2] + samp2[:,i+samp2.shape[1]//2+alpha-1] \
                                +samp2[:,i+samp2.shape[1]//2+ns] + samp2[:,i+samp2.shape[1]//2+ns+alpha-1] )/4
                        postf902.append(np.histogram((aver2[:]),bins=x90,density=True)[0])
                        aver02 = (samp2[:,i] + samp2[:,i+alpha-1] \
                                + samp2[:,i+ns] + samp2[:,i+ns+alpha-1] )/4
                        postf002.append(np.histogram((aver02[:]),bins=x90,density=True)[0])
                        moyds2.append(np.mean(aver2[:]))
                        medds2.append(np.median(aver2[:]))
                        moyss2.append(np.mean(aver02[:]))
                        medss2.append(np.median(aver02[:]))
                    except:
                        pass
            moyds=np.array(moyds)
            medds=np.array(medds)
            moyss=np.array(moyss)
            medss=np.array(medss)
            moyds2=np.array(moyds2)
            medds2=np.array(medds2)
            moyss2=np.array(moyss2)
            medss2=np.array(medss2)
    
            postf90 = np.array(postf90)
            postf902 = np.array(postf902)
            x00 = np.linspace(boundss[0],boundss[1],199)  # steps in strike
            x90 = np.linspace(boundds[0],boundds[1],1999)  # steps in dip
            
            def kldiv(x,y):
                S=0
                for i in range(len(x)):
                    if x[i]!=0 and y[i]!=0:
                        s1 = x[i]*np.log(x[i]/y[i])
                    else:
                        s1=0
                    S=S+s1
                return S
            
            
            if entrel is True:
                alpha = 0.2
                P90= [alpha*np.asarray(p90[:-1]) + (1-alpha)*np.asarray(postf90[i,:]) for i in range(postf90.shape[0]) ]
                P902=[alpha*np.asarray(p90[:-1]) + (1-alpha)*np.asarray(postf902[i,:]) for i in range(postf902.shape[0]) ]
                ent = [kldiv( P90[k],P902[k]) for k in range(np.shape(P90)[0]) ]
                ent = np.array(ent)
                plotPDFds_moycomp([str(step[0][0:3]),str(step[1][0:3])], Np, ns, nd, nalpha, x90, postf90,postf902, medds,medds2,savedir,entrel=ent,color=color,valmax=valmax)
            else:
                plotPDFds_moycomp([step[0][0:3],step[1][0:3]], Np, ns, nd, nalpha, x90, postf90,postf902, moyds,moyds2,savedir,entrel=None,color=color,valmax=valmax)
    
        elif float(samp.shape[1]) > Np: #### 2 faults (co+post) 
            sampco = samp[:,0:Np]
            samppost = samp[:,Np:2*Np]
            sampco2 = samp2[:,0:Np]
            samppost2 = samp2[:,Np:2*Np]
            x00 = np.arange(-10,10.5,0.5)  # steps in strike
            x90 = np.arange(-2,202,1)  # steps in dip
            x90p = np.arange(-2,102,1)
            n = norm(loc=0.,scale=2.) #pdf('Normal',x00,0,1)/2;
            u = uniform(-1.,201.) #pdf('Uniform',x90,-0.5,60)/2.0165;
            up = uniform(-1.,101.)
            p00 = n.pdf(x00)
            p90 = u.pdf(x90)
            p002 = n.pdf(x00[:-1])/2
            p902 = u.pdf(x90[:-1])
            p902p = u.pdf(x90p[:-1])
            
            postco00,postco90,postpo00,postpo90 = [], [], [], []
            postco002,postco902,postpo002,postpo902 = [], [], [], []
            moyco00, moyco90, moypo00, moypo90 = [], [], [], []
            moyco002, moyco902, moypo002, moypo902 = [], [], [], []
            for l in range(nalpha[0]):
                for i in range(l*ns,l*ns+ns,1):
                    postco00.append(np.histogram((sampco[:,i]),bins=x00,density=True)[0])
                    postco90.append(np.histogram((-sampco[:,i+samp2.shape[1]//2]),bins=x90,density=True)[0])                
                    moyco90.append(np.mean(-sampco[:,i+samp2.shape[1]//2]))
                    moyco00.append(np.mean(sampco[:,i]))
                    postpo00.append(np.histogram((samppost[:,i]),bins=x90,density=True)[0])
                    postpo90.append(np.histogram((-samppost[:,i+samp2.shape[1]//2]),bins=x90,density=True)[0])                
                    moypo90.append(np.mean(-samppost[:,i+samp2.shape[1]//2]))
                    moypo00.append(np.mean(samppost[:,i]))
                    
                    postco002.append(np.histogram((sampco2[:,i]),bins=x00,density=True)[0])
                    postco902.append(np.histogram((-sampco2[:,i+samp2.shape[1]//2]),bins=x90,density=True)[0])                
                    moyco902.append(np.mean(-sampco2[:,i+samp2.shape[1]//2]))
                    moyco002.append(np.mean(sampco2[:,i]))
                    postpo002.append(np.histogram((samppost2[:,i]),bins=x90,density=True)[0])
                    postpo902.append(np.histogram((-samppost2[:,i+samp2.shape[1]//2]),bins=x90,density=True)[0])                
                    moypo902.append(np.mean(-samppost2[:,i+samp2.shape[1]//2]))
                    moypo002.append(np.mean(samppost2[:,i]))
            for l in range(nalpha[0],nalpha[0]+nalpha[1]*2-1,2):
                alpha=2
                for i in range(l*ns,l*ns+ns-1,alpha):
                    averco = (sampco[:,i+samp2.shape[1]//2] + sampco[:,i+samp2.shape[1]//2+alpha-1] \
                            +sampco[:,i+samp2.shape[1]//2+ns] + sampco[:,i+samp2.shape[1]//2+ns+alpha-1] )/4
                    postco90.append(np.histogram((-averco[:]),bins=x90,density=True)[0])
                    averco0 = (sampco[:,i] + sampco[:,i+alpha-1] \
                            + sampco[:,i+ns] + sampco[:,i+ns+alpha-1] )/4
                    postco00.append(np.histogram((averco0[:]),bins=x90,density=True)[0])
                    moyco90.append(np.mean(-averco[:]))
                    moyco00.append(np.mean(averco0[:]))
                    averpost = (samppost[:,i+samp2.shape[1]//2] + samppost[:,i+samp2.shape[1]//2+alpha-1] \
                            +samppost[:,i+samp2.shape[1]//2+ns] + samppost[:,i+samp2.shape[1]//2+ns+alpha-1] )/4
                    postpo90.append(np.histogram((-averpost[:]),bins=x90,density=True)[0])
                    averpost0 = (samppost[:,i] + samppost[:,i+alpha-1] \
                            + samppost[:,i+ns] + samppost[:,i+ns+alpha-1] )/4
                    postpo00.append(np.histogram((averpost0[:]),bins=x90,density=True)[0])
                    moypo90.append(np.mean(-averpost[:]))
                    moypo00.append(np.mean(averpost0[:]))
                    
                    averco2 = (sampco2[:,i+samp2.shape[1]//2] + sampco2[:,i+samp2.shape[1]//2+alpha-1] \
                            +sampco2[:,i+samp2.shape[1]//2+ns] + sampco2[:,i+samp2.shape[1]//2+ns+alpha-1] )/4
                    postco902.append(np.histogram((-averco2[:]),bins=x90,density=True)[0])
                    averco02 = (sampco2[:,i] + sampco2[:,i+alpha-1] \
                            + sampco2[:,i+ns] + sampco2[:,i+ns+alpha-1] )/4
                    postco002.append(np.histogram((averco02[:]),bins=x90,density=True)[0])
                    moyco902.append(np.mean(-averco2[:]))
                    moyco002.append(np.mean(averco02[:]))
                    averpost2 = (samppost2[:,i+samp2.shape[1]//2] + samppost2[:,i+samp2.shape[1]//2+alpha-1] \
                            +samppost2[:,i+samp2.shape[1]//2+ns] + samppost2[:,i+samp2.shape[1]//2+ns+alpha-1] )/4
                    postpo902.append(np.histogram((-averpost2[:]),bins=x90,density=True)[0])
                    averpost02 = (samppost2[:,i] + samppost2[:,i+alpha-1] \
                            + samppost2[:,i+ns] + samppost2[:,i+ns+alpha-1] )/4
                    postpo002.append(np.histogram((averpost02[:]),bins=x90,density=True)[0])
                    moypo902.append(np.mean(-averpost2[:]))
                    moypo002.append(np.mean(averpost02[:]))
            moyco00, moyco90, moypo00, moypo90 = np.array(moyco00), np.array(moyco90), np.array(moypo00), np.array(moypo90)
            moyco002, moyco902, moypo002, moypo902 = np.array(moyco002), np.array(moyco902), np.array(moypo002), np.array(moypo902)
            
            postco00, postco90, postpo00, postpo90 = np.array(postco00), np.array(postco90), np.array(postpo00), np.array(postpo90)
            postco002, postco902, postpo002, postpo902 = np.array(postco002), np.array(postco902), np.array(postpo002), np.array(postpo902)
            x00 = np.arange(-9.75,10.25,0.5) 
            x90 = np.arange(-1,202,1)
            x90p = np.arange(-1,102,1)
    #        plotPDFds_comp([step[0][0:3],step[1][0:3]+'_co'], Np, ns, nd, fault_type, x90, p902, postco90, postco902,moydsco,moydsco2, width = '',savedir=savedir,color=colorsco,valmax=valmax[0])
    #        plotPDFds_comp([step[0][0:3],step[1][0:3]+'_post'], Np, ns, nd, fault_type, x90p, p902p, postpo90,postpo902, moydspo, moydspo2, width = '',savedir=savedir,color=colorspo,valmax=valmax[1])
            plotPDFds_moycomp([step[0][0:3],step[1][0:3]+'_co'], Np, ns, nd, nalpha, x90, postco90,postco902, moyco90,moyco902,savedir,color=colorsco,valmax=150)

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
    return postf90, postf902

def PDFcomp(steps, fault_type, ns, nd, resdir, savedir):
    '''
    Plots last iteration PDFs for different results given in 'steps'
    
    IN ARGUMENT:
    You need to specify at least 5 arguments:
     --> steps = array: the results (steps) you want to plot
          ex: [063,050,052]
          
     --> fault_type = fault geometry type: 'classical' or 'optimized'

     --> ns = number of patches in strike: total for classical type, 
                 number of large patches for optimized
          
     --> nd = number of patches in dip: total for classical type, 
                 number of large patches for optimized
                 
    
    --> resdir= directory of sampfile
     --> savedir= where to save PDF's figures

    USAGE:
    >>> import altar_pdf
    >>> altar_pdf.PDFcomp([63,50,52],'optimized',7,4)

    '''
    
    tmean = []
    tsd = []
    x00 = np.arange(-10,10.5,0.5)  # steps in strike
    x90 = np.arange(-2,152,2)  # steps in dip
    n = norm(loc=0.,scale=2.) #pdf('Normal',x00,0,1)/2;
    u = uniform(-1.,151.) #pdf('Uniform',x90,-0.5,60)/2.0165;
    p00 = n.pdf(x00)
    p90 = u.pdf(x90)
    
    for s in range(len(steps)):
        
        if not os.path.isfile(os.path.join(savedir+'/comp/','pdf_'+str(steps[s])+'_ds.png')):
            data = open(resdir+'step_0'+str(steps[s])+'_log.txt','r').read()
            # Strings to find
            param = re.compile('_parameters: '+'\d+')
            for j in data.split("\n"):
                match = param.search(j)
                if match:
                    Np = int(match.group().split(' ')[1])
            
            
            a = steps[s]
            if type(steps[s]) is str:
                a = int(a[:-2])
            theta_mean = np.empty((a,Np))  
            theta_sd = np.empty((a,Np)) 
            it = np.empty(a)
            
            for i in range(a):
                it[i]= False
                if i == 0:
                    for l in data.split('\n'):
                        if 'Theta_mean: ' in l:
                            theta_mean[i,:] = [float(x) for x in data.split('\n')[data.split('\n').index(l)+1].split(' ')[:-1]]
                        if 'Theta_sd: ' in l:
                            theta_sd[i,:] = [float(x) for x in data.split('\n')[data.split('\n').index(l)+1].split(' ')[:-1]]
        
                else:
                    for j in range(len(data.split('\n'))):
                        bla = '--       iteration: '+str(i)+' '
                        if bla in data.split('\n')[j]:
                            it[i] = True
                        if 'Theta_mean: ' in data.split('\n')[j] and it[i] == True:
                            theta_mean[i,:] = [float(x) for x in data.split('\n')[j+1].split(' ')[:-1]]
                        if 'Theta_sd: ' in data.split('\n')[j] and it[i] == True:
                            theta_sd[i,:] = [float(x) for x in data.split('\n')[j+1].split(' ')[:-1]]             
                            break
            tmean.append(theta_mean[-1,:])
            tsd.append(theta_sd[-1,:])
            
            slp2=tmean[s][Np//2:Np]
            nstrike=7
            ndip=4
            length=28000
            width=16000
            plotSlip('0'+str(steps[s]), nstrike, ndip, length, width, slip=slp2)
            
            post00 = np.zeros((Np//2,len(x00)))
            post90 = np.zeros((Np//2,len(x90)))
            
            for j in range(Np//2):
                n1 = norm(loc=tmean[s][j],scale=tsd[s][j])
                n2 = norm(loc=-tmean[s][j+Np//2],scale=tsd[s][j+Np//2]) # means are negative and I want it positive
                post00[j,:] = n1.pdf(x00)
                post90[j,:] = n2.pdf(x90)
            plotPDFss(steps[s], Np, ns, nd, fault_type, x00, p00, post00, moy= tmean[s][0:Np//2], width= '16 km',ent=None, savedir=savedir+'/comp/')
            plotPDFds(steps[s], Np, ns, nd, fault_type, x90, p90, post90, moy= tmean[s][Np//2:Np], width= '16 km',ent=None, savedir=savedir+'/comp/')
    
    if len(steps)==2:
        script1 = """
        cd {dir1}comp
        convert {f1} -transparent white {f1}
        convert {f2} -transparent white {f2}
        composite -dissolve 150  {f1} {f2} {f3}
        """.format(dir1=savedir, f1='pdf_'+str(steps[0])+'_ds.png', f2='pdf_'+str(steps[1])+'_ds.png', f3='pdfcomp'+str(steps[0])+'-'+str(steps[1])+'.png')
        subprocess.call(script1, shell=True)
    elif len(steps)==3:
        script2 = """
        cd {dir1}comp
        convert {f1} -transparent white {f1}
        convert {f2} -transparent white {f2}
        convert {f3} -transparent white {f3}
        composite -dissolve 135  {f1} {f2} {f4}
        composite -dissolve 170  {f4} {f3} {f5}
        rm -f {f4}
        """.format(dir1=savedir, f1='pdf_'+str(steps[0])+'_ds.png', f2='pdf_'+str(steps[1])+'_ds.png', f3='pdf_'+str(steps[2])+'_ds.png', f4='img2.png', f5='pdfcomp'+str(steps[0])+'-'+str(steps[1])+'-'+str(steps[2])+'.png')
        subprocess.call(script2, shell=True)
    else:
        print('This script is not written to plot more than 3 steps! You need to modify it!')
    
    return()

def PDFall(filename, ite, fault_type, ns, nd, resdir, savedir):
    
    '''
    Plots PDFs for each iteration of Altar inversion, get input from log
    
    IN ARGUMENT:
    You need to specify at least 5 arguments:
     --> filename
    -- > ite = number of iterations
     --> fault_type = fault geometry type: 'classical' or 'optimized'

     --> ns = number of patches in strike: total for classical type, 
                 number of large patches for optimized
          
     --> nd = number of patches in dip: total for classical type, 
                 number of large patches for optimized
                 
     --> Np = number of parameters
    
    --> resdir= directory of sampfile
     --> savedir= where to save PDF's figures

    USAGE:
    >>> import altar_pdf
    >>> altar_pdf.PDFall(59,'optimized',7,4)

    '''

    # ---------------------------------------------------------------------------
    # Extract data from file

    data = open(resdir+filename+'_log.txt','r').read()
    
    # Strings to find
    param = re.compile('_parameters: '+'\d+')
    for j in data.split("\n"):
        match = param.search(j)
        if match:
            Np = int(match.group().split(' ')[1])
    
    theta_mean = np.empty((ite,Np))  
    theta_sd = np.empty((ite,Np)) 
    it = np.empty(ite)
    
    for i in range(ite):
        it[i]= False
        if i == 0:
            for l in data.split('\n'):
                if 'Theta_mean: ' in l:
                    theta_mean[i,:] = [float(s) for s in data.split('\n')[data.split('\n').index(l)+1].split(' ')[:-1]]
                if 'Theta_sd: ' in l:
                    theta_sd[i,:] = [float(s) for s in data.split('\n')[data.split('\n').index(l)+1].split(' ')[:-1]]

        else:
            for j in range(len(data.split('\n'))):
                bla = '--       iteration: '+str(i)+' '
                if bla in data.split('\n')[j]:
                    it[i] = True
                if 'Theta_mean: ' in data.split('\n')[j] and it[i] == True:
                    theta_mean[i,:] = [float(s) for s in data.split('\n')[j+1].split(' ')[:-1]]
                if 'Theta_sd: ' in data.split('\n')[j] and it[i] == True:
                    theta_sd[i,:] = [float(s) for s in data.split('\n')[j+1].split(' ')[:-1]]             
                    break
                
                
    # ---------------------------------------------------------------------------
    x00 = np.arange(-10,10.5,0.5)  # steps in strike
    x90 = np.arange(-2,152,2)  # steps in dip
    n = norm(loc=0.,scale=2.) #pdf('Normal',x00,0,1)/2;
    u = uniform(-1.,151.) #pdf('Uniform',x90,-0.5,60)/2.0165;
    p00 = n.pdf(x00)
    p90 = u.pdf(x90)
    

    script1 = """
    cd {dir1}
    mkdir {dir2}
    """.format(dir1=savedir, dir2=filename+'_gif')
    subprocess.call(script1, shell=True)
    
    for i in range(ite):
        post00 = np.zeros((Np//2,len(x00)))
        post90 = np.zeros((Np//2,len(x90)))
        
        for j in range(Np//2):
            n1 = norm(loc=theta_mean[i,j],scale=theta_sd[i,j])
            n2 = norm(loc=-theta_mean[i,j+Np//2],scale=theta_sd[i,j+Np//2]) # means are negative and I want it positive
            post00[j,:] = n1.pdf(x00)
            post90[j,:] = n2.pdf(x90)
        if i < 10:
            plotPDFss('0'+str(i), Np, ns, nd, fault_type, x00, p00, post00, moy= theta_mean[i,0:Np//2], width= '16 km',ent=None, savedir=savedir+filename+'_gif/')
            plotPDFds('0'+str(i), Np, ns, nd, fault_type, x90, p90, post90, moy= theta_mean[i,Np//2:Np], width= '16 km',ent=None, savedir=savedir+filename+'_gif/')
        else:
            plotPDFss(i, Np, ns, nd, fault_type, x00, p00, post00, moy= theta_mean[i,0:Np//2], width= '16 km',ent=None, savedir=savedir+filename+'_gif/')
            plotPDFds(i, Np, ns, nd, fault_type, x90, p90, post90, moy= theta_mean[i,Np//2:Np], width= '16 km',ent=None, savedir=savedir+filename+'_gif/')
    
    script1 = """
    cd {dir1}/{dir2}
    convert -delay 30 -loop 0 *_ss.png {of}
    convert -delay 30 -loop 0 *_ds.png {of2}
    """.format(dir1=savedir, dir2=filename+'_gif', of=filename+'_ss.gif',of2=filename+'_ds.gif')
    subprocess.call(script1, shell=True)
    
    return()


def plotPDFss(filename,Np, ns, nd, x00, p00, post00, moy, width, ent, savedir,color=colorsco,valmax=150,fault_type='classical'):
    try:
        cmap.create_cmap(color, 'cptslip')
    
        if fault_type == 'classical':
            #---------------------------------------------------
            #                FOR CLASSICAL FAULT
            #              (comment or uncomment)
            #---------------------------------------------------
            
            ##--------- STRIKE SLIP ----------
            jet = cm = plt.get_cmap('cptslip') 
            cNorm  = colors.Normalize(min(x00),valmax)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='cptslip')
            scalarMap.set_array([])
            
            fig2=plt.figure(2, figsize=((5/1.5)*ns,(2/1.5)*(nd+1)))
            #colorscale
            a=plt.subplot2grid((nd+2,ns+1), (nd+1,ns-2), colspan=3, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            colo=plt.colorbar(scalarMap,orientation='horizontal',fraction=0.3)
            colo.set_ticks([0,valmax/2,valmax])
            ax=colo.ax
            ax.text(-0.1,0.5,'cm',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
            j=ns-1
            c=1
            l=0
            for i in range(Np//2): 
                if i % ns == 0:
                    #j=ns-1
                    l=l+1
                    c=1
                a = plt.subplot2grid((nd+2,ns+1), (l,c), colspan=1, rowspan=1)
                #j=j-1
                c=c+1
    #            plt.fill(x00, p00,color='lavender',zorder=1)
                colorval = scalarMap.to_rgba(abs(moy[i]))
#                plt.plot(x00,8*post00[i,:], color=colorval, lw=1.,zorder=2)
                if ent==None:
                    alph=1
                else:
                    alph = 0.2 + abs(1-ent[i])*0.8
                plt.fill_between(x00,12*post00[i,:], color=colorval,lw=0.1,zorder=2, alpha=alph)
                plt.plot(x00,12*post00[i,:], color='gray', lw=0.1,zorder=3)
                plt.xlim(min(x00)-1,max(x00)+1)
                plt.ylim(0,np.amax(8*post00)/3)
                plt.xticks([min(x00),(max(x00)-min(x00))/2,max(x00)])
                plt.yticks([])
                a.spines["top"].set_visible(False)   
                a.spines["right"].set_visible(False)  
                a.spines["left"].set_visible(False)
                a.xaxis.set_ticks_position('bottom')

                
            plt.savefig(savedir+'pdf_'+filename+'_ss.pdf',format='pdf',bbox_inches="tight", transparent=True)
#            plt.savefig(savedir+'pdf_'+filename+'_ss.png', format='png', dpi=300,pad_inches=0.0)
#            plt.show()
                    
        if fault_type == 'optimized':
            #---------------------------------------------------
            #                FOR OPTIMIZED FAULT
            #              (comment or uncomment)
            #---------------------------------------------------
            
            #--------- STRIKE SLIP ----------
            jet = cm = plt.get_cmap('cptslip') 
            cNorm  = colors.Normalize(vmin=-10, vmax=10)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='cptslip')
            scalarMap.set_array([])
            
            fig2=plt.figure(2, figsize=(ns*4,nd-2+5))
            
            j=2
            k=5
            c=1
            c2=1
            
            for i in range(Np//2): 
                if i < ns*4: # first line
                    a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (1,i+1), colspan=1, rowspan=1)
                elif ns*4 <= i <= ns*4*2-1:
                    a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (2,i-ns*4+1), colspan=1, rowspan=1)
                elif ns*4*2 <= i <= 14*ns-1: # third line
                    if i % (2*ns) == 0:
                        j=j+1
                        c=1
                    a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (j,c), colspan=1, rowspan=1)
                    c=c+2
                else: # sixth line
                    if i % ns == 0:
                        k=k+1
                        c2=1
                    a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (k,c2), colspan=1, rowspan=1)
                    c2=c2+4
                a.spines["top"].set_visible(False)   
                a.spines["right"].set_visible(False)  
                a.spines["left"].set_visible(False)
                plt.tick_params(axis="both", which="both", bottom="on", top="off",  
                            labelbottom="off", left="off", right="off", labelleft="off") 
                if max(range(Np//2))-ns+1 <= i <= max(range(Np//2)):
                    a.get_xaxis().tick_bottom()
                plt.fill(x00, p00,color='lavender',zorder=1)
                colorval = scalarMap.to_rgba(abs(moy[i]))
                plt.plot(x00+0.5,post00[i,:], color=colorval, lw=2.,zorder=2)
                if ent==None:
                    alph=1
                else:
                    alph = 0.2 + abs(1-ent[i])*0.8
                plt.fill_between(x00+0.5,post00[i,:], color=colorval,zorder=3, alpha=alph)
                plt.xlim(min(x00)-1,max(x00)+1)
                plt.ylim(0,1/1.8)
                plt.xticks([min(x00),0,max(x00)])
                plt.yticks([])
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (nd-2+3+2,ns*4-1), colspan=2, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            colo=plt.colorbar(scalarMap,orientation='horizontal',fraction=0.3)
            colo.set_ticks([-10,0,10])
            ax=colo.ax
            ax.text(-0.1,0.5,'cm',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
            #text
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (0,1), colspan=1, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            a.text(0.5,0.2,'N',fontweight='bold',horizontalalignment='center',verticalalignment='center')
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (0,ns*4), colspan=1, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            a.text(0.5,0.2,'S',fontweight='bold',horizontalalignment='center',verticalalignment='center')
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (1,0), colspan=1, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            a.text(0.5,0.1,'0 km',fontweight='bold',horizontalalignment='center',verticalalignment='center')
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (nd-2+3+2,0), colspan=1, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            a.text(0.5,0.1,width,fontweight='bold',horizontalalignment='center',verticalalignment='center')
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (0,int((ns*4+1)/2)-1), colspan=5, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (0,int((ns*4+1)/2)+int((ns*4+1)/4)), colspan=3, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            a.text(0.5,0.1,'Sampled models: 420000',color='silver',style='italic',horizontalalignment='center',verticalalignment='center')
            #fig2.subplots_adjust(hspace=0.2) 
            #plt.tight_layout()
            plt.savefig(savedir+'pdf_'+filename+'_ss.png', format='png', dpi=300,pad_inches=0.0)
            plt.show()
            
            script1 = """
            cd {}
            convert {} -trim -bordercolor White -border 10x10 +repage {}
            """.format(savedir, 'pdf_'+filename+'_ss.png', 'pdf_'+filename+'_ss.png')
            subprocess.call(script1, shell=True)
            
    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)        
    return()
        
        
def plotPDFds(filename, Np, ns, nd, x90, p90, post90, moy, width, ent, savedir,color=colorsco,valmax=150,fault_type='classical'):
    
    try:
        #---------------------------------------------------
        #                     P L O T
        #---------------------------------------------------
        #moy = np.mean( np.array([ i for i in samp2[:,:]]), axis=0 ) # mean for colorscale
        
        #gmtcpt('/u/moana/user/ragon/fig/amatrice/gmt/slip.cpt', name='cptslip')
        if color=='po':
            color=colorspo
        elif color=='co':
            color=colorsco
        cmap.create_cmap(color, 'cptslip')
        
        if fault_type == 'classical':
            #---------------------------------------------------
            #                FOR CLASSICAL FAULT
            #              (comment or uncomment)
            #---------------------------------------------------
            
            #--------- DIP SLIP ----------
            cm = colors.LinearSegmentedColormap.from_list('cptslip',color, N=256)
            cNorm = MidpointNormalize(vmin=0., vcenter=valmax, vmax=valmax+5.)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
            scalarMap.set_array([])
            
            plt.figure(3, figsize=(2.7*ns,2.8*(nd+1)))
            
            #colorbar
            a=plt.subplot2grid((nd+2,ns+1), (nd+1,ns-2), colspan=3, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            colo=plt.colorbar(scalarMap,orientation='horizontal',fraction=0.3)
            colo.set_ticks([0,valmax/2,valmax])
    #        colo.set_ticks([10,50,100,150,200])
            ax=colo.ax
            ax.text(-0.1,0.5,'cm',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
            
            c=1
            l=0
            for i in range(Np//2): 
                if i % ns == 0:
                    #j=ns-1
                    l=l+1
                    c=1
                a = plt.subplot2grid((nd+2,ns+1), (l,c), colspan=1, rowspan=1)
                #j=j-1
                c=c+1
                if max(range(Np//2))-ns <= i <= max(range(Np//2)) and i%2==0:
                    a.get_xaxis().tick_bottom()
    #            plt.fill(x90, p90,color='lavender',zorder=1)
                colorval = scalarMap.to_rgba(abs(moy[i]))
    #            plt.plot(x90+0.5,6*post90[i,:], color=colorval, lw=2.,zorder=2)
                if ent==None:
                    alph=1
                else:
                    alph = 0.2 + abs(1-ent[i])*0.8
                plt.fill_between(x90,6*post90[i,:], color=colorval,lw=0.1,zorder=2, alpha=alph)
                plt.plot(x90,6*post90[i,:], color='gray', lw=0.1,zorder=3)
                plt.xlim(min(x90)-10,max(x90)+10)
                plt.ylim(0,0.05)
                plt.xticks([0,int(max(x90)/2),max(x90)])
                plt.yticks([]) 
                a.spines["top"].set_visible(False)   
                a.spines["right"].set_visible(False)  
                a.spines["left"].set_visible(False)
                a.xaxis.set_ticks_position('bottom')

            #fig2.subplots_adjust(hspace=0.2) 
            #plt.tight_layout()
#            plt.savefig(savedir+'pdf_'+filename+'_ds.png', format='png', dpi=300,pad_inches=0.0)
            plt.savefig(savedir+'pdf_'+filename+'_ds.pdf', format='pdf', pad_inches=0.0)
#            plt.show()
            
        
        if fault_type == 'optimized':
            #---------------------------------------------------
            #                FOR OPTIMIZED FAULT
            #              (comment or uncomment)
            #---------------------------------------------------
            
            #--------- DIP SLIP ----------
            jet = cm = plt.get_cmap('cptslip') 
            cNorm  = colors.Normalize(vmin=0, vmax=150)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='cptslip')
            scalarMap.set_array([])
            
            plt.figure(3, figsize=(ns*4,nd-2+5))
            
            
            j=2
            k=5
            c=1
            c2=1
            
            for i in range(Np//2): 
                if i < ns*4: # first line
                    a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (1,i+1), colspan=1, rowspan=1)
                elif ns*4 <= i <= ns*4*2-1:
                    a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (2,i-ns*4+1), colspan=1, rowspan=1)
                elif ns*4*2 <= i <= 14*ns-1: # third line
                    if i % (2*ns) == 0:
                        j=j+1
                        c=1
                    a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (j,c), colspan=1, rowspan=1)
                    c=c+2
                else: # sixth line
                    if i % ns == 0:
                        k=k+1
                        c2=1
                    a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (k,c2), colspan=1, rowspan=1)
                    c2=c2+4
                a.spines["top"].set_visible(False)   
                a.spines["right"].set_visible(False)  
                a.spines["left"].set_visible(False)
                plt.tick_params(axis="both", which="both", bottom="on", top="off",  
                            labelbottom="off", left="off", right="off", labelleft="off") 
                if max(range(Np//2))-ns+1 <= i <= max(range(Np//2)):
                    a.get_xaxis().tick_bottom()
                plt.fill(x90, p90,color='lavender',zorder=1)
                colorval = scalarMap.to_rgba(abs(moy[i]))
                plt.plot(x90+0.5,post90[i,:], color=colorval, lw=2.,zorder=2)
                if ent==None:
                    alph=1
                else:
                    alph = 0.2 + abs(1-ent[i])*0.8
                plt.fill_between(x90+0.5,post90[i,:], color=colorval,zorder=3, alpha=alph)
                plt.xlim(min(x90)-10,max(x90)+10)
                plt.ylim(0,0.13)
                plt.xticks([0,int(max(x90)/2),max(x90)])
                plt.yticks([]) 
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (nd-2+3+2,ns*4-1), colspan=2, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            colo=plt.colorbar(scalarMap,orientation='horizontal',fraction=0.3)
            colo.set_ticks([10,50,90,130])
            ax=colo.ax
            ax.text(-0.1,0.5,'cm',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
            #text
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (0,1), colspan=1, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            a.text(0.5,0.2,'N',fontweight='bold',horizontalalignment='center',verticalalignment='center')
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (0,ns*4), colspan=1, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            a.text(0.5,0.2,'S',fontweight='bold',horizontalalignment='center',verticalalignment='center')
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (1,0), colspan=1, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            a.text(0.5,0.1,'0 km',fontweight='bold',horizontalalignment='center',verticalalignment='center')
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (nd-2+3+2,0), colspan=1, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            a.text(0.5,0.1,width,fontweight='bold',horizontalalignment='center',verticalalignment='center')
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (0,int((ns*4+1)/2)-1), colspan=5, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            # a.text(0.5,0.1,'D I P - S L I P   P D F s'+' '*10+'--'+' '*10+'S T E P :   '+filename,fontweight='bold',horizontalalignment='center',verticalalignment='center')
            a=plt.subplot2grid((nd-2+3+2+1,ns*4+1), (0,int((ns*4+1)/2)+int((ns*4+1)/4)), colspan=3, rowspan=1)
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            a.text(0.5,0.1,'Sampled models: 420000',color='silver',style='italic',horizontalalignment='center',verticalalignment='center')
            #fig2.subplots_adjust(hspace=0.2) 
            #plt.tight_layout()
            plt.savefig(savedir+'pdf_'+filename+'_ds.png', format='png', dpi=300,pad_inches=0.0)
            plt.show()
            
            script1 = """
            cd {}
            convert {} -trim -bordercolor White -border 10x10 +repage {}
            """.format(savedir, 'pdf_'+filename+'_ds.png', 'pdf_'+filename+'_ds.png')
            subprocess.call(script1, shell=True)

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    
    return()

def plotPDFds_moy(filename, Np, ns, nd, nalpha, x90, post90, moy,savedir,color=colorsco,valmax=150):
    '''
    nalpha: number of lines for each alpha type of moy, e.g. 6 lines with no filter, 2 lines with alpha =2, 1 line with alpha=3
    nalpha=[n1,n2,n3]
    '''

    #---------------------------------------------------
    #                     P L O T
    #---------------------------------------------------

    if color=='po':
        color=colorspo
    elif color=='co':
        color=colorsco
    cmap.create_cmap(color, 'cptslip')
    
    #--------- DIP SLIP ----------
    jet = cm = plt.get_cmap('cptslip') 
    cNorm  = colors.Normalize(vmin=0, vmax=valmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='cptslip')
    scalarMap.set_array([])
    
    plt.figure(3, figsize=(ns,nd+1))
    
    #colorbar
    a=plt.subplot2grid(((nd*2)+2,ns*2+1), ((nd*2),ns*2+1-6), colspan=6, rowspan=2)
    a.spines["top"].set_visible(False)   
    a.spines["right"].set_visible(False)  
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                labelbottom="off", left="off", right="off", labelleft="off") 
    colo=plt.colorbar(scalarMap,orientation='horizontal',fraction=0.3)
    colo.set_ticks([0,valmax/2,valmax])
#        colo.set_ticks([10,50,100,150,200])
    ax=colo.ax
    ax.text(-0.1,0.5,'cm',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    
    c=1
    l=1
    for i in range(len(post90)):
        if 1 <= l < nalpha[0]*2-1:
            a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
            c=c+2
            if (c-1) % (2*ns) == 0:
                l=l+2
                c=1
        elif l==nalpha[0]*2-1:
            a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
            c=c+2
            if (c-1) % (2*ns) == 0:
                l=l+3
                c=2
        elif nalpha[0]*2 <= l < (nalpha[0]+nalpha[1]*2)*2-2:
            a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
            c=c+4
            if (c-2) % (2*ns) == 0:
                l=l+4
                c=2
        elif l==(nalpha[0]+nalpha[1]*2)*2-2:
            a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
            c=c+4
            if (c-2) % (2*ns) == 0:
                l=l+5
                c=3
        elif (nalpha[0]+nalpha[1]*2)*2 <= l <= nd*2:
            a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
            c=c+6
            if (c-6) % (2*ns) == 0:
                l=l+6
                c=3
                
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="on", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        if i >= len(post90) - ns/len(nalpha):
            a.get_xaxis().tick_bottom()
        colorval = scalarMap.to_rgba(abs(moy[i]))
        plt.plot(x90,3*post90[i,:], color=colorval, lw=2.,zorder=2)
        plt.fill_between(x90,3*post90[i,:], color=colorval,zorder=3)
        plt.xlim(min(x90)-10,max(x90)+10)
        plt.ylim(0,0.13)
        plt.xticks([0,int(max(x90)/2),max(x90)])
        plt.yticks([]) 
        
    #fig2.subplots_adjust(hspace=0.2) 
    #plt.tight_layout()
    plt.savefig(savedir+'pdf_'+filename+'_ds.png', format='png', dpi=300,pad_inches=0.0)
    plt.savefig(savedir+'pdf_'+filename+'_ds.pdf', format='pdf',pad_inches=0.0)
    plt.show()
    
    script1 = """
    cd {}
    convert {} -trim -bordercolor White -border 10x10 +repage {}
    """.format(savedir, 'pdf_'+filename+'_ds.png', 'pdf_'+filename+'_ds.png')
    subprocess.call(script1, shell=True)
        
    return()
        
def plotPDFds_moycomp(filename, Np, ns, nd, nalpha, x90, post90,post902, moy,moy2,savedir,entrel=None,color=colorsco,valmax=150):

    '''
    filename is an array!

    nalpha: number of lines for each alpha type of moy, e.g. 6 lines with no filter, 2 lines with alpha =2, 1 line with alpha=3
    nalpha=[n1,n2,n3]
    '''
    try:
        #---------------------------------------------------
        #                     P L O T
        #---------------------------------------------------
    
        if color=='po':
            color=colorspo
        elif color=='co':
            color=colorsco
        cmap.create_cmap(color, 'cptslip')
        cmape = sns.cubehelix_palette(rot=-.4, light=0.98,as_cmap=True)
        
        #--------- DIP SLIP ----------
        jet = cm = plt.get_cmap('cptslip') 
        cNorm  = colors.Normalize(vmin=0, vmax=valmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='cptslip')
        scalarMap.set_array([])
        
        if 'co' in filename[1]:
                cNorm  = colors.Normalize(0,80)
        elif 'po' in filename[1]:
            cNorm  = colors.Normalize(0,15)
        else:
            cNorm  = colors.Normalize(0,80)
    #    scalarMap2 = cmx.ScalarMappable(norm=cNorm, cmap=cmape)
    #    scalarMap2.set_array([])    
        graycmap = plt.get_cmap('gist_yarg') 
        cNorm  = colors.Normalize(vmin=0, vmax=100)
        scalarMap2 = cmx.ScalarMappable(norm=cNorm, cmap='gist_yarg')
        scalarMap2.set_array([])
        
        plt.figure(3, figsize=(ns/1.4,(nd+1)/1.4))
        
        #colorbar
        a=plt.subplot2grid(((nd*2)+2,ns*2+1), ((nd*2),ns*2+1-6), colspan=6, rowspan=2)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        colo=plt.colorbar(scalarMap,orientation='horizontal',fraction=0.3)
        colo.set_ticks([0,valmax/2,valmax])
    #        colo.set_ticks([10,50,100,150,200])
        ax=colo.ax
        ax.text(-0.1,0.5,'cm',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
        
        c=1
        l=1
        for i in range(len(post90)):
            if 1 <= l < nalpha[0]*2-1:
                a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
                c=c+2
                if (c-1) % (2*ns) == 0:
                    l=l+2
                    c=1
            elif l==nalpha[0]*2-1:
                a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
                c=c+2
                if (c-1) % (2*ns) == 0:
                    l=l+3
                    c=2
            elif nalpha[0]*2 <= l < (nalpha[0]+nalpha[1]*2)*2-2:
                a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
                c=c+4
                if (c-2) % (2*ns) == 0:
                    l=l+4
                    c=2
            elif l==(nalpha[0]+nalpha[1]*2)*2-2:
                a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
                c=c+4
                if (c-2) % (2*ns) == 0:
                    l=l+5
                    c=3
            elif (nalpha[0]+nalpha[1]*2)*2 <= l <= nd*2:
                a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
                c=c+6
                if (c-6) % (2*ns) == 0:
                    l=l+6
                    c=3
#            if 1 <= l < nalpha[0]*2-1:
#                a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
#                c=c+2
#                if (c-1) % (2*ns) == 0:
#                    l=l+2
#                    c=1
#            elif l==nalpha[0]*2-1:
#                a = plt.subplot2grid(((nd*2)+2,ns*2+1), (l,c), colspan=2, rowspan=2)
#                c=c+2
#                if (c-1) % (2*ns) == 0:
#                    l=l+2
#                    c=1
#            elif nalpha[0]*2 <= l < (nalpha[0]+nalpha[1]*2)*2-2:
#                a = plt.subplot2grid(((nd*2)+1,ns*2), (l,c), colspan=4, rowspan=4)
#                c=c+4
#                if (c-2) % (2*ns) == 0:
#                    l=l+4
#                    c=2
#            elif l==(nalpha[0]+nalpha[1]*2)*2-2:
#                a = plt.subplot2grid(((nd*2)+1,ns*2), (l,c), colspan=4, rowspan=4)
#                c=c+4
#                if (c-2) % (2*ns) == 0:
#                    l=l+5
#                    c=3
#            elif (nalpha[0]+nalpha[1]*2)*2 <= l <= nd*2:
#                a = plt.subplot2grid(((nd*2)+1,ns*2), (l,c), colspan=4, rowspan=4)
#                c=c+6
#                if (c-6) % (2*ns) == 0:
#                    l=l+6
#                    c=3
                    
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="on", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            if entrel is not None:
                colorval = scalarMap2.to_rgba(entrel[i]*100)
            else:
                if i >= len(post90) - ns/len(nalpha):
                    a.get_xaxis().tick_bottom()
                elif max(abs(moy2[i]),abs(moy[i])) >= valmax/30:
                    percent = (abs( moy2[i] - moy[i] )/max(abs(moy2[i]),abs(moy[i])))*100
                else:
                    percent = 0
                colorval = scalarMap2.to_rgba(percent)    
    #            plt.fill_between(x90+0.5,4*np.amax(post902), color=colorval,linewidth=.5,zorder=1)
            colorval2 = scalarMap.to_rgba(abs(moy[i]))
            colorval3 = scalarMap.to_rgba(abs(moy2[i]))
    #        colormod2 = (0.8784,0.8784,0.8784,1)
    #        colormod1 = (0.7019,0.7019,0.7019,1)
    #        plt.vlines(abs(moy[i]),0, 4*np.amax(post90), color=colorval2,linestyle='--',linewidth=2,zorder=1)
    #        plt.vlines(abs(moy2[i]),0, 4*np.amax(post90), color=colorval3,linestyle='--',linewidth=2,zorder=2)
            plt.fill_between(x90+0.5,4*post902[i,:], color=colorval3,linewidth=.0,zorder=3)
            plt.fill_between(x90+0.5,4*post90[i,:], color=colorval2,linewidth=.0,zorder=4)  #(0.658,0.682,0.690,1)
    #        plt.fill_between(x90+0.5,4*post902[i,:],4*post90[i,:], color=(0.901,0.917,0.921,1),linewidth=.5,zorder=5)
    #        plt.plot(x90+0.5,4*post90[i,:], color=colormod1, lw=1,zorder=5)
    #        plt.plot(x90+0.5,4*post902[i,:], color=colormod2,lw=1,zorder=6)
            if entrel is not None and max(abs(moy2[i]),abs(moy[i]))>50:
                plt.text(valmax/25, 0.06, np.str(np.int(entrel[i]*100))+'%', weight = 'bold', color=colorval, fontsize=11,zorder=5)           
            else:
                plt.text(valmax/25, 0.06, np.str(np.int(percent))+'%', weight = 'bold', color=colorval, fontsize=11,zorder=5)
            plt.xlim(min(x90)-10,max(x90)+10)
            plt.ylim(0,0.13)
            plt.xticks([0,np.int(max(x90)/2),max(x90)])
            plt.yticks([]) 
        #fig2.subplots_adjust(hspace=0.2) 
        #plt.tight_layout()
        plt.savefig(savedir+'pdfcomp_'+str(filename[0])+'_'+str(filename[1])+'_ds.png', format='png', dpi=300,pad_inches=0.0)
        plt.savefig(savedir+'pdfcomp_'+str(filename[0])+'_'+str(filename[1])+'_ds.pdf', format='pdf',pad_inches=0.0)
        plt.show()
        
        script1 = """
        cd {}
        convert {} -trim -bordercolor White -border 10x10 +repage {}
        """.format(savedir, 'pdf_'+filename+'_ds.png', 'pdf_'+filename+'_ds.png')
        subprocess.call(script1, shell=True)

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
       
    return()
        
def plotPDFds_comp(filename, Np, ns, nd, fault_type, x90, p90, post90,post902, moy,moy2, width, savedir,color=colorsco,valmax=150):

    #---------------------------------------------------
    #                     P L O T
    #---------------------------------------------------
    #moy = np.mean( np.array([ i for i in samp2[:,:]]), axis=0 ) # mean for colorscale
     
    #gmtcpt('/u/moana/user/ragon/fig/amatrice/gmt/slip.cpt', name='cptslip')
    if color=='po':
        color=colorspo
    elif color=='co':
        color=colorsco
    cmap.create_cmap(color, 'cptslip')
    cmape = sns.cubehelix_palette(rot=-.4, light=0.98,as_cmap=True)

        
    if fault_type == 'classical':
        #---------------------------------------------------
        #                FOR CLASSICAL FAULT
        #---------------------------------------------------
        
        #--------- DIP SLIP ----------
        jet = cm = plt.get_cmap('cptslip')
        cNorm  = colors.Normalize(0,valmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='cptslip')
        scalarMap.set_array([])
        
        if 'co' in filename[1]:
            cNorm  = colors.Normalize(0,80)
        elif 'po' in filename[1]:
            cNorm  = colors.Normalize(0,15)
        else:
            cNorm  = colors.Normalize(0,80)
        scalarMap2 = cmx.ScalarMappable(norm=cNorm, cmap=cmape)
        scalarMap2.set_array([])
        
        plt.figure(3, figsize=(ns,nd+1))
        
        #colorbar
        a=plt.subplot2grid((nd+2,ns+1), (nd+1,ns-2), colspan=3, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        colo=plt.colorbar(scalarMap,orientation='horizontal',fraction=0.3)
        colo.set_ticks([0,valmax/2,valmax])
#        colo.set_ticks([10,50,100,150,200])
        ax=colo.ax
        ax.text(-0.1,0.5,'cm',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
        
        a=plt.subplot2grid((nd+2,ns+1), (nd+1,ns-6), colspan=3, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        colo=plt.colorbar(scalarMap2,orientation='horizontal',fraction=0.3)
        colo.set_ticks([0,40,80])
#        colo.set_ticks([10,50,100,150,200])
        ax=colo.ax
        ax.text(-0.1,0.5,'%',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
        
        c=1
        l=0
        for i in range(Np//2): 
            if i % ns == 0:
                #j=ns-1
                l=l+1
                c=1
            a = plt.subplot2grid((nd+2,ns+1), (l,c), colspan=1, rowspan=1)
            #j=j-1
            c=c+1
            a.spines["top"].set_visible(False)   
            a.spines["right"].set_visible(False)  
            a.spines["left"].set_visible(False)
            plt.tick_params(axis="both", which="both", bottom="on", top="off",  
                        labelbottom="off", left="off", right="off", labelleft="off") 
            if max(range(Np//2))-ns <= i <= max(range(Np//2)) and i%2==0:
                a.get_xaxis().tick_bottom()
            
            if max(abs(moy2[i]),abs(moy[i])) >= 20:
                percent = (abs( moy2[i] - moy[i] )/max(abs(moy2[i]),abs(moy[i])))*100
            else:
                percent = 0
            colorval = scalarMap2.to_rgba(percent)    
#            plt.fill_between(x90+0.5,4*np.amax(post902), color=colorval,linewidth=.5,zorder=1)
            colorval2 = scalarMap.to_rgba(abs(moy[i]))
            colorval3 = scalarMap.to_rgba(abs(moy2[i]))
            plt.vlines(abs(moy[i]),0, 4*np.amax(post90), color=colorval2,linestyle='--',linewidth=2,zorder=1)
            plt.vlines(abs(moy2[i]),0, 4*np.amax(post90), color=colorval3,linestyle='--',linewidth=2,zorder=2)
            plt.fill_between(x90+0.5,4*post90[i,:], color=colorval,linewidth=.0,zorder=3)  #(0.658,0.682,0.690,1)
            plt.fill_between(x90+0.5,4*post902[i,:], color=colorval,linewidth=.0,zorder=4)
            plt.fill_between(x90+0.5,4*post902[i,:],4*post90[i,:], color=(0.901,0.917,0.921,1),linewidth=.5,zorder=5)
#            plt.plot(x90+0.5,4*post90[i,:], color=colorval2, lw=1.5,zorder=4)
#            plt.plot(x90+0.5,4*post902[i,:], color=colorval3, lw=1.5,zorder=5)
            plt.xlim(min(x90)-10,max(x90)+10)
            plt.ylim(0,0.13)
            plt.xticks([0,int(max(x90)/2),max(x90)])
            plt.yticks([]) 
        #text
        a=plt.subplot2grid((nd+2,ns+1), (0,1), colspan=1, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        a.text(0.5,0.2,'N',fontweight='bold',horizontalalignment='center',verticalalignment='center')
        a=plt.subplot2grid((nd+2,ns+1), (0,ns), colspan=1, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        a.text(0.5,0.2,'S',fontweight='bold',horizontalalignment='center',verticalalignment='center')
        a=plt.subplot2grid((nd+2,ns+1), (1,0), colspan=1, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        a.text(0.5,0.1,'0 km',fontweight='bold',horizontalalignment='center',verticalalignment='center')
        a=plt.subplot2grid((nd+2,ns+1), (nd,0), colspan=1, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        a.text(0.5,0.1,width,fontweight='bold',horizontalalignment='center',verticalalignment='center')
        a=plt.subplot2grid((nd+2,ns+1), (0,int((ns+1)/2)-1), colspan=5, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        a.text(0.5,0.1,'D I P - S L I P   P D F s'+' '*10+'--'+' '*10+'S T E P :   '+filename[0],fontweight='bold',horizontalalignment='center',verticalalignment='center')
        a=plt.subplot2grid((nd+2,ns+1), (0,int((ns+1)/2)+int((ns+1)/4)), colspan=3, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        a.text(0.5,0.1,'Sampled models: 420000',color='silver',style='italic',horizontalalignment='center',verticalalignment='center')
        #fig2.subplots_adjust(hspace=0.2) 
        #plt.tight_layout()
        plt.savefig(savedir+'pdfcomp_'+filename[0]+'_'+filename[1]+'_ds.png', format='png', dpi=450,pad_inches=0.0)
        plt.savefig(savedir+'pdfcomp_'+filename[0]+'_'+filename[1]+'_ds.pdf', format='pdf', pad_inches=0.0)
        plt.show()
        
        script1 = """
        cd {}
        convert {} -trim -bordercolor White -border 10x10 +repage {}
        """.format(savedir, 'pdfcomp_'+filename[0]+'_'+filename[1]+'_ds.png', 'pdfcomp_'+filename[0]+'_'+filename[1]+'_ds.png')
        subprocess.call(script1, shell=True)
    
    if fault_type == 'optimized':
        print("You need to write the code for an optimized fault!")
        
        return()

def plotSigma(filename, ns, nd, length, width, resdir, savedir,colorbar='sigma'):
    
    '''

    IN ARGUMENT:
    You need to specify at least 5 arguments:
     --> filename

     --> ns = number of patches in strike: total for classical type, 
                 number of large patches for optimized
          
     --> nd = number of patches in dip: total for classical type, 
                 number of large patches for optimized
                 
     --> Np = number of parameters
    
    --> resdir= directory of sampfile
     --> savedir= where to save PDF's figures
    '''

    # ---------------------------------------------------------------------------
    # Extract data from file   
    data = open(resdir+filename+'_log.txt','r').read()
    
    # Strings to find
    param = re.compile('_parameters: '+'\d+')
    for j in data.split("\n"):
        match = param.search(j)
        if match:
            Np = int(match.group().split(' ')[1])
    
    theta_mean = np.empty((int(step),Np))  
    theta_sd = np.empty((int(step),Np)) 
    it = np.empty(int(step))
    
    for i in range(int(step)):
        it[i]= False
        if i == 0:
            for l in data.split('\n'):
                if 'Theta_mean: ' in l:
                    theta_mean[i,:] = [float(s) for s in data.split('\n')[data.split('\n').index(l)+1].split(' ')[:-1]]
                if 'Theta_sd: ' in l:
                    theta_sd[i,:] = [float(s) for s in data.split('\n')[data.split('\n').index(l)+1].split(' ')[:-1]]

        else:
            for j in range(len(data.split('\n'))):
                bla = '--       iteration: '+str(i)+' '
                if bla in data.split('\n')[j]:
                    it[i] = True
                if 'Theta_mean: ' in data.split('\n')[j] and it[i] == True:
                    theta_mean[i,:] = [float(s) for s in data.split('\n')[j+1].split(' ')[:-1]]
                if 'Theta_sd: ' in data.split('\n')[j] and it[i] == True:
                    theta_sd[i,:] = [float(s) for s in data.split('\n')[j+1].split(' ')[:-1]]             
                    break
    
          
    plotSlip(step, ns, nd, length, width, theta_sd[-1][0:ns*nd], np.amax(theta_sd[-1]), savedir,legend='Sigma for strike-slip parameters (mm)',colorbar=colorbar,savename='sigma_strike')          
    plotSlip(step, ns, nd, length, width, theta_sd[-1][ns*nd:nd*ns*2], np.amax(theta_sd[-1]), savedir,legend='Sigma for dip-slip parameters (mm)',colorbar=colorbar,savename='sigma_dip')
    
    return()
        
def plotSlipTents(name,fault,slip=None, valmax=None, slipdir=None, sigma=False, savedir='./', legend='Coseismic slip (cm)',colorbar=colorsco,savename='slip',epicenter=None, index=False):
    '''
    '''
    if sigma.__class__ is list or sigma.__class__ is np.ndarray:
        sigma = np.array(sigma)
        if slip in ('strikeslip','ss','strike-slip'):
            slp = sigma[0:len(sigma)//2]
        elif slip in ('dipslip','ds','dip-slip'):
            slp = sigma[len(sigma)//2:len(sigma)]
        else:
            slp = np.sqrt(sigma[0:len(sigma)//2]**2 + sigma[len(sigma)//2:len(sigma)]**2)
    else:
        if slip in ('strikeslip','ss','strike-slip'):
            slp = np.abs(fault.slip[:,0].copy())
        elif slip in ('dipslip','ds','dip-slip'):
            slp = np.abs(fault.slip[:,1].copy())
        elif slip in ['tensile']:
            slp = fault.slip[:,2].copy()
        elif slip in ('total','tot'):
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
        else:
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
    
        
    if colorbar == 'sigma':
        cm = sns.cubehelix_palette(rot=-.32,light=0.99,dark=0.15,as_cmap=True)
    elif colorbar == 'slipres':
        cm = sns.cubehelix_palette(rot=-.4,light=0.98,dark=0.2,as_cmap=True)
    else:
        cmap.create_cmap(colorbar, 'cptslip')
        cm = plt.get_cmap('cptslip') 
    if valmax is not None:
        cNorm  = colors.Normalize(vmin=0, vmax=valmax)
    else:
        cNorm  = colors.Normalize(vmin=0, vmax=np.amax(slp))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    scalarMap.set_array([])

    fault.tent = np.array(fault.tent)
    trace_idx = np.argpartition(-fault.tent[:,2], 40)
    ind = np.argmin(fault.tent[trace_idx[:40],1], axis=None)
    p0 = np.array([fault.tent[ind,0], fault.tent[ind,1]])
    ind = np.argmax(fault.tent[trace_idx[:40],1], axis=None)
    p1 = np.array([fault.tent[ind,0], fault.tent[ind,1]])
    α = np.arccos( (p1[0]-p0[0])/np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2) )
    dis = []
    dep = []
    for i in range(np.shape(fault.tent)[0]):
        x = fault.tent[i,0]
        y = fault.tent[i,1]
        dy = np.cos(α) * (x-p0[0])
        dx = + np.cos(α)*(y-p0[1])
        dis.append(dx)
        dd = np.sqrt( fault.tent[i,2]**2 ) #+ (dy)**2)
        dep.append(-dd)
    
    # Make voronoi and bound the cells
    dis = np.vstack(dis)
    dep = np.vstack(dep)
    vertex = np.hstack((dis,dep))
#    vor = Voronoi(vertex)
#    regions, vertices = voronoi_finite_polygons_2d(vor)
    boundingbox = [np.amin(dis)-0.5,np.amax(dis)+0.7,np.amin(dep)-0.5,np.amax(dep)+0.5]# [x_min, x_max, y_min, y_max]
    vor = voronoi(vertex, boundingbox)
    regions, vertices = vor.filtered_regions, vor.vertices
    
    fig, ax = plt.subplots(1,figsize=(8,4))
    ax.set_xlabel('\n Distance along strike (km)')
    ax.set_ylabel('\n Distance along dip (km)')
    
    rects = []
    for i in range(len(regions)):
        region = regions[i]
        polygon = vertices[region]
        rect = patches.Polygon( polygon )
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
    
    # triangles
    vertices = fault.Vertices.tolist()
    faces = fault.Faces
    zt = []
    dist = []
#    for face in faces:
#        verts = [vertices[f] for f in face]
#        x = [v[0] for v in verts]
#        y = [v[1] for v in verts]
#        z = [-1.0*v[2] for v in verts]
#        d = np.sqrt((x-x0)**2 + (y-y0)**2)
#        zt.append(z); dist.append(d)
#        x = [v[0] for v in verts]
#        y = [v[1] for v in verts]
#        z = [v[2] for v in verts]
##        d = np.sqrt((y-y0)**2)
#        d=y
#        z2 = np.sqrt((x-x0)**2+np.array(z)**2)
#        zt.append(-z2)
#        dist.append(d)
            
    ## Plot triangles
    rects = []
#        import pdb
#        pdb.set_trace()
    for i in range(len(dist)):
        vertex = np.vstack((dist[i],np.array(zt[i])))
        rect = patches.Polygon( vertex.T )
        rects.append(rect)
    p = PatchCollection(rects, facecolors='None', edgecolor = 'black', lw=0.2)
    ax.add_collection(p)
    
    if index is True:
        centers= np.array(fault.getcenters())
        x = centers[:,0]
        y = centers[:,1]
        z = centers[:,2]
        for i in range(len(x)):
            d = y[i]
            dd = np.sqrt((x[i]-x0)**2 + z[i]**2)
            plt.text(d,-dd,str(int(fault.index_parameter[i,0])),color='k') 
              
    if slipdir is not None:
        rake = np.loadtxt(slipdir, comments='>')
        xc = rake[:,0]
        yc = rake[:,1]
        cent = []
        dep3=[]
        for i in range(len(xc)):
#            cent.append( np.sqrt((yc[i]-y0)**2) )
            cent.append(yc[i] )
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
        
    colo=plt.colorbar(scalarMap,orientation='vertical',fraction=0.0085,pad=0.02)
    if valmax is not None:
        colo.set_ticks([0,valmax/2,valmax])  
    else:
        colo.set_ticks([0,np.amax(slp)/2,np.amax(slp)])  
    ax=colo.ax
    ax.text(-0.9,0.5, legend,rotation='vertical',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    plt.xlim(np.amin(dis),np.amax(dis))
    plt.ylim(np.amin(dep),np.amax(dep))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.savefig(savedir+name+'_'+fault.name+'_'+savename+'.pdf', format='pdf',bbox_inches="tight")
    plt.show()              
     
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
    elif colorbar == 'diverging':
        cmap.create_cmap(diverging2, 'cptslip')
        cm = plt.get_cmap('cptslip')    
        if valmax is not None:
    #        cNorm  = colors.Normalize(vmin=0., vmax=valmax)
            cNorm = MidpointNormalize(vmin=0., vcenter=valmax, vmax=valmax+5.)
        else:
    #        cNorm  = colors.Normalize(vmin=np.amin(slp), vmax=np.amax(slp))
            cNorm = MidpointNormalize(vmin=np.amin(slp), vcenter=np.amax(slp), vmax=np.amax(slp)+5.)
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
#        d = np.sqrt((x-x0)**2 + (y-y0)**2)
        d = np.sqrt( (y-y0)**2)
        dis.append(d)
#        dep.append(-fault.patch[i,:,2] )
        dep.append(-np.sqrt(fault.patch[i,:,2]**2 + (x-x0)**2))
    
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

def plotSlip(step, nstrike, ndip, length, width, slip, vmax, savedir,legend='Coseismic slip (cm)',fault_type='classical',colorbar=colorsco,savename='slip'):
    
    
#    gmtcpt('/u/moana/user/ragon/fig/amatrice/gmt/slip.cpt', name='cptslip')    
#    cmap.create_cmap(colornames, 'cptslip')
#    cmape='cptslip'
#    cmape = sns.cubehelix_palette(rot=-.4,light=1, as_cmap=True)
#    Np = len(slip)
    
    if colorbar == 'sigma':
        cmape = sns.cubehelix_palette(rot=-.32,light=0.97,dark=0.35,as_cmap=True)
    elif colorbar == 'slipres':
        cmape = sns.cubehelix_palette(rot=-.4,light=0.98,dark=0.2,as_cmap=True)
    else:
        cmap.create_cmap(colorbar, 'cptslip')
        cmape = plt.get_cmap('cptslip') 
#        cmape= 'cptslip'
        cNorm  = colors.Normalize(-10,vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='cptslip')
        scalarMap.set_array([])
    
    if fault_type=='classical':
        slip = np.reshape(slip,(ndip,nstrike))
        
        fig=plt.figure(1, figsize=((nstrike+1)/2,(ndip)/2))
        fig.subplots_adjust(wspace=0.1,hspace=0.1)
        
        if slip[2,5]<0:
            sns.heatmap(-slip,vmin=0,vmax=vmax,cmap=cmape,xticklabels=False, yticklabels=False)
        else:
            sns.heatmap(slip,vmin=0,vmax=vmax,cmap=cmape,xticklabels=False, yticklabels=False)
        plt.savefig(savedir+'step_'+str(step)+'_'+savename+'.png', format='png', dpi=300,pad_inches=0.0)
        plt.savefig(savedir+'step_'+str(step)+'_'+savename+'.pdf', format='pdf')
        plt.show()
        plt.close()
#        c=1
#        l=0
#        for i in range(Np//2): 
#            if i % nstrike == 0:
#                #j=ns-1
#                l=l+1
#                c=1
#            a = plt.subplot2grid((ndip+2,nstrike+3), (l,c), colspan=1, rowspan=1)
#            #j=j-1
#            c=c+1
#            a.spines["top"].set_visible(False)   
#            a.spines["right"].set_visible(False)  
#            a.spines["left"].set_visible(False)
#            a.spines["bottom"].set_visible(False)
#            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
#                        labelbottom="off", left="off", right="off", labelleft="off") 
#            if slip[i]<=0:
#                sns.heatmap(-slip[i],vmin=0,vmax=150,cmap='cptslip',cbar=False,xticklabels=False, yticklabels=False)
#            else:
#                sns.heatmap(slip[i],vmin=0,vmax=150,cmap='cptslip',cbar=False,xticklabels=False, yticklabels=False)
#            plt.savefig(savedir+'step_'+str(step)+'_slip.png', format='png', dpi=300,pad_inches=0.0)
                    
    elif fault_type=='opti':
        pstk = []
        pdip=[]
        for i in range(nstrike*4+1):
            pstk.append( i*length/(nstrike*4) )
        for i in range(ndip*4+1):
            pdip.append( i*width/(ndip*4) )
        
        plt.figure(figsize=((length/3000, width/3000)))
        a = plt.subplot2grid((width/1000+2, length/1000+3), (2,3), colspan=length/1000, rowspan=width/1000)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                            labelbottom="off", left="off", right="off", labelleft="off") 
        l=1
        d=0
        p=0
        while l <= 2:
            s=0
            while s < len(pstk)-1:
                colorval = scalarMap.to_rgba(slip[p])
                a.add_patch( patches.Rectangle((pstk[s], pdip[d]),length/(nstrike*4),width/(ndip*4), edgecolor = colorval, facecolor=colorval) )
                if p < nstrike*4*2:           
                    p=p+1
                s=s+1
            d=d+1
            l=l+1
        while 3 <= l <= 5:
            s=0
            while s < len(pstk)-1:
                colorval = scalarMap.to_rgba(slip[p])
                a.add_patch( patches.Rectangle((pstk[s], pdip[d]),length/(nstrike*2),width/(ndip*2), edgecolor = colorval, facecolor=colorval) )
                if p < nstrike*3*2+nstrike*4*2:           
                    p=p+1
                s=s+2
            d=d+2
            l=l+1
        while 6 <= l <= ndip-2+5:
            s=0
            while s < len(pstk)-1:
                colorval = scalarMap.to_rgba(slip[p])
                a.add_patch( patches.Rectangle((pstk[s], pdip[d]),length/(nstrike),width/(ndip),edgecolor =colorval, facecolor=colorval) )
                if p <= Np:
                    p=p+1
                s=s+4
            d=d+4
            l=l+1
        plt.xlim([0, length])
        plt.ylim([width,0])
        
        a=plt.subplot2grid((width/1000+2, length/1000+3), (1,3), colspan=1, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        a.text(0.5,0.5,'N',fontweight='bold',horizontalalignment='center',verticalalignment='center')
        a=plt.subplot2grid((width/1000+2, length/1000+3), (1,length/1000+2), colspan=1, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        a.text(0.5,0.5,'S',fontweight='bold',horizontalalignment='center',verticalalignment='center')
        a=plt.subplot2grid((width/1000+2, length/1000+3), (2,2), colspan=1, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        a.text(-0.1,0.5,'0 km',fontweight='bold',horizontalalignment='center',verticalalignment='center')
        a=plt.subplot2grid((width/1000+2, length/1000+3), (width/1000+1,2), colspan=1, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        a.text(-0.1,0.5,str(width/1000)+' km',fontweight='bold',horizontalalignment='center',verticalalignment='center')
        a=plt.subplot2grid((width/1000+2, length/1000+3), (0,int(length/2000)), colspan=5, rowspan=1)
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        a.text(0.5,-0.1,'',fontweight='bold',horizontalalignment='center',verticalalignment='center')
        #colorbar
        a=plt.subplot2grid((width/1000+2, length/1000+3), (int(width/4000)+1,0), colspan=2, rowspan=int(width/2000))
        a.spines["top"].set_visible(False)   
        a.spines["right"].set_visible(False)  
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="off", left="off", right="off", labelleft="off") 
        colo=plt.colorbar(scalarMap,orientation='vertical',fraction=0.3)
        colo.set_ticks([np.amin(slip),np.amax(slip)])
        ax=colo.ax
        ax.set_yticklabels(colo.ax.get_yticklabels(), rotation='vertical')
        ax.text(-0.7,0.5,legend,horizontalalignment='center',verticalalignment='center',rotation = 'vertical', transform = ax.transAxes)
        plt.savefig(savedir+'step_'+str(step)+'_slip.png', format='png', dpi=300,pad_inches=0.0)
        plt.savefig(savedir+'step_'+str(step)+'_slip.pdf', format='pdf')
        
        script1 = """
        cd {}
        convert {} -trim -bordercolor White -border 10x10 -transparent white +repage {}
        """.format(savedir, 'step_'+str(step)+'_slip.png', 'step_'+str(step)+'_slip.png')
        subprocess.call(script1, shell=True)

    return

def ColValSup(slip,sigma,valmax,sigmamax):
    '''
    Here only for 15 categories
    
    slip and sigma the values
    slip max: max value (24 fm for Maule)
    sigma max: max value of uncertainty (12m for Maule)
    '''
    try:
        # Define the categories
        sigma1 = sigmamax/2.
        sigma2 = (sigmamax/4.)*3
        
#        import pdb
#        pdb.set_trace()
        colval = np.zeros(np.shape(slip))
        colval[(sigma >= sigmamax)] = 0
        colval[(slip >= valmax/2.) & (sigma >= sigma2) & (sigma < sigmamax)] = 2
        colval[(slip < valmax/2.) & (sigma >= sigma2) & (sigma < sigmamax) ] = 1
        
        cat = 2
        # 3rd row of bivariate
        for s in np.arange(0,valmax,valmax/4):
            cat = cat + 1 
            colval[(sigma < sigma2) & (sigma >= sigma1) & (slip >= s) & (slip < s+valmax/4)] = cat
        colval[(sigma < sigma2) & (sigma >= sigma1) & (slip >= valmax)] = 6
        
        # 4th row of bivariate (top)
        cat = 6
        for s in np.arange(0,valmax,valmax/8):
            cat = cat + 1 
            colval[(sigma < sigma1) & (slip >= s) & (slip < s+valmax/8)] = cat
        colval[(sigma < sigma1) & (slip >= valmax)] = cat
    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return colval
        
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

def plotSlipBivTents(name,fault,sigma,slip=None, valmax=None, sigmamax=None, slipdir=None, savedir='./',epicenter=None):
    '''
     
    '''
    try:
        # Bivariate colors defined from base to top and from left to right
        
        ## light gray - yellow - dark red
        bivcolors=[ (208, 208, 208),
                    (232, 232, 232),  (164, 128, 128),
                    (250, 250, 250), (237,204,187), (214, 137, 127), (149, 75, 75),
                    (254, 248, 241), (254, 227, 190), (253, 173, 119), (248, 130, 84), (233, 87, 61), (210, 41, 27), (173, 0, 0), (127, 0, 0)]
        bivcolors_rgba= [(x[0]/255,x[1]/255,x[2]/255) for x in bivcolors]
        
        ## light blue - red
#        bivcolors_hex = ['#d0d0d0',
#                         '#e8e8e8','#c3a8a7',
#                         '#dfe8e6','#bacad6','#ada7bd','#b37872',
#                         '#eaf3ec', '#c0dbde', '#9bc2d4', '#7ba8ce', '#8886b2', '#906587', '#904356', '#861c20']
#        bivcolors = [ImageColor.getcolor(i, "RGB") for i in bivcolors_hex]
#        bivcolors_rgba= [(x[0]/255,x[1]/255,x[2]/255) for x in bivcolors]
        
#        bivcolors=[ (208, 208, 208),
#                    (221, 221, 221),  (199, 173, 173),
#                    (232, 232, 232),  (228, 178, 152), (214, 137, 127), (174, 111, 111),
#                    (250, 250, 250), (253, 214, 163), (253, 173, 119), (248, 130, 84), (233, 87, 61), (210, 41, 27), (173, 0, 0), (127, 0, 0)]
                    
#        bivcolors=[ (208, 208, 208),
#                    (221, 221, 221),  (164, 128, 128),                    
#                    (232, 232, 232), (228, 203, 176), (214,137,127), (149, 75, 75),
#                    (250, 250, 250), (255, 243, 226), (253, 214, 163), (253, 173, 119), (250, 137, 87), (232,87,61), (194,20,13), (127, 0, 0)]
        
        cmap.create_cmap(bivcolors, 'biv')
#        cmape = plt.cm.get_cmap('biv',15)
        cNorm  = colors.Normalize(0,14)
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
#        import pdb
#        pdb.set_trace()
        colval = ColValSup(np.array(slp),np.array(sgm),slipmax,uncmax)
        
        fault.tent = np.array(fault.tent)
        x0 = np.amin(fault.tent[:,0])
        ind = np.argmin(fault.tent[:,0], axis=None)
        y0 = fault.tent[ind,1]
        dis = []
        dep = []
        for i in range(np.shape(fault.tent)[0]):
            x = fault.tent[i,0]
            y = fault.tent[i,1]
            d = np.sqrt((x-x0)**2 + (y-y0)**2)
            dis.append(d)
            dep.append(-fault.tent[i,2] )
        
        # Make voronoi and bound the cells
        dis = np.vstack(dis)
        dep = np.vstack(dep)
        vertex = np.hstack((dis,dep))
        vor = Voronoi(vertex)
        regions, vertices = voronoi_finite_polygons_2d(vor)
    
        fig, ax = plt.subplots(1,figsize=((np.amax(np.abs(dis))/np.amax(np.abs(dep)))*3,3))
        ax.set_xlabel('\n Distance along strike (km)')
        ax.set_ylabel('\n Depth (km)')
        
        rects = []
        for i in range(len(regions)):
            region = regions[i]
            polygon = vertices[region]
            rect = patches.Polygon( polygon )
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
                      2.5*slp[:], 2.5*slp[:],
                      units = 'width',
                      angles = [-rake[:,7]],
                      width = 0.002,
    #                  scale = None, 
    #                  scale_units='inches',
                      scale = 10**1.5, 
                      scale_units = 'x', 
                      color='dimgrey')
        
        if epicenter is not None:
            xe, ye = fault.ll2xy(epicenter[0], epicenter[1])
            de = np.sqrt((xe-x0)**2 + (ye-y0)**2)
            plt.scatter(de, epicenter[2], s=100, c='white', edgecolors='dimgrey', marker=(5, 1))
            
        # plot colorscale 
        center = [(5.5/6)*np.amax(dis), -3.5*np.amax(np.abs(dep))/4]
        L = np.amax(np.abs(dep))/2
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
        labels = [str(i) for i in np.arange(0,valmax,valmax/4)]+[str(valmax)]+[str(uncmax//2)]+[str(3*uncmax//4)]+[str(int(uncmax))]     
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
        
        plt.xlim(np.amin(dis),np.amax(dis))
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

def plotSlipBivTentsCont(name,fault,sigma,slip=None, valmax=None, sigmamax=None, slipdir=None, savedir='./',epicenter=None, xystrides=[100, 100], index=False):
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
        
        ## colorbars
        cptslip = colors.LinearSegmentedColormap.from_list('cptslip',colorsco_above_rgba, N=256)
        if valmax is not None:
    #        cNorm  = colors.Normalize(vmin=0., vmax=valmax)
            cNorm = MidpointNormalize(vmin=0., vcenter=valmax, vmax=valmax+5.)
        else:
    #        cNorm  = colors.Normalize(vmin=np.amin(slp), vmax=np.amax(slp))
            cNorm = MidpointNormalize(vmin=0, vcenter=np.amax(slp), vmax=np.amax(slp)+5.)
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

        fault.tent = np.array(fault.tent)
        x0 = np.amin(fault.tent[:,0])
        ind = np.argmin(fault.tent[:,0], axis=None)
        y0 = fault.tent[ind,1]
        dis = []
        dep = []
        for i in range(np.shape(fault.tent)[0]):
            x = fault.tent[i,0]
            y = fault.tent[i,1]
            d = y
            dis.append(d)
            dep.append(- np.sqrt(fault.tent[i,2]**2 + (x-x0)**2) )
        dis = np.vstack(dis)
        dep = np.vstack(dep)
        
        # triangles
        vertices = fault.Vertices.tolist()
        faces = fault.Faces
        zt = []
        dist = []
        for face in faces:
            verts = [vertices[f] for f in face]
            x = [v[0] for v in verts]
            y = [v[1] for v in verts]
            z = [v[2] for v in verts]
            d = y
            z2 = -np.sqrt(np.array(z)**2 + (x-x0)**2)
            zt.append(z2); dist.append(d)
            
                
        fig, ax = plt.subplots(1,figsize=((np.amax(np.abs(dis))/np.amax(np.abs(dep)))*7,3))
        ax.set_xlabel('\n Distance along strike (km)')
        ax.set_ylabel('\n Distance along dip (km)')
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        
        ## Plot slip
        if not hasattr(fault, 'plotSources'):
            print('--------------------------------------')
            print('Please precompute sources for plotting')
        Ids = fault.plotSources[0]
        X = fault.plotSources[1]
        Y = fault.plotSources[2]
        Z = fault.plotSources[3]

        Slip = fault._getSlipOnSubSources(Ids, X, Y, Z, slp)
        Sigma = fault._getSlipOnSubSources(Ids, X, Y, Z, sgm)
        D = Y
        diss2 = np.linspace(np.nanmin(D), np.nanmax(D), 300)
        Z2 = np.sqrt((X-x0)**2 + (Z)**2)
        z2 = -np.linspace(np.nanmin(Z2)-5., np.nanmax(Z2)+5.,300)      
        diss2, z2 = np.meshgrid(diss2,z2)
        slip22 = sciint.griddata((D,-Z2),Slip,(diss2,z2),method='linear')
        sigma22 = sciint.griddata((D,-Z2),Sigma,(diss2,z2),method='linear')
        import scipy.ndimage as ndimage
        slip2 = ndimage.median_filter(slip22,size=(1,1))
        sigma2 = ndimage.median_filter(sigma22,size=(1,1))
           
        cslip = colors.LinearSegmentedColormap.from_list('cslip',scalarMap.to_rgba(range(0,int(valmax)+5)), 256)
        ax.pcolor(diss2, z2, slip2, cmap=cslip, vmin=0, vmax=slipmax,rasterized = True)
        
        ## Plot triangles
        rects = []
        for i in range(len(dist)):
            vertex = np.vstack((dist[i],np.array(zt[i])))
            rect = patches.Polygon( vertex.T )
            rects.append(rect)
        p = PatchCollection(rects, facecolors='None', edgecolor = 'white', lw=0.2)
        ax.add_collection(p)
        
        ax.pcolor(diss2, z2, sigma2, cmap=cptsig2, vmin=0, vmax=sigmamax, edgecolors=None,rasterized = True)

        if index is True:
            centers= np.array(fault.getcenters())
            x = centers[:,0]
            y = centers[:,1]
            z = centers[:,2]
            for i in range(len(x)):
                d = y[i]
                dd = np.sqrt((x[i]-x0)**2 + z[i]**2)
                plt.text(d,-dd,str(int(fault.index_parameter[i,0])),color='k') 
                
        if slipdir is not None:
            rake = np.loadtxt(slipdir, comments='>')
            xc = rake[:,0]
            yc = rake[:,1]
            cent = []
            dep3=[]
            for i in range(len(xc)):
                cent.append( yc[i] )
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
                      scale = 2.5**1.5, 
                      scale_units = 'x', 
                      color='dimgrey')
        
        if epicenter is not None:
            xe, ye = fault.ll2xy(epicenter[0], epicenter[1])
            de = np.sqrt((xe-x0)**2 + (ye-y0)**2)
            plt.scatter(de, epicenter[2], s=100, c='white', edgecolors='dimgrey', marker=(5, 1))
        
        plt.xlim(np.amin(dis),np.amax(dis))
#        plt.ylim(-np.amin(-dep)+2,-np.amax(-dep))
        
        # Plot depth axis
#        import pdb
#        pdb.set_trace()
        xold = np.linspace(np.amin(np.abs(dep)),np.amax(np.abs(dep)),10)
        xnew = np.linspace(np.amin(np.abs(fault.tent[:,2])),np.amax(np.abs(fault.tent[:,2])),10)
        def forward(x):
            return -np.interp(np.abs(x), xold, xnew)
        def inverse(x):
            return -np.interp(np.abs(x), xnew, xold)
        ax2 = ax.secondary_yaxis(-0.1, functions=(forward, inverse))
        ax2.set_ylabel('Depth (km)')
        ax2.set_yticks([0,-10,-20,-40,-60])
        
        # plot colorscale
#        center = [(5.5/6)*np.amax(dis), -3.5*np.amax(np.abs(dep))/4]
#        L = np.amax(np.abs(dep))/2
#        wdgs = []
#        #1
#        wdgs.append(patches.Wedge(center, L/4, -30+90, 30+90, width=None,ec='white',fc=bivcolors_rgba[0],lw=0.1))
#        #2
#        wdgs.append(patches.Wedge(center, 2*L/4, -30+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[2],lw=0.1))
#        wdgs.append(patches.Wedge(center, 2*L/4, 0+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[1],lw=0.1))
#        #3
#        wdgs.append(patches.Wedge(center, 3*L/4, -30+90, -15+90, width=L/4,ec='white',fc=bivcolors_rgba[6],lw=0.1))
#        wdgs.append(patches.Wedge(center, 3*L/4, -15+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[5],lw=0.1))
#        wdgs.append(patches.Wedge(center, 3*L/4, 0+90, 15+90, width=L/4,ec='white',fc=bivcolors_rgba[4],lw=0.1))
#        wdgs.append(patches.Wedge(center, 3*L/4, 15+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[3],lw=0.1))
#        #4
#        wdgs.append(patches.Wedge(center, 4*L/4, -30+90, -22.5+90, width=L/4,ec='white',fc=bivcolors_rgba[14],lw=0.1))
#        wdgs.append(patches.Wedge(center, 4*L/4, -22.5+90, -15+90, width=L/4,ec='white',fc=bivcolors_rgba[13],lw=0.1))
#        wdgs.append(patches.Wedge(center, 4*L/4, -15+90, -7.5+90, width=L/4,ec='white',fc=bivcolors_rgba[12],lw=0.1))
#        wdgs.append(patches.Wedge(center, 4*L/4, -7.5+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[11],lw=0.1))
#        wdgs.append(patches.Wedge(center, 4*L/4, 0+90, 7.5+90, width=L/4,ec='white',fc=bivcolors_rgba[10],lw=0.1))
#        wdgs.append(patches.Wedge(center, 4*L/4, 7.5+90, 15+90, width=L/4,ec='white',fc=bivcolors_rgba[9],lw=0.1))
#        wdgs.append(patches.Wedge(center, 4*L/4, 15+90, 22.5+90, width=L/4,ec='white',fc=bivcolors_rgba[8],lw=0.1))
#        wdgs.append(patches.Wedge(center, 4*L/4, 22.5+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[7],lw=0.1))
#        w = PatchCollection(wdgs, match_original=True)
#        ax.add_collection(w)
#        
#        #legend
#        coords = []
#        for i in [14,12,10,8]:
#            coords.append(wdgs[i].get_path().vertices[6])
#        for i in [7,3,1,0]:
#            coords.append(wdgs[i].get_path().vertices[0])
#        labels = [str(i) for i in np.arange(0,valmax,valmax/4)]+[str(valmax)]+[str(uncmax//2)]+[str(3*uncmax//4)]+[str(int(uncmax))]     
#        rot = [30,15,0,-15,-30,60,60,60]
#        vas = ['center']*5+['top']*3
#        offset = [[0,0.05*L]]*5+[[0.05*L,0]]*3
#        for i in range(len(labels)):
#            ax.text(coords[i][0]+offset[i][0],coords[i][1]+offset[i][1],labels[i],rotation=rot[i],rotation_mode='anchor',ha='center',va=vas[i])
#        
#        # legend titles
#        wdg = patches.Wedge(center, L, -30+90, 30+90, width=None)
#        x = wdg.get_path().vertices[:,0]
#        y = wdg.get_path().vertices[:,1]
#        text = CurvedText(
#            x = x[::-1][3:-1],
#            y = y[::-1][3:-1]+0.13*L,
#            text='Slip amplitude (m)',
#            va = 'bottom',
#            fontweight='regular',
#            axes = ax)
#        text2 = CurvedText(
#            x = x[::-1][0:2]+0.13*L,
#            y = y[::-1][0:2],
#            text='Slip uncertainty (m)',
#            va = 'top',
#            fontweight='regular',
#            axes = ax)

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
        
        plt.savefig(savedir+name+'_'+fault.name+'_slipbivcont.pdf', format='pdf',bbox_inches="tight",dpi=300)
        plt.savefig(savedir+name+'_'+fault.name+'_slipbivcont.png',bbox_inches="tight",dpi=300)
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    
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
        
def famslip(step, name, samp, nbrfam, ns, nd, length, width, savedir, valmin,valmax, slip='total', fault_type='classical', distinct_fam = False,color='co'):
    '''
    slip: 'ds', 'ss' or 'total'
    '''
    try:
        from collections import defaultdict
        if color=='po':
            color=colorspo
        elif color=='co':
            color=colorsco
        elif color=='colornames':
            color=colornames
        Np = ns*nd*2
        cmap.create_cmap(color, 'cptslip')
        jet = cm = plt.get_cmap('cptslip')
        cNorm  = colors.Normalize(valmin,valmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='cptslip')
        scalarMap.set_array([])
        
        if distinct_fam == False:
            subf=defaultdict(list)
            for j in range(Np//2):
                for i in range(1,nbrfam+1):
                    if slip == 'total':
                        subf[j].append(np.sqrt(samp[i][0][j]**2+samp[i][0][j+Np//2]**2))   
                    elif slip == 'ds' or slip == 'dip' or slip == 'dip-slip':
                        subf[j].append(np.sqrt(samp[i][0][j+Np//2]**2))
                    elif slip == 'ss' or slip == 'stk' or slip == 'strike-slip' or slip == 'strike':
                        subf[j].append(np.sqrt(samp[i][0][j]**2))
                    else:
                        subf[j].append(np.sqrt(samp[i][0][j]**2+samp[i][0][j+Np//2]**2))
                        
            for j in range(Np//2):
                subf[j]=np.array(subf[j])
                subf[j]=np.reshape(subf[j],(int(np.sqrt(nbrfam)),int(np.sqrt(nbrfam))))
                
            if fault_type == 'classical':
                fig=plt.figure(1, figsize=((ns+1),(nd+1)))
                fig.subplots_adjust(wspace=0.1,hspace=0.1)
                
                c=1
                l=0
                for i in range(Np//2): 
                    if i % ns == 0:
                        #j=ns-1
                        l=l+1
                        c=1
                    a = plt.subplot2grid((nd+2,ns+3), (l,c), colspan=1, rowspan=1)
                    #j=j-1
                    c=c+1
                    a.spines["top"].set_visible(False)   
                    a.spines["right"].set_visible(False)  
                    a.spines["left"].set_visible(False)
                    a.spines["bottom"].set_visible(False)
                    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                                labelbottom="off", left="off", right="off", labelleft="off") 
                    if subf[i][0,0]<=0:
                        sns.heatmap(-subf[i],vmin=valmin,vmax=valmax,cmap='cptslip',cbar=False,xticklabels=False, yticklabels=False)
                    else:
                        sns.heatmap(subf[i],vmin=valmin,vmax=valmax,cmap='cptslip',cbar=False,xticklabels=False, yticklabels=False)
                
                matplotlib.rcParams.update({'font.size': 15})
                a=plt.subplot2grid((nd+2,ns+3), (nd+1,1), colspan=5, rowspan=1)
                a.spines["top"].set_visible(False)   
                a.spines["right"].set_visible(False)  
                a.spines["left"].set_visible(False)
                a.spines["bottom"].set_visible(False)
                plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                            labelbottom="off", left="off", right="off", labelleft="off") 
                colo=plt.colorbar(scalarMap,orientation='horizontal',fraction=0.5)
                colo.set_ticks([0,valmax/2,valmax])
#                if color=='po':
#                    tick_locs   = [10,20,30,40]
#                    tick_labels = ['10','20','30','40 cm']
#                else:
#                    tick_locs   = [10,50,100,150]
#                    tick_labels = ['10','50','100','150 cm']
#                colo.locator     = matplotlib.ticker.FixedLocator(tick_locs)
#                colo.formatter   = matplotlib.ticker.FixedFormatter(tick_labels)
#                colo.update_ticks()
                
#                a.text(0.05,0.8,'0',horizontalalignment='center',verticalalignment='center')
#                a=plt.subplot2grid((nd+2,ns+3), (0,ns+1), colspan=1, rowspan=1)
#                a.spines["top"].set_visible(False)   
#                a.spines["right"].set_visible(False)  
#                a.spines["left"].set_visible(False)
#                a.spines["bottom"].set_visible(False)
#                plt.tick_params(axis="both", which="both", bottom="off", top="off",  
#                            labelbottom="off", left="off", right="off", labelleft="off") 
#                a.text(0.1,-0.1,'- 0',horizontalalignment='center',verticalalignment='center')
#                
#                a=plt.subplot2grid((nd+2,ns+3), (3,ns+1), colspan=1, rowspan=1)
#                a.spines["top"].set_visible(False)   
#                a.spines["right"].set_visible(False)  
#                a.spines["left"].set_visible(False)
#                a.spines["bottom"].set_visible(False)
#                plt.tick_params(axis="both", which="both", bottom="off", top="off",  
#                            labelbottom="off", left="off", right="off", labelleft="off") 
#                a.text(0.1,0.,'- 5',horizontalalignment='center',verticalalignment='center')
#                              
#                a=plt.subplot2grid((nd+2,ns+3), (nd-5,ns+1), colspan=1, rowspan=4)
#                a.spines["top"].set_visible(False)   
#                a.spines["right"].set_visible(False)  
#                a.spines["left"].set_visible(False)
#                a.spines["bottom"].set_visible(False)
#                plt.tick_params(axis="both", which="both", bottom="off", top="off",  
#                            labelbottom="off", left="off", right="off", labelleft="off") 
#                a.text(0.25,0,'Down dip distance (km)',rotation=90,horizontalalignment='center',verticalalignment='center')
#                               
#                a=plt.subplot2grid((nd+2,ns+3), (nd+1,7), colspan=1, rowspan=1)
#                a.spines["top"].set_visible(False)   
#                a.spines["right"].set_visible(False)  
#                a.spines["left"].set_visible(False)
#                a.spines["bottom"].set_visible(False)
#                plt.tick_params(axis="both", which="both", bottom="off", top="off",  
#                            labelbottom="off", left="off", right="off", labelleft="off") 
#                a.text(0.01,0.8,'10',horizontalalignment='center',verticalalignment='center')
#                
#                a=plt.subplot2grid((nd+2,ns+3), (nd+1,ns-2), colspan=4, rowspan=1)
#                a.spines["top"].set_visible(False)   
#                a.spines["right"].set_visible(False)  
#                a.spines["left"].set_visible(False)
#                a.spines["bottom"].set_visible(False)
#                plt.tick_params(axis="both", which="both", bottom="off", top="off",  
#                            labelbottom="off", left="off", right="off", labelleft="off") 
#                a.text(0.,0.75,'Along strike distance (km)',horizontalalignment='center',verticalalignment='center')
                
#                fig = plt.gcf()  
                if slip == 'total':
                    fig.savefig(savedir+'step_'+str(step)+'_slipfam_'+name+'.png', format='png', dpi=300,pad_inches=0.0)
                    fig.savefig(savedir+'step_'+str(step)+'_slipfam_'+name+'.pdf', format='pdf')
                elif slip == 'ds' or slip == 'dip' or slip == 'dip-slip':
                    fig.savefig(savedir+'step_'+str(step)+'_slipfam_'+name+'dip-slip.png', format='png', dpi=300,pad_inches=0.0)
                    fig.savefig(savedir+'step_'+str(step)+'_slipfam_'+name+'dip-slip.pdf', format='pdf')
                elif slip == 'ss' or slip == 'stk' or slip == 'strike-slip' or slip == 'strike':
                    fig.savefig(savedir+'step_'+str(step)+'_slipfam_'+name+'stk-slip.png', format='png', dpi=300,pad_inches=0.0)
                    fig.savefig(savedir+'step_'+str(step)+'_slipfam_'+name+'stk-slip.pdf', format='pdf')
                else:
                    fig.savefig(savedir+'step_'+str(step)+'_slipfam_'+name+'.png', format='png', dpi=300,pad_inches=0.0)
                    fig.savefig(savedir+'step_'+str(step)+'_slipfam_'+name+'.pdf', format='pdf')
                plt.show()
                plt.close()
                print('ik')
                script1 = """
                cd {}
                convert {} -trim -bordercolor White -border 10x10 -transparent white +repage {}
                """.format(savedir, 'step_'+str(step)+'_slipfam_'+name+'.png', 'step_'+str(step)+'_slipfam_'+name+'.png')
                subprocess.call(script1, shell=True)
                
            if fault_type == 'optimized':
                fig=plt.figure(1, figsize=(ns*4,nd-2+5))
                fig.subplots_adjust(wspace=0.1,hspace=0.1)
                j=2
                k=5
                c=1
                c2=1
                for i in range(Np//2): 
                    if i < ns*4: # first line
                        a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (1,i+1), colspan=1, rowspan=1)
                    elif ns*4 <= i <= ns*4*2-1:
                        a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (2,i-ns*4+1), colspan=1, rowspan=1)
                    elif ns*4*2 <= i <= 14*ns-1: # third line
                        if i % (2*ns) == 0:
                            j=j+1
                            c=1
                        a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (j,c), colspan=1, rowspan=1)
                        c=c+2
                    else: # sixth line
                        if i % ns == 0:
                            k=k+1
                            c2=1
                        a = plt.subplot2grid((nd-2+3+2+1,ns*4+1), (k,c2), colspan=1, rowspan=1)
                        c2=c2+4
                    a.spines["top"].set_visible(False)   
                    a.spines["right"].set_visible(False)  
                    a.spines["left"].set_visible(False)
                    a.spines["bottom"].set_visible(False)
                    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                                labelbottom="off", left="off", right="off", labelleft="off") 
                    sns.heatmap(samp[i],cmap='cptslip',cbar=False,xticklabels=False, yticklabels=False)
                fig = plt.gcf()  
                fig.savefig(savedir+'step_'+str(step)+'_slipfam_'+name+'.png', format='png', dpi=300,pad_inches=0.0)
                fig.savefig(savedir+'step_'+str(step)+'_slipfam_'+name+'.pdf', format='pdf')
                plt.show()
                script1 = """
                cd {}
                convert {} -trim -bordercolor White -border 10x10 -transparent white +repage {}
                """.format(savedir, 'step_'+str(step)+'_slipfam_'+name+'.png', 'step_'+str(step)+'_slipfam_'+name+'.png')
                subprocess.call(script1, shell=True)
    #    
        elif distinct_fam == True:
            cNorm  = colors.Normalize(valmin,valmax)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='cptslip')
            scalarMap.set_array([])
            
            fig=plt.figure(1, figsize=(ns*int(np.sqrt(nbrfam))/5,nd*int(np.sqrt(nbrfam))/5))
            fig.subplots_adjust(wspace=0.2,hspace=0.2)
            
            l1=0
            c1=0
            for i in range(1,nbrfam+1):
                if i==6:
                    l1=1
                    c1=0
                elif i==11:
                    l1=2
                    c1=0
                elif i==16:
                    l1=3
                    c1=0
                elif i==21:
                    l1=4
                    c1=0
                a = plt.subplot2grid((int(np.sqrt(nbrfam)),int(np.sqrt(nbrfam))), (l1,c1), colspan=1, rowspan=1)
                a.spines["top"].set_visible(False)   
                a.spines["right"].set_visible(False)  
                a.spines["left"].set_visible(False)
                a.spines["bottom"].set_visible(False)
                plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                                        labelbottom="off", left="off", right="off", labelleft="off") 
                
                if fault_type == 'optimized':
                    slip=np.sqrt(np.square(samp[i][0][0:Np//2])+np.square(samp[i][0][Np//2:Np]))
                    pstk = []
                    pdip=[]
                    for k in range(ns*4+1):
                        pstk.append( k*length/(ns*4) )
                    for k in range(nd*4+1):
                        pdip.append( k*width/(nd*4) )
                    
                    l=1
                    d=0
                    p=0
                    while l <= 2:
                        s=0
                        while s < len(pstk)-1:
                            print(i)
                            print(p)
                            colorval = scalarMap.to_rgba(slip[p])
                            a.add_patch( patches.Rectangle((pstk[s], pdip[d]),length/(ns*4),width/(nd*4), edgecolor = colorval, facecolor=colorval) )
                            if p < ns*4*2:           
                                p=p+1
                            s=s+1
                        d=d+1
                        l=l+1
                    while 3 <= l <= 5:
                        s=0
                        while s < len(pstk)-1:
                            colorval = scalarMap.to_rgba(slip[p])
                            a.add_patch( patches.Rectangle((pstk[s], pdip[d]),length/(ns*2),width/(nd*2), edgecolor = colorval, facecolor=colorval) )
                            if p < ns*3*2+ns*4*2:           
                                p=p+1
                            s=s+2
                        d=d+2
                        l=l+1
                    while 6 <= l <= nd-2+5:
                        s=0
                        while s < len(pstk)-1:
                            colorval = scalarMap.to_rgba(slip[p])
                            a.add_patch( patches.Rectangle((pstk[s], pdip[d]),length/(ns),width/(nd),edgecolor =colorval, facecolor=colorval) )
                            if p <= Np:
                                p=p+1
                            s=s+4
                        d=d+4
                        l=l+1
                    plt.xlim([0, length])
                    plt.ylim([width,0])
                
                if fault_type == 'classical':
                    slip=np.sqrt(np.square(samp[i][0][0:Np//2])+np.square(samp[i][0][Np//2:Np]))
                    slp=np.reshape(slip,(nd,ns))
                    if slp[0,0]<=0:
                        sns.heatmap(-slp,vmin=valmin,vmax=valmax,cmap='cptslip',cbar=False,xticklabels=False, yticklabels=False)
                    else:
                        sns.heatmap(slp,vmin=valmin,vmax=valmax,cmap='cptslip',cbar=False,xticklabels=False, yticklabels=False)
    
                
                c1=c1+1   
            fig.savefig(savedir+'step_'+str(step)+'_slipfamdist_'+name+'.png', format='png', dpi=300,pad_inches=0.0)
            plt.show() 
            script1 = """
            cd {}
            convert {} -trim -bordercolor White -border 10x10 -transparent white +repage {}
            """.format(savedir, 'step_'+str(step)+'_slipfamdist_'+name+'.png', 'step_'+str(step)+'_slipfamdist_'+name+'.png')
            subprocess.call(script1, shell=True)

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return        


def plotCp(confdir,savedir):
    cmap = sns.cubehelix_palette(rot=-.4,light=0.95, as_cmap=True)
#    cmap = sns.cubehelix_palette(50, rot=-.32,light=0.9,dark=0.3)
    
    Cpdip = np.loadtxt(confdir+'amat.cpdip.txt')
    Cppos = np.loadtxt(confdir+'amat.cppos.txt')
    Cpmu = np.loadtxt(confdir+'amat.cpmu.txt')
    
#    fig=plt.figure(figsize = (7,9))
#    sns.heatmap(Cpdip,cmap=cmap,cbar_kws={"orientation": "horizontal"})
#    
#    fig=plt.figure(figsize = (7,9))
#    sns.heatmap(Cppos,cmap=cmap,cbar_kws={"orientation": "horizontal"})
    
    fig=plt.figure(figsize = (21,9))
    ax = fig.add_subplot(131)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                   labelbottom="off", left="off", right="off", labelleft="on") 
    #cax = ax.matshow(Cp[100:200,100:200], aspect='auto', cmap=cmap)#,vmin=-0.01, vmax=0.025)
    ax = sns.heatmap(Cpmu[0:1036,0:1036],ax=ax,cmap=cmap,cbar_kws={"orientation": "horizontal"},linewidths=.0, rasterized=True)
    ax.xaxis.set_label_position("top")
#    ax.xaxis.tick_top()
#    plt.xlabel('Distance from fault (km)')
#    plt.ylabel('Distance from fault (km)')
    #cbar=fig.colorbar(cax,orientation="horizontal")
    #cbar.ax.set_aspect(0.03)
    #cbar.ax.text(0.5,1.8,r'$m^2$',horizontalalignment='center',verticalalignment='center',transform = cbar.ax.transAxes)
    
    
    ax = fig.add_subplot(132)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                   labelbottom="off", left="off", right="off", labelleft="on") 
    #cax = ax.matshow(Cpd[200:300,200:300], aspect='auto', cmap='YlGnBu')#,vmin=-0.01, vmax=0.025)
    ax = sns.heatmap(Cpmu[1036:2072,1036:2072],ax=ax,cmap=cmap,cbar_kws={"orientation": "horizontal"},linewidths=.0, rasterized=True)
    ax.xaxis.set_label_position("top")
#    ax.xaxis.tick_top()
#    plt.xlabel('Distance from fault (km)')
    #cbar=fig.colorbar(cax,orientation="horizontal")
    #cbar.ax.set_aspect(0.03)
    #cbar.ax.text(0.5,1.8,r'$m^2$',horizontalalignment='center',verticalalignment='center',transform = cbar.ax.transAxes)
    
    ax = fig.add_subplot(133)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                   labelbottom="off", left="off", right="off", labelleft="on") 
    #cax = ax.matshow(Cpd[0:100,0:100], aspect='auto', cmap='YlGnBu')#,vmin=-0.01, vmax=0.025)
    ax = sns.heatmap(Cpmu[2072:3108,2072:3108],ax=ax,cmap=cmap,cbar_kws={"orientation": "horizontal"},linewidths=.0, rasterized=True)
#    ax.xaxis.set_label_position("top")
#    ax.xaxis.tick_top()
#    plt.xlabel('Distance from fault (km)')
    #cbar=fig.colorbar(cax,orientation="horizontal")
    #cbar.ax.set_aspect(0.03)
    #cbar.ax.text(0.5,1.8,r'$m^2$',horizontalalignment='center',verticalalignment='center',transform = cbar.ax.transAxes)

#    fig.savefig('/u/moana/user/ragon/code/altar/2d/results/fig/cp.pdf',format='pdf', dpi=300)

    return
    
def plotSlip3D(step,resdir,faultparams,coord,azimuth,valmax,slip='total',savedir='./'):
    '''
    step
    faultparams
    coord: tab with lon, lat, z of all the faults
    azimuth
    valmax: slip max value for colorscale
    slip: 'total' 'dip' or 'strike'
    '''
    from mpl_toolkits.basemap import Basemap

    plt.rcParams['grid.color'] = 'ghostwhite'
    length = faultparams[0]
    width = faultparams[1]
    nstrike = faultparams[2]
    ndip = faultparams[3]
    strike = faultparams[4]
    lon = faultparams[5]
    lat =  faultparams[6]
    dip = faultparams[7]
    z = faultparams[8]
    
    cmap.create_cmap(colorsco, 'cptslip')
    cm = plt.get_cmap('cptslip') 
    cNorm  = colors.Normalize(vmin=0, vmax=valmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    scalarMap.set_array([])
    
    h5file =  h5py.File(resdir+filename+'.h5','r')
    samp = np.array(list(h5file[u'Sample Set']))
#    set_trace()
    if samp.shape[1] > nstrike*ndip*2:
        samp = samp[:,0:np.int(nstrike*ndip*2)]
    samp = np.transpose(samp)
    moy = np.mean( samp, axis=1 ) 
    Ns = float(samp.shape[0])
    Np = 2*nstrike*ndip
    
    if slip in ['total','tot','all']:
        slp = np.abs(moy[0:Np//2]) + np.abs(moy[Np//2:Np])
    if slip in ['dip','dip-slip','dip slip','ds']:
        slp = np.abs(moy[Np//2:Np])
    if slip in ['strike','strike-slip','strike slip','ss','stk']:
        slp = moy[0:Np//2]
        
    fig = plt.figure(figsize=(13,5))
    ax = a3.Axes3D(fig)
       
    ax.azim = azimuth
    ax.elev=20
    #    ax.grid(False)
    ##    ax.xaxis.pane.set_edgecolor('black')
    ##    ax.yaxis.pane.set_edgecolor('black')
    #    ax.xaxis.pane.fill = False
    #    ax.yaxis.pane.fill = False
    #    ax.zaxis.pane.fill = False
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.1
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.1
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.1
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel('\n Longitude')
    ax.set_ylabel('\n Latitude')
    ax.set_zlabel('\n Depth (km)')
    #    ax.set_xticks([])
    #    ax.set_yticks([])
    #    ax.set_zticks([])
    coord[:,:,2] = -coord[:,:,2]
    plt.xlim(np.amin(coord[:,:,0]),np.amax(coord[:,:,0]))
    plt.ylim(np.amin(coord[:,:,1]),np.amax(coord[:,:,1]))
    ax.set_zlim(np.amin(coord[:,:,2]),0)
    
    #    ax.axis('equal')
    for i in range(len(coord)):
        rect = a3.art3d.Poly3DCollection(np.array([coord[i,:,:]]))
        colorval = scalarMap.to_rgba(slp[i])
        rect.set_facecolor(colorval)
        rect.set_edgecolors('white')
        rect.set_linewidth(0.5)
    #        tri.set_edgecolor('k')
        ax.add_collection3d(rect)
    
#    line = a3.art3d.Line3DCollection(np.array([[coord[0,0,:],coord[nstrike-1,1,:]]]))
#    line.set_edgecolor('lightgray')
#    line.set_linewidth(1)
#    ax.add_collection3d(line)
    
#    extent = [np.amin(coord[:,:,0])-1,np.amax(coord[:,:,0])+1,np.amax(coord[:,:,1])-1,np.amin(coord[:,:,1])+1]
##    m = Basemap(llcrnrlat=np.amin(coord[:,:,1]),urcrnrlat=np.amax(coord[:,:,1]),llcrnrlon=np.amin(coord[:,:,0]),urcrnrlon=np.amax(coord[:,:,0]),ax=ax,resolution='h')
#    m = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[2],
#             urcrnrlon=extent[1], urcrnrlat=extent[3],ax=ax,resolution='h',projection='cyl')
#    ax.add_collection3d(m.drawcoastlines(linewidth=0.25))
#    ax.add_collection3d(m.drawcountries(linewidth=0.25))
    
    colo=plt.colorbar(scalarMap,orientation='vertical',fraction=0.0085,pad=-0.185)
    colo.set_ticks([0,valmax/2,valmax])  
    ax=colo.ax
    ax.text(-0.6,0.5,'Slip (m)',rotation='vertical',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    
    plt.savefig(savedir+filename+'_slip3d_'+slip+'.pdf', format='pdf',bbox_inches="tight")
    plt.show() 
    plt.clf() 
    
    return

def plotFault3D(name,faults,azimuth,elev,slip=None, valmax=None, slipdir=None, sigma=False, savedir='./'):
    '''
    step
    faultparams
    coord: tab with lon, lat, z of all the faults
    azimuth
    valmax: slip max value for colorscale
    slip: 'total' 'dip' or 'strike'
    '''
    plt.rcParams['grid.color'] = 'ghostwhite'
        
    fig = plt.figure(1,figsize=(10,5))
    ax = a3.Axes3D(fig)
     
    for fault in faults:
        fault.patchll = np.array(fault.patchll)
        fault.patchll[:,:,2] = -fault.patchll[:,:,2]
    ax.azim = azimuth
    ax.elev=elev
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.1
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.1
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.1
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel('\n Longitude')
    ax.set_ylabel('\n Latitude')
    ax.set_zlabel('\n Depth (km)')
    lons = [f.patchll[:,:,0] for f in faults]
    lats = [f.patchll[:,:,1] for f in faults]
    deps = [f.patchll[:,:,2] for f in faults]
    plt.xlim(min([np.amin(l) for l in lons]),max([np.amax(l) for l in lons]))
    plt.ylim(min([np.amin(l) for l in lats]),max([np.amax(l) for l in lats]))
    ax.set_zlim(-np.amax(np.abs([np.amin(l) for l in deps]),0))
    
    totslip = [np.sqrt(f.slip[:,0]**2 + f.slip[:,1]**2) for f in faults]
    cmap.create_cmap(colorsco, 'cptslip')
    cm = plt.get_cmap('cptslip') 
    if valmax is not None:
        cNorm  = colors.Normalize(vmin=0, vmax=valmax)
    elif slip is not None:
        cNorm  = colors.Normalize(vmin=0, vmax=np.amax( totslip ))
        valmax=np.amax( totslip )
    else:
        cNorm  = colors.Normalize(vmin=0, vmax=np.amax(-fault.patchll[:][0,2]))
        valmax=np.amax(-fault.patchll[:][0,2])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    scalarMap.set_array([])
    
    k=0
    for fault in faults:
        k+=1
        if slip is not None:
            if slip in ('strikeslip','ss','strike-slip'):
                slp = fault.slip[:,0].copy()
            elif slip in ('dipslip','ds','dip-slip'):
                slp = fault.slip[:,1].copy()
            elif slip in ('tensile'):
                slp = fault.slip[:,2].copy()
            elif slip in ('total','tot'):
                slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
        else:
            slp = fault.slip[:,0].copy()
            
        for i in range(len(fault.patchll)):
            rect = a3.art3d.Poly3DCollection(np.array([fault.patchll[i,:,:]]))
            if slip is None:
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmo.tempo)
                scalarMap.set_array([])
#                import pdb; pdb.set_trace()
                colorval = scalarMap.to_rgba(10*k)
                rect.set_facecolor(colorval)
            else:
                colorval = scalarMap.to_rgba(slp[i])
                rect.set_facecolor(colorval)
            rect.set_edgecolors('white')
            rect.set_linewidth(0.1)
            ax.add_collection3d(rect)
            
        if slipdir is not None:
            if np.shape(faults)==() or np.shape(faults)[0]==1 :   
                rake = np.loadtxt(slipdir, comments='>')
            else:
                direc = slipdir.replace(name,name+'_'+fault.name)
                rake = np.loadtxt(direc, comments='>')
            for i in range(0,len(rake),2):
                ax.quiver(rake[i,2], rake[i,3], -rake[i,4],
                          rake[i+1,2], rake[i+1,3], -rake[i+1,4],
                          length = 10**-4*slp[i//2], normalize = True,
                          color='dimgrey', lw=10**-2.8*slp[i//2], 
                          arrow_length_ratio = 0.15,zorder=1000)
        
    
    colo=plt.colorbar(scalarMap,orientation='vertical',fraction=0.0085,pad=0.185)
    colo.set_ticks([0,valmax/2,valmax])  
    ax=colo.ax
    if slip is not None:
        ax.text(-0.6,0.5,'Slip (m)',rotation='vertical',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    else:
        ax.text(-0.6,0.5,'Depth (m)',rotation='vertical',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
        
    plt.savefig(savedir+name+'_'+fault.name+'_fault3d.pdf', format='pdf')
    plt.show()     
    return

def plotGfVar(diprg,gfs,rvalue,data_points,nd,ns,subfaults=False,savename='./gf_var'):
    '''
    plot the variation of the GFs for a certain range of fault geometry parameters
    
    data_points: [8,12]
    
    '''
    try:       
        rvalue = np.abs(rvalue)
        jet = cm = plt.get_cmap('inferno')
        cNorm  = colors.Normalize(0,1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='inferno')
        scalarMap.set_array([])  
    
        moy = np.mean(gfs,axis=0)
        
#        fig, ax1 = plt.subplots(1,1,figsize=(10,15))
#        for i in range(gfs[0].shape[0]):
#            for j in range(gfs[0].shape[1]):
#                colorval = scalarMap.to_rgba(rvalue[i,j])
#                plt.plot(range(diprg[0],diprg[1]),[gfs[k][i,j] for k in range(len(diprg))],color=colorval)
#        
#        plt.savefig(savedir+'gf_var.png', format='png', dpi=300,pad_inches=0.0)
#        plt.savefig(savedir+'gf_var.pdf', format='pdf')
#        plt.show()
#        plt.close()
#        plt.axis('off')
#        
        Np = ns * nd * 2
        
        if subfaults is True:
            fig= plt.figure(figsize=(15,17))
    #        
            for i in range(len(data_points)):
                fig.add_subplot(3,2,int(i)+1)
    #            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
    #                   labelbottom="off", left="off", right="off", labelleft="off") 
    #            rmat = np.abs(rvalue[data_points[i],Np//2:Np])
                rmat = np.multiply((1 - rvalue[data_points[i],Np//2:Np]),
                                   np.abs(moy)[data_points[i],Np//2:Np]/np.mean(np.max([np.abs(gfs)[k][:,Np//2:Np] for k in range(len(range(diprg[0],diprg[1])))], axis=1) ))
                rmat = rmat.reshape((nd,ns))
                sns.heatmap(rmat,cmap='inferno_r',cbar=False,vmin=0,vmax=1,linewidths=.0, rasterized=True,xticklabels=False, yticklabels=False)
    
            for i in range(len(data_points)):
                fig.add_subplot(3,2,int(i)+3)
    #            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
    #                   labelbottom="off", left="off", right="off", labelleft="off") 
    #            rmat = np.abs(rvalue[data_points[i]+ns,Np//2:Np])
                rmat = np.multiply((1 - rvalue[data_points[i]+ns,Np//2:Np]),
                                   np.abs(moy)[data_points[i]+ns,Np//2:Np]/np.mean(np.max([np.abs(gfs)[k][:,Np//2:Np] for k in range(len(range(diprg[0],diprg[1])))], axis=1) ))
                rmat = rmat.reshape((nd,ns))
                sns.heatmap(rmat,cmap='inferno_r',cbar=False,vmin=0,vmax=1,linewidths=.0, rasterized=True,xticklabels=False, yticklabels=False)
            
            for i in range(len(data_points)):
                fig.add_subplot(3,2,int(i)+5)
    #            plt.tick_params(axis="both", which="both", bottom="off", top="off",  
    #                   labelbottom="off", left="off", right="off", labelleft="off") 
    #            rmat = np.abs(rvalue[data_points[i]+ns*2,Np//2:Np])
                rmat = np.multiply((1 - rvalue[data_points[i]+ns*2,Np//2:Np]),
                                   np.abs(moy)[data_points[i]+ns*2,Np//2:Np]/np.mean(np.max([np.abs(gfs)[k][:,Np//2:Np] for k in range(len(range(diprg[0],diprg[1])))], axis=1) ))
                rmat = rmat.reshape((nd,ns))
                sns.heatmap(rmat,cmap='inferno_r',vmin=0,vmax=1,cbar=False,linewidths=.0, rasterized=True,xticklabels=False, yticklabels=False)
            plt.savefig(savename+'.pdf', format='pdf')
            plt.show()
            plt.close()

        else:
            fig=plt.figure(figsize=(17,15))
            i = data_points[0]
            J = np.where(rvalue[i,Np//2:Np] < 2)
            ax = fig.add_subplot(221)
            for j in J[0]:
                if abs(gfs[1][i,j+Np//2]) <= 100:
                    colorval = scalarMap.to_rgba(rvalue[i,j+Np//2])
                    plt.plot(diprg,[gfs[k][i,j+Np//2] for k in range(len(diprg))],color=colorval,label=str(j))
                    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            ax.set_ylabel("Amplitude of the Green\'s functions (m)")  
             
            i = data_points[1]
            J = np.where(rvalue[i,Np//2:Np] < 2)
            ax = fig.add_subplot(222)
            for j in J[0]:
                if abs(gfs[1][i,j+Np//2]) <= 100:
                    colorval = scalarMap.to_rgba(rvalue[i,j+Np//2])
                    plt.plot(diprg,[gfs[k][i,j+Np//2] for k in range(len(diprg))],color=colorval,label=str(j))
                    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                    
            i = data_points[0] + ns
            J = np.where(rvalue[i,Np//2:Np] < 2)
            ax = fig.add_subplot(223)
            for j in J[0]:
                if abs(gfs[1][i,j+Np//2]) <= 100:
                    colorval = scalarMap.to_rgba(rvalue[i,j+Np//2])
                    plt.plot(diprg,[gfs[k][i,j+Np//2] for k in range(len(diprg))],color=colorval,label=str(j))
                    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            ax.set_xlabel('Dip ($^\circ$)')
            ax.set_ylabel("Amplitude of the Green\'s functions (m)")  
    
            i = data_points[1] + ns
            J = np.where(rvalue[i,Np//2:Np] < 2)
            ax = fig.add_subplot(224)
            for j in J[0]:
                if abs(gfs[1][i,j+Np//2]) <= 100:
                    colorval = scalarMap.to_rgba(rvalue[i,j+Np//2])
                    plt.plot(diprg,[gfs[k][i,j+Np//2] for k in range(len(diprg))],color=colorval,label=str(j))
                    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            ax.set_xlabel('Dip ($^\circ$)')  
            
            fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.13, hspace=0.07)
            # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
            cb_ax = fig.add_axes([0.83, 0.25, 0.02, 0.5])
            fig.colorbar(scalarMap, cax=cb_ax)
            plt.savefig(savename+'.pdf', format='pdf')
            plt.show()


    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    return

def plotMu(mus,sigmamu,depths,savename='./mu.pdf'):
    '''
    '''
    try:
        fig, ax = plt.subplots(1,figsize=(2,3.5))
        ax.set_xlabel(r"$\mu$ (GPa)")
        ax.set_ylabel('\n Depth (km)')
        
        cumdepths = -np.cumsum(depths)
        depth = [0]+ list(np.repeat(cumdepths,2))
        x = np.linspace(0,80,160)
        sigmalog = [sigmamu[l]/((l+1)*4) for l in range(len(mus))]
        import scipy.stats as st
        for l in range(len(mus)):
            rv = st.lognorm(sigmalog[l],scale=mus[l])
            if l >=1. and depths[l]<10.:
                d = cumdepths[l-1]-depths[l]/1.3
            elif l >=1. and depths[l]>=10.:   
                d = cumdepths[l-1]-depths[l]/4.
            else:
                d = -depths[l]/1.3
            plt.fill(x[rv.pdf(x)>10**-3],d+5*rv.pdf(x)[rv.pdf(x)>10**-3],color=mycyan,alpha=0.5)
        plt.plot(np.repeat(mus,2),depth[:-1],mycyan)
        plt.ylim([-20.,1.])
        plt.xlim([5.,65.])
        plt.locator_params(axis='y', nbins=5)
        plt.show()
        plt.savefig(savename,format='pdf')
    
    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    return  sigmalog
   
def plotM0(faultres, mu0=3., sigma_mu = 0.2):
    '''
    M0 in N.m
    
    moy: slip in m
    surface is an array, or a float
    '''
    import scipy.stats as stats
    mu = mu0*10**10
    mus = sigma_mu*10**10
    
    SDs = np.sum(faultres.samp[:,0:faultres.N_slip]/100.*faultres.area*1e6, axis=1)
    SDd = np.sum(faultres.samp[:,faultres.N_slip:faultres.N_slip*2]/100.*faultres.area*1e6, axis=1)
    SD = SDs + SDd
    
    M0co = SD * mu
    x0 = np.linspace(np.amin(M0co),np.amax(M0co),500)
    M0hist = np.histogram(M0co,bins=x0)[0]
#    plt.plot(x0[:-1], M0hist/np.amax(M0hist))
    
    
    M0co2 = np.mean(SD)*mu
    std2 = np.sqrt( (np.std(SD)**2+np.mean(SD)**2)*(mus**2+mu**2) - mu**2*np.mean(SD)**2 )
    x0 = np.linspace(np.amin(M0co)-std2*2,np.amax(M0co)+std2*2,500)
    st=stats.norm.pdf(x0[:-1], M0co2, std2)
#    plt.plot(x0[:-1], st/np.amax(st))
#    cov1 = surface*(cov[0:len(moy)])
#    cov2 = surface*(cov[len(moy)-1:-1])
#    M0cosd = (0.5* np.median(cov1) + 0.5* np.median(cov2) ) * 3.*10**10 
##    pdb.set_trace()
#    Mw_eq = (2./3.)*np.log10(M0co* 10**7)-10.7
#    Mw_cov = (2./3.)*np.log10(M0cosd* 10**7)-10.7
#    
    print('Moment: '+str(np.mean(M0co))+' +- '+str(np.std(M0co)))
#    print('Equivalent magnitude: '+str(Mw_eq)+' +- '+str(Mw_cov))
    return M0co, std2


def exportSlideResults(resdir, resname, targetname, text=None, names=None):
    '''
    resname can be none, in this case enter all filenames in names, as list of strings
    targetname is path + name
    '''
    matplotlib.rcParams.update({'font.size': 12})

    tgt = Image.open(targetname)
    slip = Image.open(os.path.join(resdir,resname+'_Fault_slipbivcont.png'))
    data = Image.open(os.path.join(resdir,resname+'_gps_synth.png'))
    
    fig=plt.figure(figsize=(15,9))
    a=plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=1)
    a.spines["top"].set_visible(False)   
    a.spines["right"].set_visible(False)  
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    plt.imshow(tgt)
    a.text(1000.,1000.,'(a)',
           fontweight='bold',horizontalalignment='center',
           verticalalignment='center')
    plt.axis('off')
    box = a.get_position()
    box.x0 = box.x0 + 500
    box.x1 = box.x1 + 500
    a.set_position(box)
    
    a=plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
    a.spines["top"].set_visible(False)   
    a.spines["right"].set_visible(False)  
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    
    if text is not None:
        a.text(0.5,0.5,'Topographic step: '+text[0],
               fontweight='bold',horizontalalignment='center',
               verticalalignment='center')
        a.text(0.5,0.3,'Topographic slope: '+text[1],
               fontweight='bold',horizontalalignment='center',
               verticalalignment='center')
    plt.axis('off')
    a=plt.subplot2grid((2,3), (1,0), colspan=2, rowspan=1)
    a.spines["top"].set_visible(False)   
    a.spines["right"].set_visible(False)  
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    plt.imshow(slip)
    a.text(200.,0.,'(b)',
           fontweight='bold',horizontalalignment='center',
           verticalalignment='center')
    plt.axis('off')
    
    a=plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=3)
    a.spines["top"].set_visible(False)   
    a.spines["right"].set_visible(False)  
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    plt.imshow(data)
    a.text(0.,0.,'(c)',
           fontweight='bold',horizontalalignment='center',
           verticalalignment='center')
    plt.axis('off')
    
    fig.subplots_adjust(hspace=0.1) 
    plt.tight_layout()
    plt.savefig(resdir+'test.pdf', format='pdf', pad_inches=0.0)
    plt.savefig(resdir+'test.png', format='png',bbox_inches="tight",dpi=300)
    plt.show()
    return


def exportSlideResultsTents(resdir, resname, targetname, text=None, names=None):
    '''
    resname can be none, in this case enter all filenames in names, as list of strings
    targetname is path + name
    '''
    matplotlib.rcParams.update({'font.size': 12})

    tgt = Image.open(targetname)
    slip = Image.open(os.path.join(resdir,resname+'_Fault_slipbivcont.png'))
    data = Image.open(os.path.join(resdir,resname+'_gps_synth.png'))
    
    fig=plt.figure(figsize=(15,9))
    a=plt.subplot2grid((2,12), (0,1), colspan=4, rowspan=1)
    a.spines["top"].set_visible(False)   
    a.spines["right"].set_visible(False)  
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    plt.imshow(tgt)
    a.text(250.,-50.,'(a)',
           fontweight='bold',horizontalalignment='center',
           verticalalignment='center')
    
    if text is not None:
        a.text(250,-400,'GF forward: '+text[0],
               fontweight='bold',horizontalalignment='left',
               verticalalignment='center')
        a.text(250,-300,'GF inverse: '+text[1],
               fontweight='bold',horizontalalignment='left',
               verticalalignment='center')
        a.text(250,-200,'Cp: '+text[2],
               fontweight='bold',horizontalalignment='left',
               verticalalignment='center')
        
    box = a.get_position()
    box.x0 = box.x0 + 500
    box.x1 = box.x1 + 500
    a.set_position(box)
    plt.axis('off')
    
#    a=plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
#    a.spines["top"].set_visible(False)   
#    a.spines["right"].set_visible(False)  
#    a.spines["left"].set_visible(False)
#    a.spines["bottom"].set_visible(False)
#    
#    if text is not None:
#        a.text(0.5,0.7,'GF forward: '+text[0],
#               fontweight='bold',horizontalalignment='center',
#               verticalalignment='center')
#        a.text(0.5,0.5,'GF inverse: '+text[1],
#               fontweight='bold',horizontalalignment='center',
#               verticalalignment='center')
#        a.text(0.5,0.3,'Cp: '+text[2],
#               fontweight='bold',horizontalalignment='center',
#               verticalalignment='center')
#    plt.axis('off')
    
    a=plt.subplot2grid((2,12), (1,0), colspan=12, rowspan=1)
    a.spines["top"].set_visible(False)   
    a.spines["right"].set_visible(False)  
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    plt.imshow(slip)
    a.text(0.,0.,'(c)',
           fontweight='bold',horizontalalignment='center',
           verticalalignment='center')
    plt.axis('off')
    
    a=plt.subplot2grid((2,12), (0,5), colspan=6, rowspan=1)
    a.spines["top"].set_visible(False)   
    a.spines["right"].set_visible(False)  
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    plt.imshow(data)
    a.text(-50.,0.,'(b)',
           fontweight='bold',horizontalalignment='center',
           verticalalignment='center')
    plt.axis('off')
    
    fig.subplots_adjust(hspace=0.1,wspace=0.05) 
    plt.tight_layout()
    plt.savefig(resdir+'test.pdf', format='pdf', pad_inches=0.0)
    plt.savefig(resdir+'test.png', format='png',bbox_inches="tight",dpi=300)
    plt.show()
    plt.close()
    return

def plotGPSvert(name, geoData, synths, savedir='./', valmin=-3., valmax=3.):
    '''
     
    '''
    try:
        
        # get topo
        x = np.arange(-100.,450., 3)
        y = np.arange(-310., 310., 3)
        X, Y = np.meshgrid(x, y)
        topo = return_topo(X)
        
#        colors_topo_hex = ['#003547', '#286477', '#5595a9', '#8ac9de', '#f5f5f5', '#ebebeb', '#e1e1e1', '#d7d7d7', '#cdcdcd']
        colors_topo_hex = ['#3f7f93', '#5893a2', '#72a8b3', '#8ebcc4', '#acd0d6', '#ededed', '#e5e5e5', '#dddddd', '#d5d5d5', '#cdcdcd']                  
        colors_topo = [ImageColor.getcolor(i, "RGB") for i in colors_topo_hex]  
        colors_topo = [(y[0]/255., y[1]/255., y[2]/255.) for y in colors_topo]
        topocmap = colors.LinearSegmentedColormap.from_list('topo',colors_topo, N=256)
        cNorm = MidpointNormalize(vmin=-6., vcenter=0.1, vmax=3.)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=topocmap)
        scalarMap.set_array([])
        topocmap = colors.LinearSegmentedColormap.from_list('ctopo',scalarMap.to_rgba(range(-6,3)), 256)
                  
        fig, ax = plt.subplots(1,figsize=(2.7,3))
        # fig, ax = plt.subplots(1,figsize=(8,5))
        
        ax.set_xlabel('Distance from trench (km)', horizontalalignment='left', x=0.15)
        ax.set_ylabel('Distance along strike (km)')
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        
        #background
#        ax.add_patch(patches.Rectangle((-100, -300), 220, 600, color='#bcdede'))
        # ax.pcolor(X,Y,topo, cmap=topocmap)                                       
        # ax.add_patch(patches.Rectangle((120, -300), 850, 600, color='#f1f3f4'))             
        # ax.plot([120,120],[-300.,300], lw=1, c='#cdcdcd')
        # ax.plot([0,0],[-300.,300], lw=1, c='#003547',zorder=11)
                 
        # surface defo
#        for data in geoData:
        data=geoData[0]
        # sc = plt.scatter(data.x, data.y, c=synths[data.name][:,2], s=6., cmap=cmo.curl, vmin=valmin, vmax=valmax, zorder=50)
        
        x = np.arange(-100.,450., 1)
        y = np.arange(-310., 310., 1)
        X, Y = np.meshgrid(x, y)
        
        disp = synths[data.name][:,2]
        # disp = synths[data.name][:,0]
        zi = sciint.griddata((data.x, data.y),disp,(X,Y),method='cubic')
        cmap=plt.pcolormesh(X,Y,zi,shading='gouraud', zorder=10,cmap='RdBu_r',vmin=valmin,vmax=valmax,rasterized=True)
        cbaxes = ax.inset_axes([0.4, 0.1, 0.5, 0.03], transform=ax.transAxes,zorder=20)      
        cbar = plt.colorbar(cmap, cax=cbaxes, label='Vertical displacement (m)',orientation='horizontal')
        cbar.ax.locator_params(nbins=3)
        plt.xlim(-50,300)
        plt.ylim(-200.,200.)
        plt.savefig(savedir+name+'_vertdef.pdf', format='pdf',bbox_inches="tight")
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    
    return


def plotGPS(name, data, synths, savedir='./', valmin=-1., valmax=1.):
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
                  
        fig, ax = plt.subplots(1,figsize=(3,5))
        ax.set_xlabel('Distance from trench (km)')
        ax.set_ylabel('Distance along strike (km)')
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        
        #background
        ax.add_patch(patches.Rectangle((-100, -300), 220, 600, color='#c4dade'))
#        ax.pcolor(X,Y,topo, cmap=topocmap)                                       
        ax.add_patch(patches.Rectangle((120, -300), 850, 600, color='#f1f3f4'))             
        ax.plot([120,120],[-300.,300], lw=1, c='#cdcdcd')
        ax.plot([0,0],[-300.,300], lw=2, c='#003547')
                 
        # surface defo
        plt.quiver(data.x, data.y, 
                   synths[data.name][:,0], 
                   synths[data.name][:,1],
#                   units = 'width', width = 0.2,
    #                  scale = None, 
    #                  scale_units='inches',
#                   scale = 2.5, 
                   scale_units = 'xy',
                   scale=3.e-2,
                   width = 0.005,
                   color='dimgrey' , zorder=5)
        sc = plt.scatter(data.x, data.y, c=synths[data.name][:,2], s=6., cmap=cmo.curl, vmin=valmin, vmax=valmax, zorder=6)
        
        plt.quiver([90.], [280.],
                  [-2.], 
                   [0.],
#                   units = 'width', width = 0.2,
    #                  scale = None, 
    #                  scale_units='inches',
#                   scale = 2.5, 
                   scale_units = 'xy',
                   scale=3.e-2,
                   width = 0.005,
                   color='dimgrey' , zorder=5)
        plt.text(50., 260., '2 m', color='dimgrey',weight='light',fontsize=7, zorder=5) 
        cbaxes = ax.inset_axes([0.55, 0.95, 0.4, 0.02], transform=ax.transAxes)      
        plt.colorbar(sc, cax=cbaxes, label='Vertical displacement (m)',orientation='horizontal')
        plt.xlim(-10.,300.)
        plt.ylim(-300.,300.)
        plt.savefig(savedir+name+'_surfacedef.pdf', format='pdf',bbox_inches="tight")
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

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
        
def plotGPSwSlipBivTent(name,fault, sigma, data, synths,slip=None, valmax=None, sigmamax=None, slipdir=None, savedir='./',epicenter=None, xystrides=[100, 100], vertbound=[-1.,1.], shoreline=120.,ylim=[-210.,210.],xlim=[-10.,500.], trueslip=False):
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
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
            sigma = np.array(sigma)
            sgm = np.sqrt(sigma[0:len(sigma)//2]**2 + sigma[len(sigma)//2:len(sigma)]**2)
        elif slip=='sigma':
            slp = 7*np.array(sigma[len(sigma)//2:len(sigma)])
            sgm = sigma[len(sigma)//2:len(sigma)]
        else:
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)        
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
        cptslip = colors.LinearSegmentedColormap.from_list('cptslip',colorsco_above2_rgba, N=256)
        cNorm = MidpointNormalize(vmin=0., vcenter=slipmax, vmax=slipmax+5.)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cptslip)
        scalarMap.set_array([])
#        cmap = cptslip
#        mycmap = cmap(np.arange(cptslip.N))
#        mycmap[:,-1] = np.hstack((np.zeros((cmap.N//4,)),np.linspace(0.,1.,cmap.N-cmap.N//4)))
#        cptslip2 = colors.ListedColormap(mycmap)
        
        cptsig = colors.LinearSegmentedColormap.from_list('cptsig',sigma_grey_transp_rgba, N=256)
        cmap = cptsig
        mycmap = cmap(np.arange(cptsig.N))
#        alph1 = np.hstack((np.zeros((cmap.N//4,)),np.linspace(0.,1.,cmap.N-cmap.N//2)))
#        mycmap[:,-1] = np.hstack((alph1, np.ones((cmap.N//4,)) ))
        mycmap[:,-1] = np.hstack((np.zeros((cmap.N//4,)),np.linspace(0.,1.,cmap.N-cmap.N//4)))
        cptsig2 = colors.ListedColormap(mycmap)
        cNorm2  = colors.Normalize(vmin=0., vmax=uncmax)
        scalarMap2 = cmx.ScalarMappable(norm=cNorm2, cmap=cptsig)
        scalarMap2.set_array([])

        fault.tent = np.array(fault.tent)
        x0 = np.amin(fault.tent[:,0])
        ind = np.argmin(fault.tent[:,0], axis=None)
        y0 = fault.tent[ind,1]
        x = []
        y = []
        for i in range(np.shape(fault.tent)[0]):
            xx = fault.tent[i,0]
            yy = fault.tent[i,1]
            d = y
            x.append(xx)
            y.append(yy)
        x = np.vstack(x)
        y = np.vstack(y)
        
        # triangles
        vertices = fault.Vertices.tolist()
        faces = fault.Faces
        zt = []
        dist = []
        for face in faces:
            verts = [vertices[f] for f in face]
            x = [v[0] for v in verts]
            y = [v[1] for v in verts]
            z = [v[2] for v in verts]
            zt.append(x); dist.append(y)
        
        fig, ax = plt.subplots(1,figsize=(3.4,6))
        ax.set_xlabel('Distance from trench (km)')
        ax.set_ylabel('Distance along strike (km)')
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=5)
        
        #background
#        ax.add_patch(patches.Rectangle((-100, -300), 220, 600, color='#c4dade'))
#        ax.pcolor(X,Y,topo, cmap=topocmap)                                       
#        ax.add_patch(patches.Rectangle((shoreline, -300), 850, 600, color='#f1f3f4'))             
        
        ## Plot slip
        if not hasattr(fault, 'plotSources'):
            print('--------------------------------------')
            print('Please precompute sources for plotting')
        Ids = fault.plotSources[0]
        X = fault.plotSources[1]
        Y = fault.plotSources[2]
        Z = fault.plotSources[3]

        Slip = fault._getSlipOnSubSources(Ids, X, Y, Z, slp)
        Sigma = fault._getSlipOnSubSources(Ids, X, Y, Z, sgm)
        D = Y
        diss2 = np.linspace(np.nanmin(D), np.nanmax(D), 300)
        Z2 = np.sqrt((X-x0)**2 + (Z)**2)
        
        # add fake deep slip row for plot
        # D = np.hstack((D,np.linspace(-250,250,50)))
        # D = np.hstack((D,np.linspace(-250,250,50)))
        # Z2 = np.hstack((Z2,np.linspace(250,250,50)))
        # Z2 = np.hstack((Z2,np.linspace(260,260,50)))
        # Slip = np.hstack((Slip,np.linspace(0,0,50)))
        # Slip = np.hstack((Slip,np.linspace(0,0,50)))
        # Sigma = np.hstack((Sigma,np.linspace(0,0,50)))
        # Sigma = np.hstack((Sigma,np.linspace(0,0,50)))
        
        z2 = -np.linspace(np.nanmin(Z2)-5., np.nanmax(Z2)+25.,300)      
        diss2, z2 = np.meshgrid(diss2,z2)
        slip22 = sciint.griddata((D,-Z2),Slip,(diss2,z2),method='linear')
        sigma22 = sciint.griddata((D,-Z2),Sigma,(diss2,z2),method='linear')
        import scipy.ndimage as ndimage
        slip2 = ndimage.median_filter(slip22,size=(1,1))
        sigma2 = ndimage.median_filter(sigma22,size=(1,1))
           
        cslip = colors.LinearSegmentedColormap.from_list('cslip',scalarMap.to_rgba(range(0,int(valmax)+5)), 256)
        if slip=='sigma':
            pcol = ax.pcolor(-z2, diss2, slip2, cmap=cmo.deep, vmin=0, vmax=slipmax,rasterized = True)
        else:
            ax.pcolor(-z2, diss2, slip2, cmap=cslip, vmin=0, vmax=slipmax,rasterized = True)
        
        ## Plot triangles
        rects = []
        for i in range(len(dist)):
            vertex = np.vstack((np.array(zt[i]),dist[i]))
            rect = patches.Polygon( vertex.T )
            rects.append(rect)
        p = PatchCollection(rects, facecolors='None', edgecolor = 'white', lw=0.2)
        ax.add_collection(p)
        
        if slip!='sigma':
            ax.pcolor(-z2, diss2,sigma2, cmap=cptsig2, vmin=0, vmax=uncmax, edgecolors=None,rasterized = True)
    #        sig = ax.pcolor(-z2, diss2,sigma2, cmap=cptsig, vmin=0, vmax=uncmax, edgecolors=None,rasterized = True)
    #        for i,j in zip(sig.get_facecolors(),sigma2.flatten()/np.amax(sigma2)):
    #            i[3] = j # Set the alpha value of the RGBA tuple using m2                               
        
        
        if trueslip is not False:
            # plot true patches
            if 'A' in trueslip:
                for i in range(4):
                    pat = np.loadtxt('/home/thea/projet/cp3d2_synth/fig/tmp/patch'+str(i+1)+'.txt')
                    x_cor=pat[:,0]
                    y_cor=pat[:,1]
                    plt.plot(x_cor, y_cor, c='#b2120b', lw=1)
            elif 'B' in trueslip:
                for i in [2,4]:
                    pat = np.loadtxt('/home/thea/projet/cp3d2_synth/fig/tmp/patch'+str(i)+'.txt')
                    x_cor=pat[:,0]
                    y_cor=pat[:,1]
                    plt.plot(x_cor, y_cor, c='#b2120b', lw=1)
                                                        
        # trench and shoreline                               
        ax.plot([shoreline,shoreline],[-300.,300], lw=1, c='#cdcdcd')
#        ax.plot([0,0],[-300.,300], lw=2, c='#003547')
                 
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
                    color='#1a5665' )
        
        if slip=='sigma':
           plt.scatter(data.x, data.y, c='gray', s=20) 
        else:
            sc = plt.scatter(data.x, data.y, c=data.vel_enu[:,2], s=80, cmap='RdBu_r', vmin=vertbound[0], vmax=vertbound[1])
            sc = plt.scatter(data.x, data.y, c=synths[data.name][:,2], s=25, lw=0.1, edgecolors='white', cmap='RdBu_r', vmin=vertbound[0], vmax=vertbound[1])
            cbaxes = ax.inset_axes([0.55, 0.95, 0.4, 0.02], transform=ax.transAxes)      
            plt.colorbar(sc, cax=cbaxes, label='Vertical displacement (m)',orientation='horizontal')  
            
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
                    color='dimgrey')
        plt.text(xlim[0]+60., ylim[1]-30., '2 m', color='dimgrey',weight='light',fontsize=7) 
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])

    ## bivariate colorbar
        if slip != 'sigma':
            cax = fig.add_axes([0.8, 0.02, 0.1, 0.3])
            xx, yy = np.mgrid[0:slipmax+5:100j,0:uncmax:30j]
            C_map = scalarMap.to_rgba(xx)
            cax.imshow(C_map)
            yy_plot = np.array(255*(yy-yy.min())/(yy.max()-yy.min()), dtype=np.int)
            C_map2  = cptsig2(yy_plot)
            cax.imshow(C_map2)
            cax.set_ylim((-0.,95)  )   
            cax.set_xlim((-0.,29)  )  
            if uncmax <= 2:
                cax.locator_params(axis='x', nbins=2)
                cax.xticks = [0., uncmax/2]
                x_label_list = ['0',"{:1.1f}".format(uncmax/2.)]            
            else:
                cax.locator_params(axis='x', nbins=3)
                x_label_list = ['0',"{:1.0f}".format(uncmax/3.),"{:1.0f}".format(2*uncmax/3.)]
            cax.locator_params(axis='y', nbins=4)
            y_label_list = ['0',"{:1.0f}".format(slipmax/4.),"{:1.0f}".format(2*slipmax/4.),"{:1.0f}".format(4*slipmax/4.)]
            cax.set_xticklabels(x_label_list)
            cax.set_yticklabels(y_label_list)
            cax.yaxis.set_label_position("right")
            cax.yaxis.tick_right()
            cax.set_ylabel('Slip (m)')
            cax.set_xlabel('σ (m)')
            
            plt.savefig(savedir+name+'_slipgps.pdf', format='pdf',bbox_inches="tight")
            plt.show()   
            
        else:
            cax = fig.add_axes([0.55, 0.15, 0.2, 0.02])
            plt.colorbar(pcol, cax=cax,orientation='horizontal')        
            cax.set_title('Posterior σ (m)', fontweight='normal',fontsize='small')

            plt.savefig(savedir+name+'_sigmagps.pdf', format='pdf',bbox_inches="tight")
            plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def plotGPSwSlipBivTri(name,fault, sigma, data, synths,slip=None, valmax=None, sigmamax=None, slipdir=None, savedir='./',epicenter=None, xystrides=[100, 100], vertbound=[-1.,1.], shoreline=120.,ylim=[-210.,210.],xlim=[-10.,500.]):
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
            uncmax = 0.6*np.amax(sgm)
        else:
            uncmax = sigmamax
        
        ## colorbars
        cptslip = colors.LinearSegmentedColormap.from_list('cptslip',colorsco_above2_rgba, N=256)
        cNorm = MidpointNormalize(vmin=0., vcenter=slipmax, vmax=slipmax+5.)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cptslip)
        scalarMap.set_array([])
#        cmap = cptslip
#        mycmap = cmap(np.arange(cptslip.N))
#        mycmap[:,-1] = np.hstack((np.zeros((cmap.N//4,)),np.linspace(0.,1.,cmap.N-cmap.N//4)))
#        cptslip2 = colors.ListedColormap(mycmap)
        
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
            x = np.sqrt((fault.patch[i,:,0]-x0)**2 + (fault.patch[i,:,2])**2)    
            y = fault.patch[i,:,1]
            dis.append(x)
            dep.append(y)
        
        fig, ax = plt.subplots(1,figsize=(3.4,6))
        ax.set_xlabel('Distance from trench (km)')
        ax.set_ylabel('Distance along strike (km)')
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.yticks(rotation='vertical')
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
                                       
        # trench and shoreline                               
        ax.plot([shoreline,shoreline],[-300.,300], lw=1.5, c='darkgray', zorder=4)
        plt.text(shoreline-5, 240., 'shoreline', color='w', backgroundcolor='darkgray',fontsize='x-small', rotation=90, zorder=5)
        ax.plot([0,0],[-300.,300], lw=1.5, c='#215d9e', zorder=4)
        plt.text(-5, 260., 'trench', color='w', backgroundcolor='#215d9e',fontsize='x-small', rotation=90, zorder=5)
                 
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
                    color='dimgrey' , zorder=8)
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
                    color='#1a5665' , zorder=9)
        
        sc = plt.scatter(data.x, data.y, c=data.vel_enu[:,2], s=80, cmap=cmo.delta_r, vmin=vertbound[0], vmax=vertbound[1], zorder=6)
        sc = plt.scatter(data.x, data.y, c=synths[data.name][:,2], s=25, lw=0.6, edgecolors='white', cmap=cmo.delta_r, vmin=vertbound[0], vmax=vertbound[1], zorder=7)
        
        plt.quiver([xlim[0]+70.], [ylim[0]+45.],
                  [-1.], 
                   [0.],
#                   units = 'width', width = 0.2,
    #                  scale = None, 
    #                  scale_units='inches',
#                   scale = 2.5, 
                   scale_units = 'xy',
                   scale=3.e-2,
                   width = 0.003,
                   color='k', zorder=15)
    #     plt.text(xlim[0]+60., ylim[1]-20., '2 m', color='dimgrey',weight='light',fontsize=7) 
    #     cbaxes = ax.inset_axes([0.65, 1, 0.4, 0.02], transform=ax.transAxes)      
    #     plt.colorbar(sc, cax=cbaxes,orientation='horizontal')        
    #     cbaxes.set_title('Vertical displacement (m)', fontweight='normal',fontsize='small')
    #     plt.xlim(xlim[0],xlim[1])
    #     plt.ylim(ylim[0],ylim[1])

    # ## bivariate colorbar
    #     cax = fig.add_axes([0.8, 0.02, 0.1, 0.3])
    #     xx, yy = np.mgrid[0:slipmax+5:100j,0:uncmax:30j]
    #     C_map = scalarMap.to_rgba(xx)
    #     cax.imshow(C_map)
    #     yy_plot = np.array(255*(yy-yy.min())/(yy.max()-yy.min()), dtype=np.int)
    #     C_map2  = cptsig2(yy_plot)
    #     cax.imshow(C_map2)
    #     cax.set_ylim((-0.,95)  )   
    #     cax.set_xlim((-0.,29)  )  
    #     cax.locator_params(axis='x', nbins=3)
    #     cax.locator_params(axis='y', nbins=4)
    #     x_label_list = ['0',"{:1.0f}".format(uncmax/3.),"{:1.0f}".format(2*uncmax/3.)]
    #     y_label_list = ['0',"{:1.0f}".format(slipmax/4.),"{:1.0f}".format(2*slipmax/4.),"{:1.0f}".format(4*slipmax/4.)]
    #     cax.set_xticklabels(x_label_list)
    #     cax.set_yticklabels(y_label_list)
    #     cax.yaxis.set_label_position("right")
    #     cax.yaxis.tick_right()
    #     cax.set_ylabel('Slip (m)')
    #     cax.set_xlabel('σ (m)')
        plt.text(xlim[0]+40., ylim[0]+50., '1 m ', color='k',fontsize='x-small', zorder=506) 
        plt.text(xlim[0]+40., ylim[0]+25., 'data', color='dimgrey',fontsize='small', zorder=506) 
        plt.text(xlim[0]+40, ylim[0]+10, 'predictions', color='#1a5665',fontsize='small', zorder=500) 
        ax.add_patch(
     patches.Rectangle(
        (xlim[0]+60, ylim[0]+5),
        40,
        20,
        facecolor='w', zorder=20 ))
        plt.scatter(xlim[0]+25, ylim[0]+20, s=80, 
                         c='dimgrey', vmin=vertbound[0], vmax=vertbound[1],
                         zorder = 500)
        plt.scatter(xlim[0]+25, ylim[0]+20, c='#62a9bc', s=25,
                         lw=0.5, edgecolors='white',
                         vmin=vertbound[0], vmax=vertbound[1],
                         zorder = 501)
        cbaxes = ax.inset_axes([0.65, 0.05, 0.3, 0.02], transform=ax.transAxes)      
        plt.colorbar(sc, cax=cbaxes,orientation='horizontal')        
        cbaxes.set_title('Vertical disp. (m)', fontweight='normal',fontsize='small', zorder=500)
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])

    ## bivariate colorbar
        cax = fig.add_axes([0.55, 0.83, 0.3, 0.08])
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
        x_label_list = ['0',"{:1.0f}".format(slipmax/4.),"{:1.0f}".format(2*slipmax/4.),"{:1.0f}".format(4*slipmax/4.)]
        cax.set_xticklabels(x_label_list)
        cax.set_yticklabels(y_label_list)
        cax.yaxis.set_label_position("right")
        cax.yaxis.tick_right()
        cax.set_title('Slip (m)',fontsize='small')
        cax.set_ylabel('σ (m)')
        
        
        plt.savefig(savedir+name+'.pdf', format='pdf',bbox_inches="tight")
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
 
def plotGPSwEntTri(name,fault, data, synths,slip=None, valmax=None, sigmamax=None, slipdir=None, savedir='./',epicenter=None, xystrides=[100, 100], vertbound=[-1.,1.], shoreline=120.,ylim=[-210.,210.],xlim=[-10.,500.]):
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
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
        else:
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)        
        if valmax is None:
            slipmax = np.amax(slp)
        else:
            slipmax = valmax

        fault.patch = np.array(fault.patch)
        x0 = np.amin(fault.patch[:,:,0])
    #    ind = np.unravel_index(np.argmin(fault.patch[:,:,0], axis=None), fault.patch[:,:,0].shape)
    #    y0 = fault.patch[ind[0],ind[1],1]
        y0 = np.amin(fault.patch[:,:,1])
        dis = []
        dep = []
        for i in range(np.shape(fault.patch)[0]):
            x = np.sqrt((fault.patch[i,:,0]-x0)**2 + (fault.patch[i,:,2])**2)    
            y = fault.patch[i,:,1]
            dis.append(x)
            dep.append(y)
        
        fig, ax = plt.subplots(1,figsize=(3.4,6))
        ax.set_xlabel('Distance from trench (km)')
        ax.set_ylabel('Distance along strike (km)')
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        
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
                                       
        # trench and shoreline                               
        ax.plot([shoreline,shoreline],[-300.,300], lw=1, c='#cdcdcd')
#        ax.plot([0,0],[-300.,300], lw=2, c='#003547')
                 
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
                    color='#1a5665' )
        
        sc = plt.scatter(data.x, data.y, c=data.vel_enu[:,2], s=80, cmap=cmo.delta_r, vmin=vertbound[0], vmax=vertbound[1])
        sc = plt.scatter(data.x, data.y, c=synths[data.name][:,2], s=25, lw=0.2, edgecolors='white', cmap=cmo.delta_r, vmin=vertbound[0], vmax=vertbound[1])
        
        plt.quiver([xlim[0]+100.], [ylim[1]-10.],
                  [-2.], 
                   [0.],
#                   units = 'width', width = 0.2,
    #                  scale = None, 
    #                  scale_units='inches',
#                   scale = 2.5, 
                   scale_units = 'xy',
                   scale=3.e-2,
                   width = 0.003,
                   color='dimgrey')
    #     plt.text(xlim[0]+60., ylim[1]-20., '2 m', color='dimgrey',weight='light',fontsize=7) 
    #     cbaxes = ax.inset_axes([0.65, 1, 0.4, 0.02], transform=ax.transAxes)      
    #     plt.colorbar(sc, cax=cbaxes,orientation='horizontal')        
    #     cbaxes.set_title('Vertical displacement (m)', fontweight='normal',fontsize='small')
    #     plt.xlim(xlim[0],xlim[1])
    #     plt.ylim(ylim[0],ylim[1])

    # ## bivariate colorbar
    #     cax = fig.add_axes([0.8, 0.02, 0.1, 0.3])
    #     xx, yy = np.mgrid[0:slipmax+5:100j,0:uncmax:30j]
    #     C_map = scalarMap.to_rgba(xx)
    #     cax.imshow(C_map)
    #     yy_plot = np.array(255*(yy-yy.min())/(yy.max()-yy.min()), dtype=np.int)
    #     C_map2  = cptsig2(yy_plot)
    #     cax.imshow(C_map2)
    #     cax.set_ylim((-0.,95)  )   
    #     cax.set_xlim((-0.,29)  )  
    #     cax.locator_params(axis='x', nbins=3)
    #     cax.locator_params(axis='y', nbins=4)
    #     x_label_list = ['0',"{:1.0f}".format(uncmax/3.),"{:1.0f}".format(2*uncmax/3.)]
    #     y_label_list = ['0',"{:1.0f}".format(slipmax/4.),"{:1.0f}".format(2*slipmax/4.),"{:1.0f}".format(4*slipmax/4.)]
    #     cax.set_xticklabels(x_label_list)
    #     cax.set_yticklabels(y_label_list)
    #     cax.yaxis.set_label_position("right")
    #     cax.yaxis.tick_right()
    #     cax.set_ylabel('Slip (m)')
    #     cax.set_xlabel('σ (m)')

        plt.text(xlim[0]+60., ylim[1]-20., '2 m', color='dimgrey',weight='light',fontsize=7) 
        cbaxes = ax.inset_axes([0.1, 0.05, 0.3, 0.02], transform=ax.transAxes)      
        plt.colorbar(sc, cax=cbaxes,orientation='horizontal')        
        cbaxes.set_title('Vertical disp. (m)', fontweight='normal',fontsize='small')
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])

    ## bivariate colorbar
        cax = fig.add_axes([0.55, 0.83, 0.3, 0.08])
        xx, yy = np.mgrid[0:uncmax:30j,0:slipmax+5:100j]
        C_map = scalarMap.to_rgba(yy)
        cax.imshow(C_map)
        xx_plot = np.array(255*(xx-xx.min())/(xx.max()-xx.min()), dtype=np.int)
        C_map2  = cptsig2(xx_plot)
        cax.imshow(C_map2)
        cax.set_xlim((-0.,95)  )   
        cax.set_ylim((-0.,29)  )  
        if uncmax < 2:
            cax.locator_params(axis='y', nbins=2)
            cax.yticks = [0., uncmax/2]
            y_label_list = ['0',"{:1.1f}".format(uncmax/2.)]            
        else:
            cax.locator_params(axis='y', nbins=3)
            y_label_list = ['0',"{:1.0f}".format(uncmax/3.),"{:1.0f}".format(2*uncmax/3.)]
        cax.locator_params(axis='x', nbins=4)
        x_label_list = ['0',"{:1.0f}".format(slipmax/4.),"{:1.0f}".format(2*slipmax/4.),"{:1.0f}".format(4*slipmax/4.)]
        cax.set_xticklabels(x_label_list)
        cax.set_yticklabels(y_label_list)
        cax.yaxis.set_label_position("right")
        cax.yaxis.tick_right()
        cax.set_title('Slip (m)',fontsize='small')
        cax.set_ylabel('σ (m)')
        
        
        plt.savefig(savedir+name+'.pdf', format='pdf',bbox_inches="tight")
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
def plotGPSwSlipBivTriCont(name,fault, sigma, data, synths,slip=None, valmax=None, sigmamax=None, slipdir=None, savedir='./',epicenter=None, xystrides=[100, 100], vertbound=[-1.,1.], shoreline=120.,ylim=[-210.,210.],xlim=[-10.,500.]):
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
            uncmax = 0.6*np.amax(sgm)
        else:
            uncmax = sigmamax
        
        ## colorbars
        cptslip = colors.LinearSegmentedColormap.from_list('cptslip',colorsco_above2_rgba, N=256)
        cNorm = MidpointNormalize(vmin=0., vcenter=slipmax, vmax=slipmax+5.)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cptslip)
        scalarMap.set_array([])
#        cmap = cptslip
#        mycmap = cmap(np.arange(cptslip.N))
#        mycmap[:,-1] = np.hstack((np.zeros((cmap.N//4,)),np.linspace(0.,1.,cmap.N-cmap.N//4)))
#        cptslip2 = colors.ListedColormap(mycmap)
        
        cptsig = colors.LinearSegmentedColormap.from_list('cptsig',sigma_grey_transp_rgba, N=256)
        cmap = cptsig
        mycmap = cmap(np.arange(cptsig.N))
#        alph1 = np.hstack((np.zeros((cmap.N//4,)),np.linspace(0.,1.,cmap.N-cmap.N//2)))
#        mycmap[:,-1] = np.hstack((alph1, np.ones((cmap.N//4,)) ))
        mycmap[:,-1] = np.hstack((np.zeros((cmap.N//4,)),np.linspace(0.,1.,cmap.N-cmap.N//4)))
        cptsig2 = colors.ListedColormap(mycmap)
        cNorm2  = colors.Normalize(vmin=0., vmax=1)
        scalarMap2 = cmx.ScalarMappable(norm=cNorm2, cmap=cptsig)
        scalarMap2.set_array([])

        fault.tent = np.array(fault.getcenters())
        
        x0 = np.amin(fault.tent[:,0])
        ind = np.argmin(fault.tent[:,0], axis=None)
        y0 = fault.tent[ind,1]
        x = []
        y = []
        for i in range(np.shape(fault.tent)[0]):
            xx = fault.tent[i,0]
            yy = fault.tent[i,1]
            d = y
            x.append(xx)
            y.append(yy)
        x = np.vstack(x)
        y = np.vstack(y)
        
        fault.patch = np.array(fault.patch)
        x0 = np.amin(fault.patch[:,:,0])
    #    ind = np.unravel_index(np.argmin(fault.patch[:,:,0], axis=None), fault.patch[:,:,0].shape)
    #    y0 = fault.patch[ind[0],ind[1],1]
        y0 = np.amin(fault.patch[:,:,1])
        dis = []
        dep = []
        for i in range(np.shape(fault.patch)[0]):
            x1 = np.sqrt((fault.patch[i,:,0]-x0)**2 + (fault.patch[i,:,2])**2)
            y1 = fault.patch[i,:,1]
            dis.append(x1)
            dep.append(y1)
        
        fig, ax = plt.subplots(1,figsize=(3.4,6))
        ax.set_xlabel('Distance from trench (km)')
        ax.set_ylabel('Distance along strike (km)')
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=5)
        
        #background
#        ax.add_patch(patches.Rectangle((-100, -300), 220, 600, color='#c4dade'))
#        ax.pcolor(X,Y,topo, cmap=topocmap)                                       
#        ax.add_patch(patches.Rectangle((shoreline, -300), 850, 600, color='#f1f3f4'))             
        
        ## Plot slip
        X = fault.tent[:,0]
        Y = fault.tent[:,1]
        Z = fault.tent[:,2]

        Slip = slp
        Sigma = sgm
        D = Y
        # diss2 = np.linspace(np.nanmin(D), np.nanmax(D), 300)
        # z2 = -np.linspace(np.nanmin(Z2)-5., np.nanmax(Z2)+5.,300)           
        Z2 = np.sqrt((X-x0)**2 + (Z)**2)
        Z2 = np.hstack((Z2, np.array([0,0,0]
                            +np.linspace(0,np.amax(np.abs(Z2)),10).tolist()
                            +np.linspace(0,np.amax(np.abs(Z2)),10).tolist()
                            +[np.amax(np.abs(Z2))+50,np.amax(np.abs(Z2))+50,np.amax(np.abs(Z2))+50])))
        D = np.hstack((D, np.array([-260,0,260]
                                   +[-260]*10
                                   +[260]*10
                                   +[-260,0,260])))
        Slip = np.hstack((Slip, np.array([0,0,0]
                                         +[0]*20
                                         +[0,0,0])))
        Sigma = np.hstack((Sigma, np.array([0,0,0]
                                         +[0]*20
                                         +[0,0,0])))
        diss2 = np.linspace(-250, 250, 300)
        z2 = -np.linspace(0, 230,300) 
        diss2, z2 = np.meshgrid(diss2,z2)
        slip22 = sciint.griddata((D,-Z2),Slip,(diss2,z2),method='cubic')
        sigma22 = sciint.griddata((D,-Z2),Sigma,(diss2,z2),method='cubic')
        import scipy.ndimage as ndimage
        slip2 = ndimage.median_filter(slip22,size=(1,1))
        sigma2 = ndimage.median_filter(sigma22,size=(1,1))
           
        cslip = colors.LinearSegmentedColormap.from_list('cslip',scalarMap.to_rgba(range(0,int(valmax)+5)), 256)
        ax.pcolormesh(-z2, diss2, slip2, cmap=cslip, vmin=0, vmax=slipmax+5,rasterized = True)
        
        ## Plot triangles
        rects = []
        for i in range(len(dis)):
            vertex = np.vstack((np.array(dis[i]),dep[i]))
            rect = patches.Polygon( vertex.T )
            rects.append(rect)
        p = PatchCollection(rects, facecolors='None', edgecolor = 'white', lw=0.2)
        ax.add_collection(p)
        
        ax.pcolormesh(-z2, diss2,sigma2, cmap=cptsig2, vmin=0, vmax=uncmax, edgecolors=None,rasterized = True)
#        sig = ax.pcolor(-z2, diss2,sigma2, cmap=cptsig, vmin=0, vmax=uncmax, edgecolors=None,rasterized = True)
#        for i,j in zip(sig.get_facecolors(),sigma2.flatten()/np.amax(sigma2)):
#            i[3] = j # Set the alpha value of the RGBA tuple using m2                               
                                       
        # trench and shoreline                               
        ax.plot([shoreline,shoreline],[-300.,300], lw=1, c='#cdcdcd')
#        ax.plot([0,0],[-300.,300], lw=2, c='#003547')
                 
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
                    color='#1a5665' )
        from matplotlib.colors import ListedColormap
        pal=ListedColormap(sns.diverging_palette(80, 240, s=120, l=30, n=35))
        sc = plt.scatter(data.x, data.y, c=data.vel_enu[:,2], s=80, cmap=cmo.delta_r, vmin=vertbound[0], vmax=vertbound[1])
        sc = plt.scatter(data.x, data.y, c=synths[data.name][:,2], s=25, lw=0.2, edgecolors='white', cmap=cmo.delta_r, vmin=vertbound[0], vmax=vertbound[1])
        
        plt.quiver([xlim[0]+100.], [ylim[1]-10.],
                  [-2.], 
                   [0.],
#                   units = 'width', width = 0.2,
    #                  scale = None, 
    #                  scale_units='inches',
#                   scale = 2.5, 
                   scale_units = 'xy',
                   scale=3.e-2,
                   width = 0.003,
                   color='dimgrey')
        plt.text(xlim[0]+60., ylim[1]-20., '2 m', color='dimgrey',weight='light',fontsize=7) 
        cbaxes = ax.inset_axes([0.1, 0.05, 0.3, 0.02], transform=ax.transAxes)      
        plt.colorbar(sc, cax=cbaxes,orientation='horizontal')        
        cbaxes.set_title('Vertical disp. (m)', fontweight='normal',fontsize='small')
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])

    ## bivariate colorbar
        cax = fig.add_axes([0.55, 0.83, 0.3, 0.08])
        xx, yy = np.mgrid[0:uncmax:30j,0:slipmax+5:100j]
        C_map = scalarMap.to_rgba(yy)
        cax.imshow(C_map)
        xx_plot = np.array(255*(xx-xx.min())/(xx.max()-xx.min()), dtype=np.int)
        C_map2  = cptsig2(xx_plot)
        cax.imshow(C_map2)
        cax.set_xlim((-0.,95)  )   
        cax.set_ylim((-0.,29)  )  
        if uncmax < 2:
            cax.locator_params(axis='y', nbins=2)
            cax.yticks = [0., uncmax/2]
            y_label_list = ['0',"{:1.1f}".format(uncmax/2.)]            
        else:
            cax.locator_params(axis='y', nbins=3)
            y_label_list = ['0',"{:1.0f}".format(uncmax/3.),"{:1.0f}".format(2*uncmax/3.)]
        cax.locator_params(axis='x', nbins=4)
        x_label_list = ['0',"{:1.0f}".format(slipmax/4.),"{:1.0f}".format(2*slipmax/4.),"{:1.0f}".format(4*slipmax/4.)]
        cax.set_xticklabels(x_label_list)
        cax.set_yticklabels(y_label_list)
        cax.yaxis.set_label_position("right")
        cax.yaxis.tick_right()
        cax.set_title('Slip (m)',fontsize='small')
        cax.set_ylabel('σ (m)')

        plt.savefig(savedir+name+'.pdf', format='pdf',bbox_inches="tight")
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def plotGPSwEntTriCont(name,fault, data, synths,slip=None, valmax=None, slipdir=None, savedir='./',epicenter=None, xystrides=[100, 100], vertbound=[-1.,1.], shoreline=120.,ylim=[-210.,210.],xlim=[-10.,500.],outline=None,target=None):
    '''
     
    '''
    try:        
        if slip in ('strikeslip','ss','strike-slip'):
            slp = np.abs(fault.slip[:,0].copy())
        elif slip in ('dipslip','ds','dip-slip'):
            slp = np.abs(fault.slip[:,1].copy())
        elif slip in ('total','tot'):
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
        else:
            slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)        
        if valmax is None:
            slipmax = np.amax(slp)
        else:
            slipmax = valmax
        
        fault.tent = np.array(fault.getcenters())
        
        x0 = np.amin(fault.tent[:,0])
        ind = np.argmin(fault.tent[:,0], axis=None)
        y0 = fault.tent[ind,1]
        x = []
        y = []
        for i in range(np.shape(fault.tent)[0]):
            xx = fault.tent[i,0]
            yy = fault.tent[i,1]
            d = y
            x.append(xx)
            y.append(yy)
        x = np.vstack(x)
        y = np.vstack(y)
        
        fault.patch = np.array(fault.patch)
        x0 = np.amin(fault.patch[:,:,0])
    #    ind = np.unravel_index(np.argmin(fault.patch[:,:,0], axis=None), fault.patch[:,:,0].shape)
    #    y0 = fault.patch[ind[0],ind[1],1]
        y0 = np.amin(fault.patch[:,:,1])
        dis = []
        dep = []
        for i in range(np.shape(fault.patch)[0]):
            x1 = np.sqrt((fault.patch[i,:,0]-x0)**2 + (fault.patch[i,:,2])**2)
            y1 = fault.patch[i,:,1]
            dis.append(x1)
            dep.append(y1)
        
        fig, ax = plt.subplots(1,figsize=(3.4,6))
        ax.set_xlabel('Distance from trench (km)')
        ax.set_ylabel('Distance along strike (km)')
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=5)
        
        ## Plot slip
        X = fault.tent[:,0]
        Y = fault.tent[:,1]
        Z = fault.tent[:,2]
        
        Slip = slp
        D = Y
        # diss2 = np.linspace(np.nanmin(D), np.nanmax(D), 300)
        # z2 = -np.linspace(np.nanmin(Z2)-5., np.nanmax(Z2)+5.,300)           
        Z2 = np.sqrt((X-x0)**2 + (Z)**2)
        Z2 = np.hstack((Z2, np.array([0,0,0]
                            +np.linspace(0,np.amax(np.abs(Z2)),10).tolist()
                            +np.linspace(0,np.amax(np.abs(Z2)),10).tolist()
                            +[np.amax(np.abs(Z2))+50,np.amax(np.abs(Z2))+50,np.amax(np.abs(Z2))+50])))
        D = np.hstack((D, np.array([-260,0,260]
                                   +[-260]*10
                                   +[260]*10
                                   +[-260,0,260])))
        Slip = np.hstack((Slip, np.array([0,0,0]
                                         +[0]*20
                                         +[0,0,0])))
        diss2 = np.linspace(-250, 250, 300)
        z2 = -np.linspace(0, 230, 300) 
        diss2, z2 = np.meshgrid(diss2,z2)
        slip22 = sciint.griddata((D,-Z2),Slip,(diss2,z2),method='cubic')
        import scipy.ndimage as ndimage
        slip2 = ndimage.median_filter(slip22,size=(1,1))
         
        import cmasher as cmr
        slp = ax.pcolormesh(-z2, diss2, slip2, cmap=cmr.get_sub_cmap('bone_r', 0., 0.7), vmin=0, vmax=1,rasterized = True)
        
        ## Plot triangles
        rects = []
        for i in range(len(dis)):
            vertex = np.vstack((np.array(dis[i]),dep[i]))
            rect = patches.Polygon( vertex.T )
            rect.set_color('None')
            rect.set_edgecolor('white')
            rect.set_linewidth(0.3)
            rects.append(rect)
                
        if outline is not None:   
            k = 1
            for i in outline:
                vertex = np.vstack((np.array(dis[i]),dep[i]))
                rect = patches.Polygon( vertex.T )
                rect.set_color('None')
                rect.set_edgecolor('#808080ff')
                rect.set_linewidth(1.9)
                rects.append(rect)
                plt.text(dis[i][0]+3,dep[i][0]+1,str(k),fontsize='small',fontweight='bold')
                k+=1
                
        if target is not None:   
            idx = np.array(np.where(target!=0)[0])
            for i in idx:
                vertex = np.vstack((np.array(dis[i]),dep[i]))
                rect = patches.Polygon( vertex.T )
                rect.set_color('None')
                rect.set_edgecolor('#c822169b')
                rect.set_linewidth(0.7)
                rects.append(rect)    
                
        p = PatchCollection(rects, match_original=True)
        ax.add_collection(p)
                                    
        # trench and shoreline                               
        ax.plot([shoreline,shoreline],[-300.,300], lw=1, c='#cdcdcd')
#        ax.plot([0,0],[-300.,300], lw=2, c='#003547')
                 
        # surface defo
        plt.quiver(data.x, data.y, 
                    synths[data.name][:,0], 
                    synths[data.name][:,1],
#                   units = 'width', width = 0.2,
    #                  scale = None, 
    #                  scale_units='inches',
#                   scale = 2.5, 
                    scale_units = 'xy',
                    scale=6.e-3,
                    width = 0.003,
                    color='#1a5665' )
        from matplotlib.colors import ListedColormap
        pal=ListedColormap(sns.diverging_palette(80, 240, s=120, l=30, n=35))
        sc = plt.scatter(data.x, data.y, c=synths[data.name][:,2], s=25, lw=0.2, edgecolors='white', cmap=cmo.delta_r, vmin=vertbound[0], vmax=vertbound[1])
        
        plt.quiver([xlim[0]+50.], [ylim[1]-10.],
                  [-0.2], 
                   [0.],
#                   units = 'width', width = 0.2,
    #                  scale = None, 
    #                  scale_units='inches',
#                   scale = 2.5, 
                   scale_units = 'xy',
                   scale=6.e-3,
                   width = 0.003,
                   color='dimgrey')
        plt.text(xlim[0]+20., ylim[1]-20., '20 cm', color='dimgrey',weight='light',fontsize=7) 
        cbaxes = ax.inset_axes([0.1, 0.05, 0.3, 0.02], transform=ax.transAxes)      
        plt.colorbar(sc, cax=cbaxes,orientation='horizontal')        
        cbaxes.set_title('Vertical residuals (m)', fontweight='normal',fontsize='small')
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])

        cax = fig.add_axes([0.55, 0.87, 0.3, 0.018])
        colo=plt.colorbar(slp, cax=cax, orientation='horizontal')    
        colo.set_ticks([0,.5,1])
        cax.set_title('KL divergence',fontsize='small')
        

        plt.savefig(savedir+name+'.pdf', format='pdf',bbox_inches="tight")
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
def plotHist(idx, samp1, samp2, resdir, outname, slipmax=12., target=None):
            
    cmap = colors.LinearSegmentedColormap.from_list('cptslip',colorsco_above2_rgba, N=256)
    cNorm = MidpointNormalize(vmin=0., vcenter=slipmax, vmax=slipmax+5.)  
    
    fig, ax = plt.subplots(1,len(idx),figsize=(len(idx)*1.5,.75))
    ave1 = np.mean(samp1,axis=1)
    ave2 = np.mean(samp2,axis=1)
    
    for i in range(len(idx)):
        x = ave1[idx[i]]
        h1=ax[i].hist(samp1[idx[i],:], bins=50, histtype='step',color='#808080ff',lw=0.5, density=True)
        x = ave2[idx[i]]
        h2=ax[i].hist(samp2[idx[i],:], bins=200,histtype='step', fill=True, lw=0.5, color='#990e1b', facecolor=cmap(cNorm(x)),density=True)
        if target is not None:
            ax[i].axvline(target[idx[i]-np.shape(samp1)[0]//2],lw=2, color='#808080ff')
        ax[i].spines["left"].set_visible(False)
        ax[i].set_yticks([])
        if np.amax(h1[0]) >= 8*np.amax(h2[0]):
            ax[i].set_ylim(top=2*np.amax(h2[0]))
        plt.text(-0.05,0.1,str(i+1), transform=ax[i].transAxes, fontweight='bold')

       
    # ax[-1].set_xlabel('Slip (m)')
    plt.savefig(resdir+outname+'_hist.pdf')  
    
    return
    
if __name__ == "__main__":
    
    a=0
#    exportSlideResultsTents('/home/thea/projet/cp3d2_synth/fig/res/', 'sa1', '/home/thea/projet/cp3d2_synth/fig/synth2.png',text=['3D', '3D', 'No'])
    
#    PDFall(78,'optimized',7,4, resdir='/u/moana/user/ragon/code/altar/am_static/results/amstatic_cpfg/')
#    PDFcomp([49,'53_3'],'optimized',7,4,resdir='/u/moana/user/ragon/code/altar/am_static/results/amstatic_224_cpfg/')
    
#    slipres_gmt('/u/moana/user/ragon/code/altar/am_static/results/amat_edks', '037', '031_2')
#    length=28000
#    width=16000
#    nstrike=20
#    ndip=12
#    slipres('/u/moana/user/ragon/code/altar/am_static/results/amat_edks', '028', '037', nstrike, ndip, length, width, savedir='/u/moana/user/ragon/code/altar/am_static/results/fig/comp/',faulttype='classical')

#    plotCp('/u/moana/user/ragon/code/altar/am_static/config/',[])
    
#    resdir = '/u/moana/user/ragon/code/altar/maule/results/'
#    savedir = '/u/moana/user/ragon/code/altar/maule/results/fig/'
#    faultpar = [570000,240000,13,10,180+18.,-74.78718,-37.70672,180+18.,9760.]
#    length = faultpar[0]
#    width = faultpar[1]
#    ns = faultpar[2]
#    nd = faultpar[3]
#    slipmax=(5,24)
#    uncmax=(2,12)
#    step = '058'
#    plotSlipBivariate(step, ns, nd, length, width, resdir, savedir, slipmax, uncmax, slip = 'dip')

#    faultFile = '/u/moana/user/ragon/mycode/python/amatrice/fault_patches_166_45.rectangles'
#    resdir = '/u/moana/user/ragon/code/altar/am_static/results/amat_edks/'
#    savedir='/u/moana/user/ragon/code/altar/am_static/results/fig/'  # for figures outputs for cp_fg
#    faultpar = [28000,16000,20,12,166,13.2508, 42.8575,50,0]
#    length = faultpar[0]
#    width = faultpar[1]
#    ns = faultpar[2]
#    nd = faultpar[3]
#    slipmax=(50,200)
#    uncmax=(2,90)
#    step = '037'
#    plotSlipBivariate(step, ns, nd, length, width, resdir, savedir, slipmax, uncmax, slip = 'dip')

