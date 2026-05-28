import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colormaps
import matplotlib.cm as cmx
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
import seaborn as sns
import cmocean.cm as cmo
from PIL import ImageColor
import colormaps as cmap
from thea.CurvedText import CurvedText
import sys
import os
import cartopy as ccrs
import matplotlib.ticker as mticker
import matplotlib as mpl

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

def plotSlip2D(fault, save_path="./slip_2d.pdf", slip=None, valmax=None, slipdir=None, sigma=False, legend='Coseismic slip (cm)',colorbar=colorsco_above_rgba,savename='slip2d',epicenter=None,index=False):
    '''
    '''
    labelsize = 13.
    ticksize=11.
    mpl.rcParams['xtick.labelsize'] = ticksize
    mpl.rcParams['ytick.labelsize'] = ticksize
    mpl.rcParams['axes.labelsize'] = labelsize
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
    
    fig, ax = plt.subplots(1,figsize=(10,4))
    ax.set_xlabel('Distance along strike (km)')
    ax.set_ylabel('Depth (km)')
    
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
    # ax2 = ax.secondary_yaxis(-0.1, functions=(forward, inverse))
    # ax2.set_ylabel('Depth (km)')
    # ax2.set_yticks([0,-10,-20,-40,-60,-80])

    ax.invert_xaxis()
    x_max = np.amax(dis)
    ax.set_xticks([x_max - 0, x_max - 20, x_max - 40])  # or however many ticks you want
    ax.set_xticklabels([0,20,40])  # set the labels to be the same as the tick values])
        

    ## bivariate colorbar
    cax = fig.add_axes([0.93, 0.23, 0.05, 0.5])
    ax.set_position([0.02, 0.1, 0.85, 0.9])
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
    
    
    plt.savefig(save_path, format='pdf',bbox_inches="tight")
    plt.show()              
     
    return

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    


def plotSlipBivariate(fault,sigma,save_path="./2d_slip_bivariate.pdf",slip=None, valmax=None, sigmamax=None, slipdir=None, savedir='./',epicenter=None):
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
        cmap = colors.ListedColormap(bivcolors_rgba)
        # colormaps.register(cmap=cmap, name='biv')
        # cmap.create_cmap(bivcolors, 'biv')
        # cmape = plt.cm.get_cmap('biv',15)
        cNorm  = colors.Normalize(0,15)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        scalarMap.set_array([])
                
        if slip in ('strikeslip','ss','strike-slip'):
            slp = fault.slip[:,0].copy()
            sgm = sigma
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
        
        fig, (ax, ax_colbar) = plt.subplots(1,2,figsize=(10,4), gridspec_kw={'width_ratios': [7.5,2.5]}, layout="constrained")
        # fig, ax = plt.subplots(1,figsize=(7,4))
        ax.set_xlabel('\n Distance along strike (km)')
        ax.set_ylabel('\n Depth (km)')
        
        # MAIN PLOTTING SECTION 
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

        # END MAIN PLOTTING SECTION
        
    #     if slipdir is not None:
    #         rake = np.loadtxt(slipdir, comments='>')
    #         xc = rake[:,0]
    #         yc = rake[:,1]
    #         cent = []
    #         for i in range(len(xc)):
    #             d = np.sqrt((xc-x0)**2 + (yc-y0)**2)
    #             cent.append(d)
    #         cent = np.array(cent)
    #         ax.quiver(cent, -rake[:,4],
    #                   0.5*slp[:], 0.5*slp[:],
    #                   units = 'width',
    #                   angles = [-rake[:,7]],
    #                   width = 0.002,
    # #                  scale = None, 
    # #                  scale_units='inches',
    #                   scale = 10**2.1, 
    #                   scale_units = 'x', 
    #                   color='dimgrey')
        
    #     if epicenter is not None:
    #         xe, ye = fault.ll2xy(epicenter[0], epicenter[1])
    #         de = np.sqrt((xe-x0)**2 + (ye-y0)**2)
    #         plt.scatter(de, epicenter[2], s=250, c='white', edgecolors='dimgrey', marker=(5, 1))
            
        
        # plot colorscale 
        # print(np.array(dep).shape, np.array(dis).shape)
        center = [1.2*np.amax(dis), np.amax(dep)]
        # ax.scatter(center[1, center[0]], s=100, c='white', edgecolors='dimgrey', marker=(5, 1))
        print("center", center)
        L = np.amax(np.abs(dep))/3
        print("L", L)
        wdgs = []

        center = [8.77, 1]
        L = 1.7
        # patch = patches.Wedge(center, L/4, 45, 135, fc="orange", transform=fig.dpi_scale_trans)
        # ax_colbar.add_patch(patch)
        ax_colbar.set_axis_off()
        ax_colbar.set_xticks([])
        ax_colbar.set_yticks([])

        # 1
        wdgs.append(patches.Wedge(center, L/4, -30+90, 30+90, width=None,ec='white',fc=bivcolors_rgba[0],lw=0.1, transform=fig.dpi_scale_trans))
        #2
        wdgs.append(patches.Wedge(center, 2*L/4, 0+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[1],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 2*L/4, -30+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[2],lw=0.1, transform=fig.dpi_scale_trans))
        #3
        wdgs.append(patches.Wedge(center, 3*L/4, -30+90, -15+90, width=L/4,ec='white',fc=bivcolors_rgba[6],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 3*L/4, -15+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[5],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 3*L/4, 0+90, 15+90, width=L/4,ec='white',fc=bivcolors_rgba[4],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 3*L/4, 15+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[3],lw=0.1, transform=fig.dpi_scale_trans))
        #4
        wdgs.append(patches.Wedge(center, 4*L/4, -30+90, -22.5+90, width=L/4,ec='white',fc=bivcolors_rgba[14],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, -22.5+90, -15+90, width=L/4,ec='white',fc=bivcolors_rgba[13],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, -15+90, -7.5+90, width=L/4,ec='white',fc=bivcolors_rgba[12],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, -7.5+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[11],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, 0+90, 7.5+90, width=L/4,ec='white',fc=bivcolors_rgba[10],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, 7.5+90, 15+90, width=L/4,ec='white',fc=bivcolors_rgba[9],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, 15+90, 22.5+90, width=L/4,ec='white',fc=bivcolors_rgba[8],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, 22.5+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[7],lw=0.1, transform=fig.dpi_scale_trans))
        for wdg in wdgs:
            ax_colbar.add_patch(wdg)
        # w = PatchCollection(wdgs, match_original=True)
        # ax_colbar.add_collection(w)
        
        #legend
        coords = []
        for i in [14,12,10,8]:
            coords.append(wdgs[i].get_path().vertices[6])
        for i in [7,3,2,0]:
            coords.append(wdgs[i].get_path().vertices[0])
        print("coords", coords)
        labels = [str(i) for i in np.arange(0,valmax,valmax/4)]+[str(valmax)]+[str(sigmamax//2)]+[str(3*sigmamax//4)]+[str(sigmamax)]     
        rot = [30,15,0,-15,-30,60,60,60]
        vas = ['center']*5+['top']*3
        offset = [[0,0.05*L]]*5+[[0.05*L,0]]*3
        for i in range(len(labels)):
            ax_colbar.text(coords[i][0]+offset[i][0],coords[i][1]+offset[i][1],labels[i],rotation=rot[i],rotation_mode='anchor',ha='center',va=vas[i], transform=fig.dpi_scale_trans, fontsize=8.)
        
        # legend titles
        ax_colbar.text(center[0], center[1]+1.15*L, 'Slip (m)', rotation=0, ha='center', va='center', transform=fig.dpi_scale_trans, fontsize=9.)
        ax_colbar.text(center[0]+0.39*L, center[1]+0.27*L, 'Standard deviation (m)', rotation=60, ha='center', va='center', transform=fig.dpi_scale_trans, fontsize=9.)

        # wdg = patches.Wedge(center, L, -30+90, 30+90, width=None)
        # x = wdg.get_path().vertices[:,0]
        # y = wdg.get_path().vertices[:,1]
        # text = CurvedText(
        #     x = x[::-1][3:-1],
        #     y = y[::-1][3:-1]+0.13*L,
        #     text='Slip amplitude (m)',
        #     va = 'bottom',
        #     fontweight='regular',
        #     axes = ax)
        # text2 = CurvedText(
        #     x = x[::-1][0:2]+0.13*L,
        #     y = y[::-1][0:2],
        #     text='Slip uncertainty (m)',
        #     va = 'top',
        #     fontweight='regular',
        #     axes = ax)
        
        ax.set_xlim(np.amin(dis),np.amax(dis) )
        ax.set_ylim(np.amin(dep),np.amax(dep))
        ax.locator_params(axis='x', nbins=5)
        ax.locator_params(axis='y', nbins=5)
        plt.savefig(save_path, format='pdf',bbox_inches="tight")
        # plt.tight_layout()
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    
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

#        import pdb
#        pdb.set_trace()
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



def animateSlipConvergenceBivariate(fault,slips,sigmas,save_path="./2d_slip_convergence_bivariate.mp4",slip=None, valmax=None, sigmamax=None, slipdir=None, savedir='./',epicenter=None):
    '''
     
    '''
    try:
        # Bivariate colors defined from base to top and from left to right
        bivcolors=[ (208, 208, 208),
                    (232, 232, 232),  (164, 128, 128),
                    (250, 250, 250), (237,204,187), (214, 137, 127), (149, 75, 75),
                    (254, 248, 241), (254, 227, 190), (253, 173, 119), (248, 130, 84), (233, 87, 61), (210, 41, 27), (173, 0, 0), (127, 0, 0)]
        bivcolors_rgba= [(x[0]/255,x[1]/255,x[2]/255) for x in bivcolors]

        cmap = colors.ListedColormap(bivcolors_rgba)
        cNorm  = colors.Normalize(0,15)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        scalarMap.set_array([])
                
        # if slip in ('strikeslip','ss','strike-slip'):
        #     slp = fault.slip[:,0].copy()
        #     sgm = sigma
        # elif slip in ('dipslip','ds','dip-slip'):
        #     slp = fault.slip[:,1].copy()
        #     sgm = sigma[len(sigma)//2:len(sigma)]
        # elif slip in ('total','tot'):
        #     slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
        #     sgm = np.sqrt(sigma[0:len(sigma)//2]**2 + sigma[len(sigma)//2:len(sigma)]**2)
        # else:
        #     slp = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)        
        #     sgm = np.sqrt(sigma[0:len(sigma)//2]**2 + sigma[len(sigma)//2:len(sigma)]**2)
            
        if valmax is None:
            slipmax = np.amax(slips)
        else:
            slipmax = valmax
        if sigmamax is None:
            uncmax = np.amax(sigmas)
        else:
            uncmax = sigmamax
        # colval = ColValSup(slp,sgm,slipmax,uncmax)
        # print("slipmax", slipmax)
        # print("uncmax", uncmax)
        
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
        
        fig, (ax, ax_colbar) = plt.subplots(1,2,figsize=(10,4), gridspec_kw={'width_ratios': [7.5,2.5]}, layout="constrained")
        # fig, ax = plt.subplots(1,figsize=(7,4))
        ax.set_xlabel('\n Distance along strike (km)')
        ax.set_ylabel('\n Depth (km)')
        
        # MAIN PLOTTING SECTION 
        rects = []
        for i in range(len(dis)):
            dis[i] = np.vstack(dis[i])
            dep[i] = np.vstack(dep[i])
            vertex = np.hstack((dis[i],dep[i]))
            rect = patches.Polygon( vertex )
            # if slip is None:
            #     rect.set_color('gray')
            # else:
            #     colorval = scalarMap.to_rgba(colval[i])
            #     rect.set_color(colorval)
            rect.set_edgecolor('white')
            rect.set_linewidth(0.1)
            rects.append(rect)
        p = PatchCollection(rects, match_original=True)
        ax.add_collection(p)

        # END MAIN PLOTTING SECTION

        wdgs = []

        center = [8.77, 1]
        L = 1.7
        ax_colbar.set_axis_off()
        ax_colbar.set_xticks([])
        ax_colbar.set_yticks([])

        # 1
        wdgs.append(patches.Wedge(center, L/4, -30+90, 30+90, width=None,ec='white',fc=bivcolors_rgba[0],lw=0.1, transform=fig.dpi_scale_trans))
        #2
        wdgs.append(patches.Wedge(center, 2*L/4, 0+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[1],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 2*L/4, -30+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[2],lw=0.1, transform=fig.dpi_scale_trans))
        #3
        wdgs.append(patches.Wedge(center, 3*L/4, -30+90, -15+90, width=L/4,ec='white',fc=bivcolors_rgba[6],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 3*L/4, -15+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[5],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 3*L/4, 0+90, 15+90, width=L/4,ec='white',fc=bivcolors_rgba[4],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 3*L/4, 15+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[3],lw=0.1, transform=fig.dpi_scale_trans))
        #4
        wdgs.append(patches.Wedge(center, 4*L/4, -30+90, -22.5+90, width=L/4,ec='white',fc=bivcolors_rgba[14],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, -22.5+90, -15+90, width=L/4,ec='white',fc=bivcolors_rgba[13],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, -15+90, -7.5+90, width=L/4,ec='white',fc=bivcolors_rgba[12],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, -7.5+90, 0+90, width=L/4,ec='white',fc=bivcolors_rgba[11],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, 0+90, 7.5+90, width=L/4,ec='white',fc=bivcolors_rgba[10],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, 7.5+90, 15+90, width=L/4,ec='white',fc=bivcolors_rgba[9],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, 15+90, 22.5+90, width=L/4,ec='white',fc=bivcolors_rgba[8],lw=0.1, transform=fig.dpi_scale_trans))
        wdgs.append(patches.Wedge(center, 4*L/4, 22.5+90, 30+90, width=L/4,ec='white',fc=bivcolors_rgba[7],lw=0.1, transform=fig.dpi_scale_trans))
        for wdg in wdgs:
            ax_colbar.add_patch(wdg)
        # w = PatchCollection(wdgs, match_original=True)
        # ax_colbar.add_collection(w)
        
        #legend
        coords = []
        for i in [14,12,10,8]:
            coords.append(wdgs[i].get_path().vertices[6])
        for i in [7,3,2,0]:
            coords.append(wdgs[i].get_path().vertices[0])
        print("coords", coords)
        labels = [str(i) for i in np.arange(0,valmax,valmax/4)]+[str(valmax)]+[str(sigmamax//2)]+[str(3*sigmamax//4)]+[str(sigmamax)]     
        rot = [30,15,0,-15,-30,60,60,60]
        vas = ['center']*5+['top']*3
        offset = [[0,0.05*L]]*5+[[0.05*L,0]]*3
        for i in range(len(labels)):
            ax_colbar.text(coords[i][0]+offset[i][0],coords[i][1]+offset[i][1],labels[i],rotation=rot[i],rotation_mode='anchor',ha='center',va=vas[i], transform=fig.dpi_scale_trans, fontsize=8.)
        
        # legend titles
        ax_colbar.text(center[0], center[1]+1.15*L, 'Slip (m)', rotation=0, ha='center', va='center', transform=fig.dpi_scale_trans, fontsize=9.)
        ax_colbar.text(center[0]+0.39*L, center[1]+0.27*L, 'Standard deviation (m)', rotation=60, ha='center', va='center', transform=fig.dpi_scale_trans, fontsize=9.)
        
        ax.set_xlim(np.amin(dis),np.amax(dis) )
        ax.set_ylim(np.amin(dep),np.amax(dep))
        ax.locator_params(axis='x', nbins=5)
        ax.locator_params(axis='y', nbins=5)



        def update(frame):
            # Example: replace this with your evolving data
            slp = slips[frame]
            sgm = sigmas[frame]

            colval = ColValSup(slp, sgm, slipmax, uncmax)
            # print("colval", colval)

            colors = [scalarMap.to_rgba(colval[i]) for i in range(len(colval))]
            p.set_facecolor(colors)

            return [p]
        
        ani = FuncAnimation(
            fig,
            update,
            frames=slips.shape[0],        # number of timesteps
            interval=250,      # 0.5 seconds
            blit=False
        )


        # plt.savefig(save_path, format='pdf',bbox_inches="tight")
        ani.save(save_path, writer="ffmpeg", fps=4)
        # plt.tight_layout()
        plt.show()              

    except Exception as err:
        print(type(err))
        print("ERROR :", str(err))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    
    return




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
