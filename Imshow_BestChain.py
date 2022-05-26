# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:45:09 2020

@author: emadg
"""

import numpy as np
import matplotlib.pyplot as plt
from Chain2xz import Chain2xz
import faiss 
from scipy.spatial import Voronoi, voronoi_plot_2d


def Imshow_BestChain(x1,x2,z1,z2,XnZn,CX,CZ,globals_par,Chain,fpath,figname,fignum):
    
#     x1 = dis_min/1000
#     x2 = dis_max/1000
#     z1 = z_min/1000
#     z2 = z_max/1000

    ind = np.argsort(Chain[:,0])[::-1]
    Chain_maxL = Chain[ind[0]].copy()
    [xm, zm, xc, zc, rhoc, ARgc, ARTc]= Chain2xz(Chain_maxL).copy()
    
    TrainPoints = np.column_stack((xc,zc)).copy()
    index = faiss.IndexFlatL2(2)
    index.add(TrainPoints)
    D, I = index.search(XnZn, 1)     # actual search    
    #DensityModel = rho[I].squeeze()
    DensityModel = rhoc[I[:,0]].copy()
    DensityModel = DensityModel.reshape((CZ,CX),order="F").copy() 

    rho_salt_min = globals_par[1,0]
    rho_base_max = globals_par[2,1]


    fig, ax = plt.subplots(gridspec_kw={'wspace': 0, 'hspace': 0},figsize=(10, 10))
    plt.rc('font', weight='bold')
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)


    pos00 = ax.get_position() # get the original position
    pos00.x0 += 0.2  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
    # pos00.top -= 0.2  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
    plt.subplots_adjust(left=0.3, right=0.9, bottom=0.2, top=0.7)
    # ax.set_position(pos00) # set a new position

    xbins = 5
    zbins = 5
    Xticklabels = np.around(np.linspace(x1,x2,xbins), 2)
    Zticklabels = np.around(np.linspace(z1,z2,zbins), 2)


    im00 = ax.imshow(DensityModel,interpolation='none',
           vmin=rho_salt_min, vmax=rho_base_max, extent=(0,1,1,0), aspect='auto', cmap='seismic')

    # plt.locator_params(nbins=4)
    plt.locator_params(axis='y', nbins=zbins-1)
    plt.locator_params(axis='x', nbins=xbins-1)


#     ax.set_xticklabels(Xticklabels)
#     ax.set_yticklabels(Zticklabels)
    

    plt.xlabel("Distance (km)",fontweight="bold", fontsize = 20)
    plt.ylabel("Depth (km)",fontweight="bold", fontsize = 20)

    cbar_pos_density = fig.add_axes([0.1, 0.2, 0.03, 0.4]) 
    cbar_density = plt.colorbar(im00, ax=ax ,shrink=0.3, cax = cbar_pos_density,
                        orientation='vertical', ticklocation = 'left')
    cbar_density.ax.tick_params(labelsize=15)
    cbar_density.set_label(label = 'density contrast ($\mathregular{g/cm^{3}}$)', weight='bold')

    ax.plot(xc,zc,'ko')

    TrainPoints = np.column_stack((xc,zc)).copy()
    #TrainPoints = np.vstack((TrainPoints, [0.2, 1.3]))

    vor = Voronoi(TrainPoints)
    voronoi_plot_2d(vor,  show_vertices=False, line_colors='black',
                    line_width=3, line_alpha=1.0, point_size=0, ax=ax)
    
    ax.plot(xm,zm,'mx')
    ParentNodes = np.column_stack((xm,zm)).copy()
    vorm = Voronoi(ParentNodes)
    voronoi_plot_2d(vorm,  show_vertices=False, line_colors='m',
                    line_width=3, line_alpha=1.0, point_size=0, ax=ax)

    # for region in vor.regions:
    #     if not -1 in region:
    #         polygon = [vor.vertices[i] for i in region]
    #         plt.fill(*zip(*polygon), facecolor='none', edgecolor='black')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.invert_yaxis()
    
    
    

    plt.show()
    fig.savefig(fpath+'/'+figname+str(fignum)+'.pdf')
    plt.close(fig)    # close the figure window