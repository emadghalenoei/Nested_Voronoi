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


def Imshow_AllChain(x1,x2,z1,z2,XnZn,CX,CZ,globals_par,Chain,fpath,figname,fignum):
    
#     x1 = dis_min/1000
#     x2 = dis_max/1000
#     z1 = z_min/1000
#     z2 = z_max/1000

    Nchain = Chain.shape[0]
    ichain = Nchain-1
    
    fig, axs = plt.subplots(3, 3, sharex='col', sharey='row',gridspec_kw={'hspace': 0, 'wspace': 0},figsize=(10, 10))

    plt.rc('font', weight='bold')
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    
    axs = axs.ravel()
            
    for iplot in np.arange(9):
        

        [xm, zm, xc, zc, rhoc, ARgc, ARTc]= Chain2xz(Chain[ichain,:]).copy()

        TrainPoints = np.column_stack((xc,zc)).copy()
        index = faiss.IndexFlatL2(2)
        index.add(TrainPoints)
        D, I = index.search(XnZn, 1)     # actual search    
        #DensityModel = rho[I].squeeze()
        DensityModel = rhoc[I[:,0]].copy()
        DensityModel = DensityModel.reshape((CZ,CX),order="F").copy() 

        rho_salt_min = globals_par[1,0]
        rho_base_max = globals_par[2,1]


        im00 = axs[iplot].imshow(DensityModel,interpolation='none',
               vmin=rho_salt_min, vmax=rho_base_max, extent=(0,1,1,0), aspect='auto', cmap='seismic')

        # plt.locator_params(nbins=4)
#         plt.locator_params(axis='y', nbins=zbins-1)
#         plt.locator_params(axis='x', nbins=xbins-1)


    #     ax.set_xticklabels(Xticklabels)
    #     ax.set_yticklabels(Zticklabels)


#         plt.xlabel("Distance (km)",fontweight="bold", fontsize = 20)
#         plt.ylabel("Depth (km)",fontweight="bold", fontsize = 20)

#         cbar_pos_density = fig.add_axes([0.1, 0.2, 0.03, 0.4]) 
#         cbar_density = plt.colorbar(im00, ax=ax ,shrink=0.3, cax = cbar_pos_density,
#                             orientation='vertical', ticklocation = 'left')
#         cbar_density.ax.tick_params(labelsize=15)
#         cbar_density.set_label(label = 'density contrast ($\mathregular{g/cm^{3}}$)', weight='bold')

        axs[iplot].plot(xc,zc,'ko')

        TrainPoints = np.column_stack((xc,zc)).copy()
        #TrainPoints = np.vstack((TrainPoints, [0.2, 1.3]))

        vor = Voronoi(TrainPoints)
        voronoi_plot_2d(vor,  show_vertices=False, line_colors='black',
                        line_width=3, line_alpha=1.0, point_size=0, ax=axs[iplot])

        axs[iplot].plot(xm,zm,'mx')
        ParentNodes = np.column_stack((xm,zm)).copy()
        vorm = Voronoi(ParentNodes)
        voronoi_plot_2d(vorm,  show_vertices=False, line_colors='m',
                        line_width=3, line_alpha=1.0, point_size=0, ax=axs[iplot])

        # for region in vor.regions:
        #     if not -1 in region:
        #         polygon = [vor.vertices[i] for i in region]
        #         plt.fill(*zip(*polygon), facecolor='none', edgecolor='black')

        axs[iplot].set_xlim([0, 1])
        axs[iplot].set_ylim([0, 1])

        axs[iplot].invert_yaxis()
        
        axs[iplot].text(0.2,0.8,"{:.2f}".format(Chain[ichain,0]),horizontalalignment='center',transform=axs[iplot].transAxes,fontsize = 12)

        ichain -= 1
        plt.show()

    
    

    plt.show()
    fig.savefig(fpath+'/'+figname+str(fignum)+'.pdf')
    plt.close(fig)    # close the figure window