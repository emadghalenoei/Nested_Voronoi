# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:45:09 2020

@author: emadg

"""

import numpy as np
import matplotlib.pyplot as plt
from Chain2xz import Chain2xz
from Log_Likelihood import Log_Likelihood

def Imshow_Data(dis_s,dg_obs,dT_obs,XnZn,Kernel_Grv,Kernel_Mag,Chain,fpath,figname,fignum):
    dis = dis_s/1000.
    
   
    
    ind = np.argsort(Chain[:,0])[::-1]
    Chain_maxL = Chain[ind[0]].copy()
    [xm, zm, x, z, rho, ARg, ART]= Chain2xz(Chain_maxL).copy()
    
    rg = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,x,z,rho,ARg,ART,XnZn)[3]
    rT = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,x,z,rho,ARg,ART,XnZn)[4]
    
    dg_pre = dg_obs - rg
    dT_pre = dT_obs - rT
    
    fig, axe = plt.subplots(1, 2)

    axe[0].plot(dis,dg_obs, 'k-.',linewidth=2) #row=0, col=0
    axe[0].plot(dis,dg_pre, 'r--',linewidth=2) #row=0, col=0
    axe[0].set(xlabel='X Profile (km)', ylabel='Gravity (mGal)')
    plt.show()
    axe[1].plot(dis,dT_obs, 'k-.',linewidth=2) #row=0, col=0
    axe[1].plot(dis,dT_pre, 'r--',linewidth=2) #row=0, col=0
    axe[1].set(xlabel='X Profile (km)', ylabel='Magnetic (nT)')
    axe[1].yaxis.set_label_position("right")
    axe[1].yaxis.tick_right()
    plt.show()
    
          
#     ax[0, 0].xlabel("X Profile (km)")
#     ax[0, 0].ylabel("Gravity (mGal)")

    
    
    # plt.show()
    # plt.pause(0.00001)
    # plt.draw()
    #fig.savefig(fpath+'/'+figname+str(fignum)+'.png')
    fig.savefig(fpath+'/'+figname+str(fignum)+'.pdf')
    plt.close(fig)    # close the figure window