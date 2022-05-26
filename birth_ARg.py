# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:51:33 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
def birth_ARg(XnZn,AR_bounds,LogLc,xc,zc,rhoc,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR):
    
#     AR0_min = AR_bounds[0,0]
#     AR0_max = AR_bounds[0,1]
#     AR1_min = AR_bounds[1,0]
#     AR1_max = AR_bounds[1,1]
#     AR2_min = AR_bounds[2,0]
#     AR2_max = AR_bounds[2,1]
#     AR3_min = AR_bounds[3,0]
#     AR3_max = AR_bounds[3,1]
    
#     AR_min = AR_bounds[0,len(ARgc)+1]
#     AR_max = AR_bounds[1,len(ARgc)+1]
#     arp = AR_min + np.random.rand() * (AR_max-AR_min)

    if ARgc[0] == 0:
        AR_min = AR_bounds[1,0]
        AR_max = AR_bounds[1,1]
        arp = AR_min + np.random.rand() * (AR_max-AR_min)
 
        ARgp = ARgc.copy()
        ARgp[0] = arp
        
    else:
        AR_min = AR_bounds[len(ARgc)+1, 0]
        AR_max = AR_bounds[len(ARgc)+1, 1]
        arp = AR_min + np.random.rand() * (AR_max-AR_min)
        ARgp = np.append(ARgc,arp).copy() # new ar coeff will be added at the end of the array 
        bk_AR = 1.
    
    
    # Check if AR model is stationary
#     coeff = np.flipud(-ARgp)
#     coeff = np.append(coeff,1)
#     zroots=np.roots(coeff)
#     TF = all(abs(zroots)>1) # True means it is stationary
    
    #if TF == True:
    LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,zc,rhoc,ARgp,ARTc,XnZn)[0]

    MHP = bk_AR * np.exp((LogLp - LogLc)/T)
    if np.random.rand()<=MHP:
        LogLc = LogLp
        ARgc = ARgp.copy() 

    return [LogLc,ARgc]


