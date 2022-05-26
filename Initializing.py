# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:22:13 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
from Identify import Identify
from Id2rho import Id2rho
from Chain2xz import Chain2xz

def Initializing(Chain,XnZn,globals_par,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,Chain_MaxL,loaddesk):
    
    Kmin = int(globals_par[0,0])
    Kmax = int(globals_par[0,1])
    zn_min = globals_par[4,0].astype('float32')
    rho_salt_min = globals_par[1,0]
    rho_salt_max = globals_par[1,1]
    rho_base_min = globals_par[2,0]
    rho_base_max = globals_par[2,1]
    KminAR = int(globals_par[3,0])
    KmaxAR = int(globals_par[3,1])
#     AR_min = globals_par[4,0]
#     AR_max = globals_par[4,1]

    if loaddesk == 0:

        Nnode = Kmin
        xm = np.random.rand(3).astype('float32')
        zm = (zn_min + np.random.rand(3)*(1-zn_min)).astype('float32')
        xc = np.random.rand(Nnode).astype('float32')
        zc = (zn_min + np.random.rand(Nnode)*(1-zn_min)).astype('float32')
        [identity, kcell] = Identify(xm,zm,xc,zc)
        rhoc = Id2rho(identity,globals_par)
        ARgc = np.array([0.])
        ARTc = np.array([0.])
        
     
    else:
        
        [xm, zm, xc, zc, rhoc, ARgc, ARTc]= Chain2xz(Chain_MaxL)
#         [identity, kcell] = Identify(xm,zm,xc,zc)
#         rhoc = Id2rho(identity,globals_par)
        ARgc = np.array([0.])
        ARTc = np.array([0.])
        
        
    LogLc = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,zc,rhoc,ARgc,ARTc,XnZn)[0]
    
    Chain[0] = LogLc.copy()
    Chain[1] = np.size(xc)
    Chain[2] = np.size(ARgc)
    Chain[3] = np.size(ARTc)
    Chain[4:4+np.size(ARgc)] = ARgc.copy()
    Chain[4+np.size(ARgc):4+np.size(ARgc)+np.size(ARTc)] = ARTc.copy()
    Chain[4+np.size(ARgc)+np.size(ARTc):10+np.size(ARgc)+np.size(ARTc)+np.size(xc)*3] = np.concatenate((xm,zm,xc,zc,rhoc)).copy()
    
    return Chain

    
    