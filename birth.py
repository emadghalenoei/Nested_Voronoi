# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:51:33 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
from Identify import Identify
from Id2rho import Id2rho

def birth(XnZn,globals_par,LogLc,xmc,zmc,xc,zc,rhoc,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs):
    
    zn_min = globals_par[4,0]
    xp = np.random.rand()
    zp = zn_min + np.random.rand()*(1-zn_min)
    
    [identity, kcell] = Identify(xmc,zmc,xp,zp)
    rhop = Id2rho(identity,globals_par)

    xp = np.append(xc,xp).astype('float32').copy()
    zp = np.append(zc,zp).astype('float32').copy()
    rhop = np.append(rhoc,rhop).astype('float32').copy()
    
    LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xp,zp,rhop,ARgc,ARTc,XnZn)[0]
    MHP = np.exp((LogLp - LogLc)/T)
    if np.random.rand()<=MHP:
        LogLc = LogLp
        xc = xp.copy() 
        zc = zp.copy() 
        rhoc = rhop.copy() 
        
    return [LogLc,xc,zc,rhoc]
    
    
    


