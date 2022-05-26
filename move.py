# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:52:59 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
from cauchy_dist import cauchy_dist
from Identify import Identify
from Id2rho import Id2rho
import sys


def move(XnZn,globals_par,LogLc,xmc,zmc,xc,zc,rhoc,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs):
    
    Nnode=int(np.size(xc))
    rho_salt_min = globals_par[1,0]
    rho_salt_max = globals_par[1,1]
    rho_base_min = globals_par[2,0]
    rho_base_max = globals_par[2,1]
    zn_min = globals_par[4,0]

    dsalt = rho_salt_max-rho_salt_min
    dbase = rho_base_max-rho_base_min
    
#     possible_indx = np.arange(np.size(xmc))
    
    for inode in np.arange(Nnode):
        for ipar in np.arange(1,4): # 1 or 2 or 3

            xp = xc.copy()
            zp = zc.copy()
            rhop = rhoc.copy()
            
            if ipar == 1:
                xp[inode] = cauchy_dist(xc[inode],0.1,0,1,xc[inode])
                if np.isclose(xc[inode] , xp[inode])==1: continue
                
            elif ipar == 2:
                zp[inode] = cauchy_dist(zc[inode],0.1,zn_min,1,zc[inode])
                if np.isclose(zc[inode] , zp[inode])==1: continue
        
            else:
                if rhoc[inode]<0:
                    rhop[inode] = cauchy_dist(rhoc[inode],0.02,rho_salt_min,rho_salt_max,rhoc[inode])
                    if np.isclose(rhoc[inode] , rhop[inode])==1: continue
                elif rhoc[inode]>0:
                    rhop[inode] = cauchy_dist(rhoc[inode],0.02,rho_base_min,rho_base_max,rhoc[inode])
                    if np.isclose(rhoc[inode] , rhop[inode])==1: continue
        
            
         
        
            if ipar<=2:
                
                [identity_c, kcell] = Identify(xmc,zmc,xc[inode],zc[inode])
                [identity_p, kcell] = Identify(xmc,zmc,xp[inode],zp[inode])    
            
                if identity_c != identity_p:
                    
                    rhop[inode] = Id2rho(identity_p,globals_par)
                    rhop = rhop.astype('float32')     
            
            LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xp,zp,rhop,ARgc,ARTc,XnZn)[0]
            
            MHP = np.exp((LogLp - LogLc)/T)
            if np.random.rand()<=MHP:
                LogLc = LogLp
                xc = xp.copy()
                zc = zp.copy()
                rhoc = rhop.copy()
            
    ### Hyper Parameters (Parent Nodes)
    for im in np.arange(2): # 0 for x and 1 for z
        for inode in np.arange(3): # 0 or 1 or 2
            
            rhop = rhoc.copy()
            xmp = xmc.copy()
            zmp = zmc.copy()
        
        if im==0: # x
            xmp[inode] = cauchy_dist(xmc[inode],0.1,0,1,xmc[inode])
            if np.isclose(xmc[inode] , xmp[inode])==1: continue

        elif im==1: # z
            zmp[inode] = cauchy_dist(zmc[inode],0.1,zn_min,1,zmc[inode])
            if np.isclose(zmc[inode] , zmp[inode])==1: continue
                
#         else:

#             indxp = np.delete(possible_indx, inode).copy()
#             inodep = np.random.choice(indxp,1)
#             xmp[inode]  = xmc[inodep].copy()
#             zmp[inode]  = zmc[inodep].copy()
#             xmp[inodep] = xmc[inode].copy()
#             zmp[inodep] = zmc[inode].copy()
            

        [identity_c, kcell] = Identify(xmc,zmc,xc,zc)
        [identity_p, kcell] = Identify(xmp,zmp,xc,zc)
        
        iden_diff = identity_p!=identity_c
        
        if iden_diff.any() == True:
            rhop = Id2rho(identity_p,globals_par)
            rhop = (1-iden_diff)*rhoc + iden_diff*rhop
            rhop = rhop.astype('float32')
        
        LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,zc,rhop,ARgc,ARTc,XnZn)[0]
        
        MHP = np.exp((LogLp - LogLc)/T)
        if np.random.rand()<=MHP:
            LogLc = LogLp
            rhoc = rhop.copy()
            xmc = xmp.copy()
            zmc = zmp.copy()
            
    return [LogLc,xmc,zmc,xc,zc,rhoc]