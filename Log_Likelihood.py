# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:06:46 2020

@author: emadg
"""

import numpy as np
import faiss 
from scipy.signal import lfilter
# from profile_each_line import profile_each_line
#import math
# @profile_each_line
def Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,x,z,rho,ARg,ART,XnZn):
    TrainPoints = np.column_stack((x,z)).copy()
    index = faiss.IndexFlatL2(2)
    index.add(TrainPoints)
    D, I = index.search(XnZn, 1)     # actual search
    #DensityModel = rho[I].squeeze()
    DensityModel = rho[I[:,0]].copy()   
    rg = dg_obs - (Kernel_Grv @ DensityModel)
    
    SusModel = DensityModel/50.
    SusModel[DensityModel<0.2]=0
    rT = dT_obs - (Kernel_Mag @ SusModel) #(nT)
    
    N = len(rg)
    SqN = np.sqrt(N)
    #sigma_g = np.linalg.norm(rg)/SqN 
    #sigma_T = np.linalg.norm(rT)/SqN
    
    Arg = np.insert(ARg,0,0).copy()
    da_g = lfilter(Arg , 1, rg)
    
    ArT = np.insert(ART,0,0).copy()
    da_T = lfilter(ArT , 1, rT)
    uncor_g = rg-da_g
    uncor_T = rT-da_T

    sigma_rg = np.linalg.norm(rg)/SqN
    sigma_rT = np.linalg.norm(rT)/SqN 
    
    if sigma_rg<0.1: sigma_rg = 0.1
    if sigma_rT<0.1: sigma_rT = 0.1 

    LogL = -N*np.log(sigma_rg*sigma_rT) - (0.5*np.sum((uncor_g/sigma_rg)**2)) - (0.5*np.sum((uncor_T/sigma_rT)**2))
    
    return LogL, DensityModel, SusModel, rg, rT, sigma_rg, sigma_rT, uncor_g, uncor_T