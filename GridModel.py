# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:45:33 2020

@author: emadg
"""
import numpy as np
import faiss 

def GridModel(x,z,rho,XnZn,CX,CZ):

    TrainPoints = np.column_stack((x,z)).copy()
    index = faiss.IndexFlatL2(2)
    index.add(TrainPoints)
    D, I = index.search(XnZn, 1)     # actual search    
    
    DensityModel = rho[I[:,0]].copy() 
    DensityModel = DensityModel.reshape((CX,CZ),order="F").copy() 
    return DensityModel