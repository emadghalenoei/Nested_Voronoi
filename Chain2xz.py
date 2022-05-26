# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:25:37 2020

@author: emadg
"""

def Chain2xz(Chain):
    Nnode = int(Chain[1])
    NARg = int(Chain[2])
    NART = int(Chain[3])
    ARg = Chain[4:4+NARg].copy()
    ART = Chain[4+NARg:4+NARg+NART].copy()
    xm = Chain[4+NARg+NART:7+NARg+NART].copy()
    zm = Chain[7+NARg+NART:10+NARg+NART].copy()
    x = Chain[10+NARg+NART:10+NARg+NART+Nnode].copy()
    z = Chain[10+NARg+NART+Nnode:10+NARg+NART+2*Nnode].copy()
    rho = Chain[10+NARg+NART+2*Nnode:10+NARg+NART+3*Nnode].copy()
    
    return [xm, zm, x, z, rho, ARg, ART]
