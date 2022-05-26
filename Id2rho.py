

import numpy as np
import faiss 

def Id2rho(identity,globals_par):
    
    rho_salt_min = globals_par[1,0]
    rho_salt_max = globals_par[1,1]
    rho_base_min = globals_par[2,0]
    rho_base_max = globals_par[2,1]
    
    logic_sed = identity==0
    logic_salt = identity==1
    logic_base = identity==2
    Nnode = len(identity)
    r = np.random.rand(Nnode)
    
    rho = logic_salt*(rho_salt_min+r*(rho_salt_max-rho_salt_min))+(logic_base)*(rho_base_min+r*(rho_base_max-rho_base_min))
    rho = rho.astype('float32')

    return rho

