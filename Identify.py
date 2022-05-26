

import numpy as np
import faiss 

def Identify(xm,zm,xchild,zchild):
    
    #Nnode = xchild.size
    ParentNodes = np.column_stack((xm,zm)).copy()
    Sorted_ParentNodes = ParentNodes[np.argsort(ParentNodes[:, 1])]

    ChildNodes = np.column_stack((xchild,zchild)).copy()
    ChildNodes = ChildNodes.astype('float32')
    index = faiss.IndexFlatL2(2)
#     index.add(ParentNodes)
    index.add(Sorted_ParentNodes)
    D, I = index.search(ChildNodes, 1)     # actual search    
    identity = I[:,0]
    
    ksed = np.sum(I==0)
    ksalt = np.sum(I==1)
    kbase = np.sum(I==2)
    kcell = np.array([ksed,ksalt,kbase])

    return identity, kcell

