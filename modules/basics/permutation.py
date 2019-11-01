# -*- coding: utf-8 -*-
## module basics.permutation


import sys
from pathlib import Path

if __name__ == "__main__":
    parentPath  = Path().absolute().parent
    sys.path.append( str(parentPath) )

import numpy as np


########## INPUT  ##########
## permList -> List [n]
########## OUTPUT ##########
## matPerm  -> Binary Matrix [n*n]  |  matPerm[i,j] = 1. if permList[i] = j
def buildPermutationsMatrix(permList):
        
        n = len(permList)
        
        matPerm = np.zeros((n,n), dtype=float)
        
        for k in range(n):
            matPerm[ k,permList[k] ] = 1.
        
        return matPerm
