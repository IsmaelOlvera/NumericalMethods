# -*- coding: utf-8 -*-
## module decompositionLU.applications


import sys
from pathlib import Path

if __name__ == "__main__":
    parentPath  = Path().absolute().parent
    sys.path.append( str(parentPath) )

import numpy as np

from basics          import substitution,verification
from decompositionLU import doolittle,cholesky


########## INPUT  ##########
## matA    -> Matrix [n*n]
########## OUTPUT ##########
## matAinv -> Matrix [n*n]  |  matA * matAinv = I
def findInverse(matA, method="doolittle"):
    
    n = len(matA)
    I = np.eye(n)
    
    matAinv = np.zeros((n,n),dtype=float)
    
    matL = np.array(0)
    matU = np.array(0)
    
    if( method == "doolittle" ):
        matL,matU = doolittle.findLU(matA)
    elif( method == "cholesky" ):
        matL,matU = cholesky.findLU(matA)
        
    for j in range(n):
        vecY         = substitution.forwardSubstitution(matL,I[:,j])
        matAinv[:,j] = substitution.backwardSubstitution(matU,vecY)
    
    return matAinv



if( __name__ == "__main__" ):
    
    matA    = np.array( [ [1.,-3.,7.], [0.,3.,-2.], [-2.,6.,1.] ] )
    matAinv = findInverse(matA)
    
    print("Matrix A: \n", matA, "\n")
    print("Matrix A ^ {-1}: \n", matAinv, "\n")
    
    print( verification.verifyInverse(matA,matAinv) )
