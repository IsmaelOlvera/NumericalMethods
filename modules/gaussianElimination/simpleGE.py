# -*- coding: utf-8 -*-
## module gaussianElimination.simpleGE


import sys
from pathlib import Path

if __name__ == "__main__":
    parentPath  = Path().absolute().parent
    sys.path.append( str(parentPath) )

import numpy as np
import copy

from basics import substitution,verification


########## INPUT  ##########
## matA   -> Matrix [n*n]
## vecB   -> Vector [n]
########## OUTPUT ##########
## matA   -> Modified matA     -> Upper Triangular Matrix [n*n]
## vecB   -> Modified vecB     -> Vector [n]
## matLam -> Modifiers of matA -> Lower Triangular Matrix [n*n]
def elimination(matA,vecB):
    
    matA = copy.deepcopy(matA)
    vecB = copy.deepcopy(vecB)
    
    n = len(vecB)
    matLam = np.zeros( (n,n), dtype=float)
    
    for j in range(0,n-1):
        matLam[j,j] = 1.
        for i in range(j+1,n):
            if( matA[i,j] != 0.0 ):
                lam         = matA[i,j] / matA[j,j]
                matLam[i,j] = lam
                matA[i,j:n] = matA[i,j:n] - lam*matA[j,j:n]
                vecB[i]     = vecB[i] - lam*vecB[j]
    matLam[n-1,n-1] = 1.
    
    # Use to debug
    # print(matA)
    # print(vecB)
    # print(matLam)
    
    return matA,vecB,matLam


########## INPUT  ##########
## matA   -> Matrix [n*n]
## vecB   -> Vector [n]
########## OUTPUT ##########
## vecSol -> Vector [n]  |  matA * vecSol = vecB
def solve(matA,vecB):
    
    results = elimination(matA,vecB)
    vecSol  = substitution.backwardSubstitution(results[0],results[1])
    
    return vecSol

        

if __name__ == "__main__":
    
    matA = np.array( [ [4.,1.,2.,-3.,5.], [-3.,3.,-1.,4.,-2.], [-1.,2.,5.,1.,3.], [5.,4.,3.,-1.,2.], [1.,-2.,3.,-4.,5.] ] )
    vecB = np.array( [ -16.,20.,-4.,-10.,3. ] )
        
    vecSol = solve(matA,vecB)
    
    print("Matrix A: \n", matA, "\n")
    print("Vector b: \n", vecB, "\n")
    print("Vector Sol: \n", vecSol, "\n")
    
    print( verification.verifySystemSolution(matA,vecB,vecSol) )
