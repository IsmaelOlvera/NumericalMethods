# -*- coding: utf-8 -*-
## module gaussianElimination.trivialPivotingGE


import sys
from pathlib import Path

if __name__ == "__main__":
    parentPath  = Path().absolute().parent
    sys.path.append( str(parentPath) )

import numpy as np
import copy

from basics import substitution,verification


########## INPUT  ##########
## matA -> Matrix [n*n]
## j    -> Integer  |  j in [0,n-1]
########## OUTPUT ##########
## i    -> Integer  |  matA[i,j] != 0  and  matA[w,j] == 0 with w in [i+1,j-1]  =>  i in [0,n-1]
def findRowToSwap(matA,j):
    
    for i in range(j+1,len(matA)):
        
        if( matA[i,j] != 0.0 ):
            return i
        
    return -1


########## INPUT  ##########
## matA     -> Matrix [n*n]
## vecB     -> Vector [n]
########## OUTPUT ##########
## matA     -> Modified matA     -> Upper Triangular Matrix [n*n]
## vecB     -> Modified vecB     -> Vector [n]
## matLam   -> Modifiers of matA -> Lower Triangular Matrix [n*n]
## permList -> Permutation list of rows swaps in matA
def elimination(matA,vecB):
    
    matA = copy.deepcopy(matA)
    vecB = copy.deepcopy(vecB)
    
    n = len(vecB)
    matLam = np.zeros( (n,n), dtype=float)
    permList = [ k for k in range(n) ]
    
    for j in range(0,n-1):
        
        if( matA[j,j] == 0.0 ):
            
            w = findRowToSwap(matA,j)
            
            if( w != -1 ):
                matA  [ [j,w] ] = matA  [ [w,j] ]
                vecB  [ [j,w] ] = vecB  [ [w,j] ]
                matLam[ [j,w] ] = matLam[ [w,j] ]
                permList[j],permList[w] = permList[w],permList[j]
            else:
                return "Sistema Irresoluble por este método"
        
        matLam[j,j] = 1.
        
        for i in range(j+1,n):
            
            lam         = matA[i,j] / matA[j,j]
            matLam[i,j] = lam
            matA[i,j:n] = matA[i,j:n] - lam*matA[j,j:n]
            vecB[i]     = vecB[i] - lam*vecB[j]
        
    matLam[n-1,n-1] = 1.
    
    return matA,vecB,matLam,permList


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
    
    matA = np.array( [ [1.,-1.,2.,1.], [-1.,1.,-2.,3.], [-1.,3.,1.,1.], [-1.,2.,0.,0.] ] )
    vecB = np.array( [ 9.,7.,12.,3. ] )
    
    vecSol = solve(matA,vecB)
    
    print("Matrix A: \n", matA, "\n")
    print("Vector b: \n", vecB, "\n")
    print("Vector Sol: \n", vecSol, "\n")
    
    print( verification.verifySystemSolution(matA,vecB,vecSol) )
