# -*- coding: utf-8 -*-
## module gaussianElimination.scaledPartialPivotingGE


import sys
from pathlib import Path

if __name__ == "__main__":
    parentPath  = Path().absolute().parent
    sys.path.append( str(parentPath) )

import numpy as np
import copy

from basics import substitution,verification


########## INPUT  ##########
## matA         -> Matrix [n*n]
## j            -> Integer  |  j in [0,n-1]
########## OUTPUT ##########
## bestRowIndex -> Integer  |  f( matA[bestRowIndex,j] ) > f( matA[w,j] ) for all w, with w in [j,n-1] and f() as relative size function
def findBestRow(matA,j):
    
    n = len(matA)
    
    actualColumnAbsValues = abs(matA[j:n,j])
    maxAbsElementPerRow   = [  max( abs(matA[i]) ) for i in range(j,n)  ]
        
    bestRowIndex = np.argmax(actualColumnAbsValues / maxAbsElementPerRow) + j
                
    return bestRowIndex


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
    matLam   = np.zeros( (n,n), dtype=float)
    permList = [ k for k in range(n) ]
    
    for j in range(0,n-1):
        
        bestRowIndex = findBestRow(matA,j)
        
        if( bestRowIndex != j ):
        
            matA  [ [j,bestRowIndex] ] = matA  [ [bestRowIndex,j] ]
            vecB  [ [j,bestRowIndex] ] = vecB  [ [bestRowIndex,j] ]
            matLam[ [j,bestRowIndex] ] = matLam[ [bestRowIndex,j] ]
            
            permList[j],permList[bestRowIndex] = permList[bestRowIndex],permList[j]
        
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
    
    matA = np.array( [ [3.,-13.,9.,3.], [-6.,4.,1.,-18.], [6.,-2.,2.,4.], [12.,-8.,6.,10.] ] )
    vecB = np.array( [ -19.,-34.,16.,26. ] )
    
    vecSol = solve(matA,vecB)
    
    print("Matrix A: \n", matA, "\n")
    print("Vector b: \n", vecB, "\n")
    print("Vector Solución: \n", vecSol, "\n")
    
    print( verification.verifySystemSolution(matA,vecB,vecSol) )
