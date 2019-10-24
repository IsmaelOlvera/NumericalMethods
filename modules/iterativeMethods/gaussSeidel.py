# -*- coding: utf-8 -*-
## module iterativeMethods.gaussSeidel


import sys
from pathlib import Path

if __name__ == "__main__":
    parentPath  = Path().absolute().parent
    sys.path.append( str(parentPath) )

import numpy as np
import copy

from basics import norm,verification


########## INPUT  ##########
## matA      -> Matrix [n*n]
## vecB      -> Vector [n]
## vecX      -> Vector [n]
## threshold -> Integer
########## OUTPUT ##########
## vecSol    -> Vector [n]  |  matA * vecSol ≈ vecB
def solve(matA,vecB,vecX,threshold):
    
    n = len(vecB)
    
    indexesToUse = [ np.arange(n) != k for k in range(n) ]
    
    estimationDiff = vecB - np.dot(matA,vecX)
    
    iterations = 0
    while( norm.euclidean(estimationDiff) > threshold ):
        
        vecXexceptList  = [ vecX[ indexesToUse[k] ] for k in range(n) ]
        matAexceptList  = [ matA[k,indexesToUse[k]] for k in range(n) ]
        dotProductsList = [ np.dot(matAexceptList[k],vecXexceptList[k]) for k in range(n) ]
    
        vecX = np.array( [ (vecB[k]-dotProductsList[k]) / matA[k,k] for k in range(n) ] )
        
        estimationDiff = vecB - np.dot(matA,vecX)
        
        iterations += 1
        
    return vecX,iterations
        


if __name__ == "__main__":
    
    matA = np.array( [ [12.,3.,-5.], [1.,5.,3.], [3.,7.,13.] ] )
    vecB = np.array( [ 1.,28.,76. ] )
    
    vecInitSol  = np.zeros( len(vecB) )
    threshold   = 1 * 10**(-10)
    
    vecFinalSol,iterations = solve(matA,vecB,vecInitSol,threshold)
    
    print("Matrix A: \n", matA, "\n")
    print("Vector b: \n", vecB, "\n")
    print("Initial Solution Vector: \n", vecInitSol, "\n")
    print("Threshold: \n", threshold, "\n")
    print("Iterations: \n", iterations, "\n")
    print("Final Solution Vector: \n", vecFinalSol, "\n")
    
    print( verification.verifySystemSolution(matA,vecB,vecFinalSol) )
