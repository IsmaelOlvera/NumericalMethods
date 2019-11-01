# -*- coding: utf-8 -*-
## module decompositionLU.cholesky


import sys
from pathlib import Path

if __name__ == "__main__":
    parentPath  = Path().absolute().parent
    sys.path.append( str(parentPath) )

import numpy as np
import copy
import math

from basics import substitution,verification


########## INPUT  ##########
## matA     -> Matrix [n*n]
########## OUTPUT ##########
## matL     -> Matrix [n*n]
## matLtran -> Matrix [n*n]  |  matA  = matL * matLtran
def findLU(matA):
    
    n    = len(matA)
    matL = np.zeros( (n,n), dtype=float)
    
    for j in range(n):
        
        matL[j,j] = math.sqrt( matA[j,j] - sum( [ matL[j,k]**2 for k in range(j) ] ) )
        
        for i in range(j+1,n):
            matL[i,j] = ( matA[i,j] - sum([ matL[i,k]*matL[j,k] for k in range(j) ]) )  /  matL[j,j]
        
    return matL,matL.transpose()


########## INPUT  ##########
## matA -> Matrix [n*n]
## vecB -> Vector [n]
########## OUTPUT ##########
## vecX -> Vector [n]  |  matA * vecX = vecB
##      => matA = matL * matLtran
##      => matL * (matLtran * vecX) = vecB
def solveSystem(matA,vecB):
    
    matL,matLtran = findLU(matA)
    
    vecY = substitution.forwardSubstitution(matL,vecB)
    #print(vecY)
    vecX = substitution.backwardSubstitution(matLtran,vecY)
    #print(vecX)
    
    return vecX



if __name__ == "__main__":
    
    print("Enter any of the following options: ")
    print("\t A. Find LU Decomposition of A, using Cholesky's Method.")
    print("\t B. Solve Ax = b, given A and b, using Cholesky's Method.")
    caseInput = input("Input: ")
    print()
    
    if( caseInput == "A" ):
        matA      = np.array( [ [25.,15.,-5.], [15.,18.,0.], [-5.,0.,11.] ] )
        
        matL,matU = findLU(matA)
        
        print("Matrix A: \n", matA, "\n")
        print("Matrix L: \n", matL, "\n")
        print("Matrix U: \n", matU, "\n")
        
        print( verification.verifyMatrixMultiplication(matA,matL,matU) )
        
        
    elif( caseInput == "B" ):
        matA = np.array( [ [25.,15.,-5.], [15.,18.,0.], [-5.,0.,11.] ] )
        vecB = [ 40.,9.,-4. ]
        
        vecSol = solveSystem(matA,vecB)
        
        print("Matrix A: \n", matA, "\n")
        print("Vector b: \n", vecB, "\n")
        print("Vector Sol: \n", vecSol, "\n")
        
        print( verification.verifySystemSolution(matA,vecB,vecSol) )
