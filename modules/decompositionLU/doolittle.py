# -*- coding: utf-8 -*-
## module decompositionLU.doolittle


import sys
from pathlib import Path

if __name__ == "__main__":
    parentPath   = Path().absolute().parent
    sys.path.append( str(parentPath) )

import numpy as np

from basics              import substitution,verification
from gaussianElimination import simpleGE


########## INPUT  ##########
## matA -> Matrix [n*n]
########## OUTPUT ##########
## matL -> Matrix [n*n]
## matU -> Matrix [n*n]  |  matA = matL * matU
def findLU(matA):
    
    n    = len(matA)
    vecB = np.zeros(n,dtype=float)
    
    matU,vecBmod,matL = simpleGE.elimination(matA,vecB)
    
    return matL,matU


########## INPUT  ##########
## matA -> Matrix [n*n]
## vecB -> Vector [n]
########## OUTPUT ##########
## vecX -> Vector [n]  |  matA * vecX = vecB
##      => matA = matL * matU
##      => matL * (matU * vecX) = vecB
def solveSystem(matA,vecB):
    
    matL,matU = findLU(matA)
    vecY = substitution.forwardSubstitution(matL,vecB)
    # print(vecY)
    vecX = substitution.backwardSubstitution(matU,vecY)
    # print(vecX)
    return vecX



if __name__ == "__main__":
    
    print("Enter any of the following options: ")
    print("\t A. Find LU Decomposition of A, using Doolittle's Method.")
    print("\t B. Solve Ax = b, given A and b, using Doolittle's Method.")
    caseInput = input("Input: ")
    print()
    
    
    if( caseInput == "A" ):
        matA      = np.array( [ [4.,1.,2.,-3.,5.], [-3.,3.,-1.,4.,-2.], [-1.,2.,5.,1.,3.], [5.,4.,3.,-1.,2.], [1.,-2.,3.,-4.,5.] ] )
        matL,matU = findLU(matA)
        
        print("Matrix A: \n", matA, "\n")
        print("Matrix L: \n", matL, "\n")
        print("Matrix U: \n", matU, "\n")
                
        print( verification.verifyMatrixMultiplication(matA,matL,matU) )
        
        
    elif( caseInput == "B" ):
        matA = np.array( [ [4.,1.,2.,-3.,5.], [-3.,3.,-1.,4.,-2.], [-1.,2.,5.,1.,3.], [5.,4.,3.,-1.,2.], [1.,-2.,3.,-4.,5.] ] )
        vecB = np.array( [ -16.,20.,-4.,-10.,3. ] )
    
        vecSol = solveSystem(matA,vecB)
        
        print("Matrix A: \n", matA, "\n")
        print("Vector b: \n", vecB, "\n")
        print("Vector Sol: \n", vecSol, "\n")
        
        print( verification.verifySystemSolution(matA,vecB,vecSol) )
