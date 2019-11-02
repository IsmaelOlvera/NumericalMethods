# -*- coding: utf-8 -*-
## module decompositionLU.doolittle


import sys
from pathlib import Path

if __name__ == "__main__":
    parentPath   = Path().absolute().parent
    sys.path.append( str(parentPath) )

import numpy as np
import copy

from basics              import substitution,permutation,verification
from gaussianElimination import simpleGE,scaledPartialPivotingGE


########## INPUT  ##########
## matA     -> Matrix [n*n]
## strategy -> String  |  Pivoting strategy to use, no pivoting as default.
########## OUTPUT ##########
## matL     -> Matrix [n*n]
## matU     -> Matrix [n*n]
## matPerm  -> Matrix [n*n]  |  matPerm * matA = matL *matU ;
##             If no pivoting is used, matPerm = I.
##             Only matPerm is returned when strategy has not "simple" value.
def findLU(matA, strategy="simple"):
    
    n    = len(matA)
    vecB = np.zeros(n,dtype=float)
    
    if( strategy == "simple" ):
        matU,vecBmod,matL = simpleGE.elimination(matA,vecB)
        return matL,matU
    
    elif( strategy == "scaledPartial" ):
        results = scaledPartialPivotingGE.elimination(matA,vecB)
        
        matL,matU,permList = results[2],results[0],results[3]
        
        matPerm = permutation.buildPermutationsMatrix(permList)
    
    return matL,matU,matPerm


########## INPUT  ##########
## matA     -> Matrix [n*n]
## vecB     -> Vector [n]
## strategy -> String  |  Pivoting strategy to use, no pivoting as default.
########## OUTPUT ##########
## vecX -> Vector [n]  |  matA * vecX = vecB
##      => matPerm * matA = matL * matU
##      => matPerm * ( matL * (matU * vecX) ) = matPerm * vecB
##         If no pivoting is used, matPerm = I.
##         Only matPerm is returned when strategy has not "simple" value.
def solveSystem(matA,vecB,strategy="simple"):
    
    n = len(matA)
    matA = copy.deepcopy(matA)
    vecB = copy.deepcopy(vecB)
    
    if( strategy == "simple" ):
        matL,matU = findLU(matA,strategy)
        matPerm   = np.eye(n)
        
    elif( strategy == "scaledPartial" ):
        matL,matU,matPerm = findLU(matA,strategy)
            
    vecY = substitution.forwardSubstitution(matL, np.matmul(matPerm,vecB) )
    # print(vecY)
    vecX = substitution.backwardSubstitution(matU,vecY)
    # print(vecX)
    return vecX



if __name__ == "__main__":
    
    print("Enter any of the following options: ")
    print("\t A. Find LU Decomposition of A, using Doolittle's Method.")
    print("\t B. Solve Ax = b, given A and b, using Doolittle's Method with no pivoting strategy.")
    print("\t C. Solve Ax = b, given A and b, using Doolittle's Method with scaled partial pivoting strategy.")
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
        
        
    elif( caseInput == "C" ):
        matA = np.array( [ [3.,-13.,9.,3.], [-6.,4.,1.,-18.], [6.,-2.,2.,4.], [12.,-8.,6.,10.] ] )
        vecB = np.array( [ -19.,-34.,16.,26. ] )
        
        vecSol = solveSystem(matA,vecB,strategy="scaledPartial")
        
        print("Matrix A: \n", matA, "\n")
        print("Vector b: \n", vecB, "\n")
        print("Vector Sol: \n", vecSol, "\n")
        
        print( verification.verifySystemSolution(matA,vecB,vecSol) )
