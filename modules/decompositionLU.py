# -*- coding: utf-8 -*-
## module decompositionLU


import numpy as np
import copy

import substitution
import verification
import gaussianElimination


########## INPUT  ##########
## matA -> Matrix [n*n]
########## OUTPUT ##########
## matL -> Matrix [n*n]
## matU -> Matrix [n*n]  |  matA = matL * matU
def simpleLU(matA):
    
    n    = len(matA)
    vecB = np.zeros(n,dtype=float)
    
    matU,vecBmod,matL = gaussianElimination.elimination(matA,vecB)
    
    return matL,matU


########## INPUT  ##########
## matA   -> Matrix [n*n]
## vecB   -> Vector [n]
########## OUTPUT ##########
## vecX   -> Vector [n]  |  matA * vecX = vecB
##        => matA = matL * matU
##        => matL * (matU * vecX) = vecB
def doolittleMethod(matA,vecB):
    
    matL,matU = simpleLU(matA)
    vecY = substitution.forwardSubstitution(matL,vecB)
    # print(vecY)
    vecX = substitution.backwardSubstitution(matU,vecY)
    # print(vecX)
    return vecX


########## INPUT  ##########
## matA    -> Matrix [n*n]
########## OUTPUT ##########
## matAinv -> Matrix [n*n]  |  matA * matAinv = I
def findInverse(matA):
    
    n = len(matA)
    I = np.eye(n)
    
    matAinv = np.zeros((n,n),dtype=float)
    
    matL,matU = simpleLU(matA)
    
    for j in range(n):
        vecY         = substitution.forwardSubstitution(matL,I[:,j])
        matAinv[:,j] = substitution.backwardSubstitution(matU,vecY)
    
    return matAinv
    
    

if __name__ == "__main__":
    
    print("Enter any of the following options: ")
    print("\t A. Find LU Decomposition of A.")
    print("\t B. Solve Ax = b using Doolittle's Method.")
    print("\t C. Find inverse matrix of A.")
    caseInput = input("Input: ")
    
    
    if( caseInput == "A" ):
        matA      = np.array( [ [4.,1.,2.,-3.,5.], [-3.,3.,-1.,4.,-2.], [-1.,2.,5.,1.,3.], [5.,4.,3.,-1.,2.], [1.,-2.,3.,-4.,5.] ] )
        matL,matU = simpleLU(matA)
        
        print("Matrix L: \n", matL, "\n")
        print("Matrix U: \n", matU, "\n")
        
        print( verification.verifySolution(matL,matA,matU) )
        
        
    elif( caseInput == "B" ):
        matA = np.array( [ [4.,1.,2.,-3.,5.], [-3.,3.,-1.,4.,-2.], [-1.,2.,5.,1.,3.], [5.,4.,3.,-1.,2.], [1.,-2.,3.,-4.,5.] ] )
        vecB = np.array( [ -16.,20.,-4.,-10.,3. ] )
    
        vecSol = doolittleMethod(matA,vecB)
        
        print("Matrix A: \n", matA, "\n")
        print("Vector b: \n", vecB, "\n")
        print("Vector Solución: \n", vecSol, "\n")
        
        print( verification.verifySolution(matA,vecB,vecSol) )
        
        
    elif( caseInput == "C" ):
        matA    = np.array( [ [1.,-3.,7.], [0.,3.,-2.], [-2.,6.,1.] ] )
        matAinv = findInverse(matA)
        
        print("Matrix A: \n", matA, "\n")
        print("Matrix A ^ {-1}: \n", matAinv, "\n")
        
        print( verification.verifyInverse(matA,matAinv) )
