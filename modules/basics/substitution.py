# -*- coding: utf-8 -*-
## module basics.substitution


import numpy as np
import copy


########## INPUT  ##########
## matU   -> Upper Triangular Matrix [n*n]
## vecB   -> Vector [n]
########## OUTPUT ##########
## vecSol -> Vector [n]  |  matU * vecSol = vecB
def backwardSubstitution(matU,vecB):
    
    matA = copy.deepcopy(matU)
    vecB = copy.deepcopy(vecB)
    
    n      = len(vecB)
    vecSol = np.zeros(n,dtype=float)
    
    for i in reversed(range(n)):
        vecSol[i] = ( vecB[i] - np.dot(matU[i,i+1:n],vecSol[i+1:n]) )  /  matU[i,i]
    
    # Use to debug
    # print(vecSol)
    
    return vecSol


########## INPUT  ##########
## matL   -> Lower Triangular Matrix [n*n]
## vecB   -> Vector [n]
########## OUTPUT ##########
## vecSol -> Vector [n]  |  matL * vecSol = vecB
def forwardSubstitution(matL,vecB):
    
    matL = copy.deepcopy(matL)
    vecB = copy.deepcopy(vecB)
        
    n      = len(vecB)
    vecSol = np.zeros(n,dtype=float)
    
    for i in range(n):
        vecSol[i] = ( vecB[i] - sum( [ matL[i,j]*vecSol[j] for j in range(i) ] ) )  /  matL[i,i]
        
    # Use to debug
    # print(vecSol)
    
    return vecSol



if __name__ == "__main__":
    
    matA   = np.array(0)
    vecB   = np.array(0)
    vecSol = np.array(0)
    
    print("Enter any of the following options: ")
    print("\t A. Backward Substitution Example (Very Simple).")
    print("\t B. Forward  Substitution Example (Very Simple).")
    caseInput = input("Input: ")
    
    if( caseInput == "A" ):
        matA = np.array( [ [1.,1.,1.,1.,1.], [0.,1.,1.,1.,1.], [0.,0.,1.,1.,1.], [0.,0.,0.,1.,1.], [0.,0.,0.,0.,1.] ] )
        vecB = np.array( [ 15.,10.,6.,3.,1. ] )
        vecSol = backwardSubstitution(matA,vecB)
    elif( caseInput == "B" ):
        matA = np.array( [ [1.,0.,0.,0.,0.], [1.,1.,0.,0.,0.], [1.,1.,1.,0.,0.], [1.,1.,1.,1.,0.], [1.,1.,1.,1.,1.] ] )
        vecB = np.array( [ 1.,3.,6.,10.,15. ] )
        vecSol = forwardSubstitution(matA,vecB)
        
    
    print("Matrix A: \n", matA, "\n")
    print("Vector b: \n", vecB, "\n")    
    print("Vector Sol: \n", vecSol, "\n")
