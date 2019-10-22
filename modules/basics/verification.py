# -*- coding: utf-8 -*-
## module basics.verification


import numpy as np


########## INPUT  ##########
## matA  -> Matrix [n*n]
## matA1 -> Matrix [n*n]
## matA2 -> Matrix [n*N]
########## OUTPUT ##########
## String -> ( matA = matA1 * matA2 ) is correct?
def verifyMatrixMultiplication(matA,matA1,matA2):
    
    expectedAnswer = np.matmul(matA1,matA2)
    wA,wB = np.asarray(matA), np.asarray(expectedAnswer)
    
    if( bool( np.asarray( abs(wA-wB) < 10**(-10) ).all() ) ):
        return "All right"
    return "Something is wrong"


########## INPUT  ##########
## matA   -> Matrix [n*n]
## vecB   -> Vector [n]
## vecSol -> Vector [n]
########## OUTPUT ##########
## String -> ( matA * vecSol = vecB ) is correct?
def verifySystemSolution(matA,vecB,vecSol):
    
    expectedAnswer = np.matmul(matA,vecSol)
    wA,wB = np.asarray(expectedAnswer), np.asarray(vecB)
    
    if( bool( np.asarray( abs(wA-wB) < 10**(-10) ).all() ) ):
        return "All right"
    return "Something is wrong"


########## INPUT  ##########
## matA    -> Matrix [n*n]
## matAinv -> Matrix [n*n]
########## OUTPUT ##########
## String -> ( matA * matAinv = I ) is correct?
def verifyInverse(matA,matAinv):
    
    n = len(matA)
    
    expectedOnes = np.matmul(matA,matAinv)
    wA,wB = np.asarray(expectedOnes), np.asarray( np.eye(n,dtype=float) )
    
    if( bool( np.asarray( abs(wA-wB) < 10**(-10) ).all() ) ):
        return "All right"
    return "Something is wrong"
