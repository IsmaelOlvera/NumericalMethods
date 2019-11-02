# -*- coding: utf-8 -*-
## activity 08.2


import sys
from pathlib import Path

parentPath  = Path().absolute().parent
siblingPath = parentPath.joinpath('modules')
sys.path.append( str(siblingPath) )

import numpy as np

from decompositionLU import doolittle


def createMatrixA(n):
    matA = np.array( [ [ (0.+i+j)**2 for i in range(n) ] for j in range(n) ] )
    return matA


def createVectorB(n):
    vecB = np.array( [ sum([ (0.+i+j)**2 for i in range(n) ]) for j in range(n) ] )
    return vecB



if __name__ == '__main__':
    
    n = 3
    
    matA = createMatrixA(n)
    vecB = createVectorB(n)
    
    matL,matU,matPerm = doolittle.findLU(matA, strategy="scaledPartial")
    
    matPLU = np.matmul( matPerm, np.matmul(matL,matU) )
    
    vecSol = doolittle.solveSystem(matA,vecB,strategy="scaledPartial")
    
    print("N = ",n,"\n")
    
    print("Matrix A: \n", np.around(matA,7), "\n")
    print("Vector b: \n", np.around(vecB,7), "\n")
    
    print("Matrix L:   \n", np.around(matL  ,7), "\n")
    print("Matrix U:   \n", np.around(matU  ,7), "\n")
    print("Matrix PLU: \n", np.around(matPLU,7), "\n")
    
    print("Vector Sol: \n", np.around(vecSol,7), "\n")
