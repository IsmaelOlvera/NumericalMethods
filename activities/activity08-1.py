# -*- coding: utf-8 -*-
## activity 08.1


import sys
from pathlib import Path

parentPath  = Path().absolute().parent
siblingPath = parentPath.joinpath('modules')
sys.path.append( str(siblingPath) )

import numpy as np

from basics          import verification
from decompositionLU import doolittle



if __name__ == '__main__':
    
    matA = np.array( [ [3.,-13.,9.,3.], [-6.,4.,1.,-18.], [6.,-2.,2.,4.], [12.,-8.,6.,10.] ] )
    vecB = np.array( [ -19.,-34.,16.,26. ] )
    
    vecSol = doolittle.solveSystem(matA,vecB,strategy="scaledPartial")
    
    print("Matrix A: \n", matA, "\n")
    print("Vector b: \n", vecB, "\n")
    print("Vector Solución: \n", vecSol, "\n")
    
    print( verification.verifySystemSolution(matA,vecB,vecSol) )
