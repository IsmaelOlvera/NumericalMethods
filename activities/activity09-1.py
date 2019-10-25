# -*- coding: utf-8 -*-
## activity 09.1


import sys
from pathlib import Path

parentPath  = Path().absolute().parent
siblingPath = parentPath.joinpath('modules')
sys.path.append( str(siblingPath) )

import numpy as np

from basics           import verification
from iterativeMethods import jacobi,gaussSeidel



if __name__ == '__main__':
    
    matA = np.array( [  [ 4.,-1., 0.,-1., 0., 0., 0., 0., 0.],
                        [-1., 4.,-1., 0.,-1., 0., 0., 0., 0.],
                        [ 0.,-1., 4., 0., 0.,-1., 0., 0., 0.],
                        [-1., 0., 0., 4.,-1., 0.,-1., 0., 0.],
                        [ 0.,-1., 0.,-1., 4.,-1., 0.,-1., 0.],
                        [ 0., 0.,-1., 0.,-1., 4., 0., 0.,-1.],
                        [ 0., 0., 0.,-1., 0., 0., 4.,-1., 0.],
                        [ 0., 0., 0., 0.,-1., 0.,-1., 4.,-1.],
                        [ 0., 0., 0., 0., 0.,-1., 0.,-1., 4.]   ] )
    
    vecB = np.array( [0.,0.,0.,0.,0.,0.,1.,1.,1.] )
    
    vecInitSol = np.zeros( len(vecB) )
    threshold  = 1 * 10**(-10)
    
    vecFinalSol_Jac,iterations_Jac = jacobi.solve(matA,vecB,vecInitSol,threshold)
    vecFinalSol_GS ,iterations_GS  = gaussSeidel.solve(matA,vecB,vecInitSol,threshold)
    
    print("Matrix A: \n", matA, "\n")
    print("Vector B: \n", vecB, "\n")
    print("Vector Initial Solution: \n", vecInitSol, "\n")
    print("Threshold: \n", threshold, "\n")
    
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - ","\n")
    
    print("Jacobi method iterations needed: ", iterations_Jac, "\n")
    print("Vector Final Solution with Jacobi method: \n", vecFinalSol_Jac, "\n")
    print( verification.verifySystemSolution(matA,vecB,vecFinalSol_Jac), "\n" )
    
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - ","\n")
    
    print("Gauss-Seidel method iterations needed: ", iterations_GS, "\n")
    print("Vector Final Solution with Gauss-Seidel method: \n", vecFinalSol_GS, "\n")
    print( verification.verifySystemSolution(matA,vecB,vecFinalSol_GS), "\n" )
