import math
import numpy as np

import matplotlib.pyplot as plt



def g(x):
    return x**3 - 3.23*(x**2) -5.54*x + 9.84


def h(x):
    return math.tan(x) - np.tanh(x)


def metodoIncremental(f,a,b,dx):
    xIzq = a     ; fIzq = f(xIzq)
    xDer = a + dx; fDer = f(xDer)
    
    while( fIzq * fDer > 0.0 ):
        if( xDer >= b ): 
            return None,None
        else:
            xIzq  = xDer; fIzq = fDer
            xDer += dx  ; fDer = f(xDer)
    else:
        return xIzq,xDer


def metodoBiseccion(a,b,f,tolerancia=1.0e-5):

    xIzq = a
    fIzq = f(a)
    if( fIzq  == 0.0 ):
        return a
    
    xDer = b
    fDer = f(b)
    if( fDer == 0.0 ):
        return b
    
    while( abs(xDer-xIzq) > tolerancia ):
        
        xCen = xIzq + (xDer-xIzq)/2;
        fCen = f(xCen)
        
        x = np.linspace(-5.0,8.0,1001)
        y_f = [ f(x[i]) for i in range(len(x)) ]
        
        if( fCen == 0.0 ): 
            return xCen
        elif( fIzq*fCen <= 0.0): 
            xDer = xCen
            fDer = fCen
        elif( fCen*fDer <= 0.0):
            xIzq = xCen
            fIzq = fCen
        
    return xIzq + (xDer-xIzq)/2


def metodoSecante(a,b,f,tolerancia=1.0e-5):
    
    x_iless1 = a
    x_i      = b
    
    f_iless1 = f(x_iless1)
    f_i      = f(x_i)
    
    while( abs(x_i - x_iless1) > tolerancia ):
        x_iplus1 = x_i - f_i*( (x_i-x_iless1) / (f_i-f_iless1) )
        f_iplus1 = f(x_iplus1)
        
        x_iless1 = x_i
        x_i      = x_iplus1
        
        f_iless1 = f_i
        f_i      = f_iplus1
    
    return (x_iless1+x_i) / 2.

def grafica(f,x,a,b,x_biseccion,x_secante,nombre,formula):
    

    y_f = [ f(x[i]) for i in range(len(x)) ]
    
    plt.plot(x,y_f, color='blue')
    plt.plot(x,x*0, color='black')
    plt.plot(x_biseccion,f(x_biseccion), marker='o',color='red', markersize=4)
    plt.plot(x_secante,f(x_secante), marker='o',color='yellow', markersize=4)
    plt.plot(a,f(a), marker='*',color='green')
    plt.plot(b,f(b), marker='*',color='green')
    plt.title("Función " + nombre)
    plt.xlabel("x")
    plt.ylabel(formula)
    plt.show()
    


# Función G

#x   = np.linspace(-4.0,6.0,1001)
x   = np.linspace(.0,2.0,1001)
a,b = metodoIncremental(g,0.0,25.0,0.5)

x_G_biseccion = metodoBiseccion(a,b,g)
x_G_secante   = metodoSecante(a,b,g)

print("Función G")
print("   Usando metodo de bisección:  " + str(x_G_biseccion))
print("   Usando metodo de la secante: " + str(x_G_secante))
grafica(g,x,a,b,x_G_biseccion,x_G_secante,"G","G(x) = x^3 - 3.23 * x^2 - 5.54 * x + 9.84")



# Función H

x   = np.linspace(7.0,7.4,1001)
a,b = metodoIncremental(h,7.0,7.4,0.1)

x_H_biseccion = metodoBiseccion(a,b,h)
x_H_secante   = metodoSecante(a,b,h)

print("Función H")
print("   Usando metodo de bisección:  " + str(x_H_biseccion))
print("   Usando metodo de la secante: " + str(x_H_secante))
grafica(h,x,a,b,x_H_biseccion,x_H_secante,"H","H(x) = tan(x) - tanh(x)")
