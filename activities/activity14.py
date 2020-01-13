import math
import numpy as np

import matplotlib.pyplot as plt

# Datos de prueba
#x_datos = [ 1.,2.,3. ]
#y_datos = [ 3.98,5.44,8.95 ]


# Gráfica de los datos del paciente
x_datos = [ 1.,2.,3.,4.,5.,6.,7.,8.]
y_datos = [ 8.0,12.3,15.5,16.8,17.1,15.8,15.2,14.0 ]

plt.plot(x_datos,y_datos, marker='*', linestyle='', color='red')
plt.title('Registro de concentración en sangre de norfluoxetina')
plt.xlabel('Tiempo (horas)')
plt.ylabel('Concentración (ng/ml)')
plt.show()



# Función a ajutarse
# Creo que la función es exponencial con alguna distribución



# Ajuste con polinomio
polinomio = np.poly1d( np.polyfit(x_datos,y_datos,4) )

x_polinomio = np.linspace(1.,8.,101)
y_polinomio = polinomio(x_polinomio)

plt.plot(x_polinomio,y_polinomio, color='blue')
plt.plot(x_datos,y_datos, marker='*', linestyle='', color='red')
plt.title('Registro de concentración en sangre de norfluoxetina\n(Ajuste con Polinomio)')
plt.xlabel('Tiempo (horas)')
plt.ylabel('Concentración (ng/ml)')
plt.show()



# Ajuste con exponencial
y_logaritmo = list( map(math.log,y_datos) )

S_x    = sum( [ x_datos[i]     for i in range(len(x_datos)) ] )
S_LNy  = sum( [ y_logaritmo[i] for i in range(len(x_datos)) ] )
S_xx   = sum( [ x_datos[i]**2  for i in range(len(x_datos)) ] )
S_xLNy = sum( [ x_datos[i]*y_logaritmo[i] for i in range(len(x_datos)) ] )

numerador   = len(x_datos) * S_xLNy - S_x * S_LNy
denominador = len(x_datos) * S_xx   - S_x**2

b = numerador/denominador



numerador   =    S_xx * S_LNy      -  S_xLNy * S_x
denominador = len(x_datos) * S_xx  -    S_x**2

a = math.exp(numerador/denominador)

x_exponencial = np.linspace(1.,7.,101)
y_exponencial = [ a * math.exp(b*x_exponencial[i]) for i in range( len(x_exponencial) ) ]

plt.plot(x_exponencial,y_exponencial, color='green')
plt.plot(x_datos,y_datos, marker='*', linestyle='', color='red')
plt.title('Registro de concentración en sangre de norfluoxetina\n(Ajuste con Exponencial)')
plt.xlabel('Tiempo (horas)')
plt.ylabel('Concentración (ng/ml)')
plt.show()
