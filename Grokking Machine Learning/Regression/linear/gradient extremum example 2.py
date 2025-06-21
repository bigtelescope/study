# задача: найти минимум функции f(xy) = sqrt(x^2 + y^2 + 1) + x/2 - y/2

import sympy as sp
import math

def gradient_descent(x_start, y_start, learning_rate=0.01, iterations=10000):
    x, y = sp.symbols('x y')
    f = (x**2 + y**2 + 1)**(1/2) + x/2 - y/2

    # символьные производные функции по каждой из переменных
    df_x = sp.diff(f, x)
    df_y = sp.diff(f, y)

    # численные функции производных по каждой переменной. Должна принять значение точек, от которой зависит
    df_x_value = sp.utilities.lambdify((x, y), df_x, 'numpy')
    df_y_value = sp.utilities.lambdify((x, y), df_y, 'numpy')

    for i in range(iterations):
        x_start -= learning_rate * df_x_value(x_start, y_start)
        y_start -= learning_rate * df_y_value(x_start, y_start)

    return x_start, y_start


print(gradient_descent(0, 0))

