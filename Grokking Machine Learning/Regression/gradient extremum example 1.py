# в чем задача: найти минимум фнукции f(xy) = (x - 3)^2 + 2(y + 1)^2
# шаг итерации learning_rate = 0.1
# начальная точка (0, 0)

import matplotlib.pyplot as plt
import sympy as sp

def x_derivative(x):
    return 2 * (x - 3)


def y_derivative(y):
    return 4 * (y + 1)


def gradient_descent(x_start, y_start, learning_rate=0.01, iterations=1000):
    x_current = x_start
    y_current = y_start
    iteration_metadata = []
    for i in range(iterations):
        iteration_metadata.append((x_current, y_current, x_derivative(x_current), y_derivative(y_current)))
        x_current = x_current - learning_rate * x_derivative(x_current)
        y_current = y_current - learning_rate * y_derivative(y_current)

    return x_current, y_current, iteration_metadata


# без ручного подсчета функции частных производных
def gradient_descent_auto(x_start, y_start, learning_rate=0.01, iterations=1000):
    # перевод аргументов и функции в выражение из переменных
    x, y = sp.symbols('x y')
    f = (x - 3)**2 + 2*((y + 1)**2)

    # функция (буквально сущность - функция), вычисляющая производную f по второму аргументу в формате символов (без чисел)
    df_x = sp.diff(f, x)
    df_y = sp.diff(f, y)

    # функция (буквально сущность - функция), принимающая координаты точки, для которой вернет численное значение производной по одной из переменных
    df_x_value_func = sp.utilities.lambdify((x, y), df_x, 'numpy')
    df_y_value_func = sp.utilities.lambdify((x, y), df_y, 'numpy')

    for i in range(iterations):
        x_start -= learning_rate * df_x_value_func(x_start, y_start)
        y_start -= learning_rate * df_y_value_func(x_start, y_start)

    return x_start, y_start


print(gradient_descent_auto(0, 0))


# строит процесс подбора точек минимума + прогресс шага
def plot_both_separate_subplots(data):
    """
    Рисует (x1,y1) и (x2,y2) в двух разных subplots.
    """
    x1 = [item[0] for item in data]
    y1 = [item[1] for item in data]
    x2 = [item[2] for item in data]
    y2 = [item[3] for item in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 строка, 2 столбца

    # График 1: (x1, y1)
    ax1.plot(x1, y1, 'b-o', label='x1 vs y1')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('y1')
    ax1.set_title('График 1: x1 по y1')
    ax1.grid(True)
    ax1.legend()

    # График 2: (x2, y2)
    ax2.plot(x2, y2, 'r--s', label='x2 vs y2')
    ax2.set_xlabel('x2')
    ax2.set_ylabel('y2')
    ax2.set_title('График 2: x2 по y2')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()  # Автоматическая настройка отступов
    plt.show()


# x, y, metadata = gradient_descent(0, 0)
# print(x, y) # 3, -1
# plot_both_separate_subplots(metadata)
