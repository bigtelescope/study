import numpy as np
import random as rand

from matplotlib import pyplot

features = np.array([1,2,3,5,6,7])
labels = np.array([155, 197, 244, 356, 407, 448])

print(features, labels)


def plot_points(features, labels):
    X = np.array(features)
    y = np.array(labels)
    pyplot.scatter(X, y)
    pyplot.xlabel('number of rooms')
    pyplot.ylabel('prices')


def draw_line(slope, y_intercept, color='grey', linewidth=0.7, starting=0, ending=8):
    x = np.linspace(starting, ending, 1000)
    pyplot.plot(x, y_intercept + slope*x, linestyle='-', color=color, linewidth=linewidth)


# как работает метод: в каждой итерации выбирается произвольная опорная точка, к которой
# поворачивается и подвигается прямая путем изменения линейной функции последней
def square_regression(features, labels, learning_rate=0.01, iterations=1000):
    base_weight, base_offset = rand.random(), rand.random()
    print('initial params: ', base_weight, base_offset)
    for iteration in range(iterations):
        # опорная точка в текущей итерации
        ref_point = rand.randint(0, len(features) - 1)

        # предсказанное значение для опорной точки
        predict_value = features[ref_point] * base_weight + base_offset

        # изменение смещения в текущей итерации
        base_offset += learning_rate * (labels[ref_point] - predict_value)

        # изменение наклона в текущей итерации
        # множитель, отображающий наклон таргетной прямой = вес (features[ref_point])
        base_weight += learning_rate * features[ref_point] * (labels[ref_point] - predict_value)

    return base_weight, base_offset


weight, offset = square_regression(features, labels)
plot_points(features, labels)
draw_line(weight, offset)
pyplot.show()
