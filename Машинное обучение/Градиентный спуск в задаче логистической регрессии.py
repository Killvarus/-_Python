import numpy as np

np.random.seed(42)

# Функция подсчета градиента
def gradient(y_true: int, y_pred: float, x: np.array) -> np.array:
    """
    y_true - истинное значение ответа для объекта x
    y_pred - значение степени принадлежности объекта x классу 1, предсказанное нашей моделью
    x - вектор признакового описания данного объекта

    На выходе ожидается получить вектор частных производных H по параметрам модели, предсказавшей значение y_pred
    """
    error = y_pred - y_true
    grad = np.concatenate((error * x, [error]))  # Добавляем свободный коэффициент в конце
    return grad

# Функция обновления весов
def update(alpha: np.array, gradient: np.array, lr: float) -> np.array:
    """
    alpha: текущее приближение вектора параметров модели
    gradient: посчитанный градиент по параметрам модели
    lr: learning rate, множитель перед градиентом в формуле обновления параметров
    """
    return alpha - lr * gradient

# Функция тренировки модели
def train(
    alpha0: np.array, x_train: np.array, y_train: np.array, lr: float, num_epoch: int
) -> np.array:
    """
    alpha0 - начальное приближение параметров модели
    x_train - матрица объект-признак обучающей выборки
    y_train - верные ответы для обучающей выборки
    lr - learning rate, множитель перед градиентом в формуле обновления параметров
    num_epoch - количество эпох обучения, то есть полных 'проходов' через весь датасет
    """
    alpha = alpha0.copy()
    for epo in range(num_epoch):
        for i, x in enumerate(x_train):
            # Логистическая функция для предсказания вероятности класса 1
            z = np.dot(alpha[:-1], x) + alpha[-1]  # Свободный коэффициент a0 в конце
            y_pred = 1 / (1 + np.exp(-z))

            # Считаем градиент
            grad = gradient(y_train[i], y_pred, x)

            # Обновляем веса
            alpha = update(alpha, grad, lr)
    
    return alpha
# Пример данных и начальные параметры
x_train = np.array([[0.5, 1.5], [1.0, -1.0], [-1.5, 2.0]])
y_train = np.array([1, 0, 1])
alpha0 = np.zeros(x_train.shape[1] + 1)  # Свободный коэффициент добавляется в конец
lr = 0.01
num_epoch = 1000

# Обучение модели
alpha_trained = train(alpha0, x_train, y_train, lr, num_epoch)
print("Обученные веса:", alpha_trained)
