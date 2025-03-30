import numpy as np

class KMeans(object):
    def __init__(self, K, init):
        """
        Инициализация алгоритма KMeans
        :param K: Количество кластеров
        :param init: Массив с начальными центрами кластеров (размерность KxM)
        """
        self.K = K
        self.centroids = init  # Начальные центры кластеров

    def fit(self, X):
        """
        Обучение модели KMeans.
        :param X: Данные для кластеризации (матрица объектов)
        """
        prev_centroids = np.zeros_like(self.centroids)
        converged = False
        
        while not converged:
            # Шаг 1: Распределение объектов по кластерам
            labels = self.predict(X)

            # Шаг 2: Обновление центров кластеров
            for k in range(self.K):
                # Все точки, принадлежащие кластеру k
                points_in_cluster = X[labels == k]
                if points_in_cluster.shape[0] > 0:
                    # Новые центры кластеров - среднее значение всех точек в кластере
                    self.centroids[k] = np.mean(points_in_cluster, axis=0)

            # Шаг 3: Проверка на сходимость (изменение центров кластеров)
            max_change = np.max(np.linalg.norm(self.centroids - prev_centroids, axis=1))
            if max_change <= 0.001:
                converged = True

            prev_centroids = np.copy(self.centroids)

    def predict(self, X):
        """
        Кластеризация новых объектов на основании текущих центров кластеров.
        :param X: Новый набор данных
        :return: Метки кластеров для каждого объекта
        """
        # Вычисление евклидовых расстояний от каждой точки до каждого центра
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        # Для каждой точки выбираем ближайший центр
        return np.argmin(distances, axis=1)
