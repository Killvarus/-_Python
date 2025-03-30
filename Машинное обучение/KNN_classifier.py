import numpy as np

def r2(y_true, y_pred):
  y_error = y_true - y_pred
  s = 1-(np.var(y_error)/np.var(y_true))
  return s

def r2(y_true, y_pred):
  y_error = y_pred -y_true
  s = 1-(np.var(y_error)/np.var(y_true))
  return s

class KNN_classifier:
    def __init__(self, n_neighbors: int, **kwargs):
        self.K = n_neighbors  # Количество ближайших соседей

    def fit(self, x: np.array, y: np.array):
        """
        Запоминает обучающую выборку для последующего предсказания.
        """
        self.x_train = x  # Запоминаем обучающую выборку
        self.y_train = y  # Запоминаем метки классов

    def predict(self, x: np.array):
        
        predictions = []  # Список для предсказаний
        
        for test_point in x:
            # Вычисляем расстояния от тестовой точки до всех точек обучающей выборки
            distances = np.linalg.norm(self.x_train - test_point, axis=1)
            
            # Находим индексы K ближайших соседей
            nearest_neighbor_ids = np.argsort(distances)[:self.K]
            
            # Извлекаем классы этих соседей
            nearest_neighbor_classes = self.y_train[nearest_neighbor_ids]
            
            # Определяем класс, который встречается чаще всего среди соседей
            unique, counts = np.unique(nearest_neighbor_classes, return_counts=True)
            most_common_class = unique[np.argmax(counts)]
            
            # Добавляем предсказание в список
            predictions.append(most_common_class)
        
        return np.array(predictions)
  
X =  np.array([[ 0.56510722,  0.68599596, -0.92388505, -0.29546048, -0.12437532],
       [-0.79617537,  0.98406791,  1.19542652, -0.05626863, -0.69868076],
       [ 0.9629688 , -1.00423925, -0.53842833, -0.23744358,  0.83226685],
       [ 0.24671269, -0.41624448,  0.81679337,  1.59227446,  0.16192583],
       [-0.36972363,  0.17425997,  1.33668078,  1.16687907,  0.31709134],
       [-1.30482844, -0.05354323, -0.88862186, -1.121785  , -0.78442809],
       [-0.53975018,  0.90074877, -1.09317408,  1.52989481, -0.43375015],
       [-0.64709803, -0.09775791,  1.3506503 , -1.46957788,  1.63325543],
       [-0.73858464, -0.60678229,  0.31420272, -0.43100129, -0.37665876],
       [-0.29208809, -0.68795722,  0.06586655,  0.9583851 ,  1.70640775]])
y = np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 1])
knn = KNN_classifier(n_neighbors=3)
knn.fit(X, y)
x_test=np.array([[-0.12489725,  0.65791923, -0.73112495,  1.42660225,  1.64728976],
       [ 0.01913388, -1.11351208, -0.63244098, -0.98121107,  0.38060892],
       [-0.92074931,  1.39812225,  0.39692147,  0.7717827 ,  0.44604002]])
print(knn.predict(x_test))