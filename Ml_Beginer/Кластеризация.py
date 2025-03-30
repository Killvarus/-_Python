import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Загрузка данных
datasets = [pd.read_csv(f'D:/Desktop/Вся прога на питоне/Python МФК/{i}.csv', index_col=None) for i in range(1, 6)]

# Функция для визуализации данных
def plot_data(X, labels=None):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Функция для оценки качества кластеризации
def evaluate_clustering(X, y_true, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    
    # Получение меток кластеров
    y_pred = kmeans.labels_
    
    # Для корректной оценки можно привести y_pred к правильному порядку
    # Сравниваем предсказанные и реальные метки
    accuracy = accuracy_score(y_true, y_pred)
    
    # Визуализация результата
    plot_data(X, y_pred)
    
    # Если более 90% правильных меток, считаем алгоритм удачным
    if accuracy > 0.9:
        print(f"Кластеризация успешна для {n_clusters} кластеров!")
    else:
        print(f"Кластеризация не удалась для {n_clusters} кластеров.")
    
    return accuracy

# Применяем алгоритм для каждого датасета
valid_datasets = []
for i, dataset in enumerate(datasets, 1):
    X = dataset[['x', 'y']].values
    y_true = dataset['class'].values
    n_clusters = len(np.unique(y_true))  # Количество кластеров для данного датасета
    
    print(f"Обрабатываем датасет {i} с количеством кластеров {n_clusters}")
    
    accuracy = evaluate_clustering(X, y_true, n_clusters)
    
    if accuracy > 0.9:
        valid_datasets.append(i)

# Выводим результат
print(f"Подходящие датасеты для K-Means: {''.join(map(str, valid_datasets))}")
