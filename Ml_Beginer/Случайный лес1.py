import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Гиперпараметры, которые нужно будет настроить
N_ESTIMATORS = 10
MAX_DEPTH = 3
SUBSPACE_DIM = 2

class random_forest:
    def __init__(self, n_estimators: int, max_depth: int, subspaces_dim: int, random_state: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subspaces_dim = subspaces_dim
        self.random_state = random_state
        self._estimators = []  # Список деревьев
        self.subspace_idx = []  # Логирование выбранных признаков для каждого дерева

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)  # Случайная выборка с возвратом
        return X[indices], y[indices]

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            # Генерация случайного подпространства признаков
            feature_indices = np.random.choice(X.shape[1], self.subspaces_dim, replace=False)
            self.subspace_idx.append(feature_indices)

            # Создание обучающей подвыборки
            X_sample, y_sample = self._bootstrap_sample(X, y)
            X_sample_subspace = X_sample[:, feature_indices]  # Используем только выбранные признаки

            # Обучение дерева на подвыборке
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_sample_subspace, y_sample)
            self._estimators.append(tree)

    def predict(self, X):
        # Сбор предсказаний от всех деревьев
        predictions = np.zeros((X.shape[0], self.n_estimators), dtype=int)
        for i, tree in enumerate(self._estimators):
            feature_indices = self.subspace_idx[i]
            X_subspace = X[:, feature_indices]  # Используем те же признаки, что и при обучении
            predictions[:, i] = tree.predict(X_subspace)
        
        # Финальное предсказание — мажоритарное голосование
        final_predictions = [np.bincount(pred_row).argmax() for pred_row in predictions]
        return np.array(final_predictions)

my_dict = {
  'N_ESTIMATORS':0,
  'MAX_DEPTH' : 0,
  'SUBSPACE_DIM':0,
  'accuracy' : 0
}



X, y = load_iris(return_X_y=True)
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
forest = random_forest(1,2,3,42)
forest.fit(X_train,y_train)
print(forest.predict(x_test))
# for i in range(20):
#   N_ESTIMATORS = i+1
#   for j in range(20):
#     MAX_DEPTH = j+1
#     for g in range(4):
#       SUBSPACE_DIM = g+1

# # u = pd.DataFrame(X)
# # print(u.shape)
#       X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
#       forest = random_forest(N_ESTIMATORS,MAX_DEPTH,SUBSPACE_DIM,42)
#       forest.fit(X_train,y_train)
#       if accuracy_score(y_test,forest.predict(x_test)) > my_dict['accuracy']:
#         my_dict['MAX_DEPTH'] = j+1
#         my_dict['N_ESTIMATORS']=i+1
#         my_dict['SUBSPACE_DIM']=g+1
#         my_dict['accuracy'] = accuracy_score(y_test,forest.predict(x_test))
#       # print(accuracy_score(y_test,forest.predict(x_test)))
# print(my_dict)
# Загрузка данных Ирисов Фишера и тестирование модели
# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# # Подбор лучших гиперпараметров (здесь вы можете экспериментировать)
# N_ESTIMATORS = 10
# MAX_DEPTH = 4
# SUBSPACE_DIM = 3

# # Создание и обучение модели
# model = RandomForest(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, subspaces_dim=SUBSPACE_DIM, random_state=42)
# model.fit(X_train, y_train)

# # Оценка точности на тестовой выборке
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")
