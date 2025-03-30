import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
np.random.seed(42)
class sample(object):
  def __init__(self, X, n_subspace):
    self.idx_subspace = self.random_subspace(X, n_subspace)

  def __call__(self, X, y):
    idx_obj = self.bootstrap_sample(X)
    X_sampled, y_sampled = self.get_subsample(X, y, self.idx_subspace, idx_obj)
    return X_sampled, y_sampled

  @staticmethod
  def bootstrap_sample(X):
    """
    Заполните тело этой функции таким образом, чтобы она возвращала массив индексов выбранных при помощи бэггинга индексов.
    Пользуйтесь только инструментами, реализованными в numpy.random, выставляя везде, где это необходимо, random_state=42
    """
    # choise = np.random.choice(X[0], X.shape[1], replace=True, p=None)
    # indices = [np.where(choise == elem)[0][0] for elem in choise]
    #rng = np.random.default_rng(42)
    total = X.shape[0]
    selected_indices = np.random.choice(total, total, replace=True)
    return np.unique(selected_indices)
  @staticmethod
  def random_subspace(X, n_subspace):
    """
    Заполните тело этой функции таким образом, чтобы она возвращала массив индексов выбранных при помощи метода случайных подпространств признаков
    Количество этих признаков передается при помощи аргумента n_subspace
    Пользуйтесь только инструментами, реализованными в numpy.random, выставляя везде, где это необходимо, random_state=42
    """
    # rng = np.random.default_rng(42)
    total_features = X.shape[1]
    selected_indices = np.random.choice(total_features, n_subspace,replace=False)
    return(np.unique(selected_indices))

  @staticmethod
  def get_subsample(X, y, idx_subspace, idx_obj):
    """
    Заполните тело этой функции таким образом, чтобы она возвращала подвыборку x_sampled, y_sampled
    по значениям индексов признаков(idx_subspace) и объектов(idx_obj) , которые должны в неё попасть
    """
    x_result = X[idx_obj, :][:, idx_subspace]  # Выбираем строки и столбцы по индексам
    y_result = y[idx_obj]  # Выбираем только объекты (цели)

   
    # for i in range(idx_subspace.shape[0]):6
    #   for j in range(idx_obj.shape[0]):
    #     result_x[i][j] = X[i][j]
    #     result_y[i][j] = y[i]
    return(x_result,y_result)



class random_forest(object):
  def __init__(self, n_estimators: int, max_depth: int, subspaces_dim: int, random_state: int):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.subspaces_dim = subspaces_dim
    self.random_state = random_state
    self._estimators = [] 
    self.subspace_idx=[]
    """
      Задайте все необходимые поля в рамках конструктора класса
    """

  def fit(self, X, y):
    np.random.seed(self.random_state)
    self._estimators = [] 
    self.subspace_idx=[]
    clf = DecisionTreeClassifier(
    max_depth=self.max_depth,
    random_state=self.random_state
    )
    sample_n = sample(X,self.subspaces_dim)
    for i in range(self.n_estimators):
        self.subspace_idx.append([sample_n.bootstrap_sample(X),sample_n.random_subspace(X,n_subspace=self.subspaces_dim)])
        # x_sample,y_sample = sample_n.get_subsample(X,y,sample_n.random_subspace(X,n_subspace=self.subspaces_dim),sample_n.bootstrap_sample(X))
        x_sample = X[:][:, sample_n.random_subspace(X,n_subspace=self.subspaces_dim)] 
        clf.fit(x_sample,y)
        (self._estimators).append(clf)
  def predict(self, X):
    sample_n = sample(X,self.subspaces_dim)
    pred = []
    predictions = np.zeros((X.shape[0], self.n_estimators))
    for i in range(self.n_estimators):
      x_result = X[:, :][:, (self.subspace_idx)[i][1]]
      # pred.append(self._estimators[i].predict(x_result))
      predictions[:, i] = self._estimators[i].predict(x_result)
      most_common_values = []
      for row in range(predictions.shape[0]):
          column = predictions[row, :]  # Извлекаем столбец
          unique_values, counts = np.unique(column, return_counts=True)
          most_common_value = unique_values[np.argmax(counts)]
          most_common_values.append(int(most_common_value))
      predictions = predictions.astype(int)
    # final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=pred)
    final_predictions = [np.bincount(pred_row).argmax() for pred_row in predictions]
    return(final_predictions)
my_dict = {
  'N_ESTIMATORS':0,
  'MAX_DEPTH' : 0,
  'SUBSPACE_DIM':0,
  'accuracy' : 0
}



X, y = load_iris(return_X_y=True)
# X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
# forest = random_forest(1,2,4,42)
# # forest.fit(X_train,y_train)
# print(forest.predict(x_test),y_test)
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
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
forest = random_forest(2,1,2,42)
forest.fit(X_train,y_train)
print(forest.predict(x_test),accuracy_score(y_test,forest.predict(x_test)))