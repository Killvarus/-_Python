import numpy as np

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
X = np.array([[1,2,3], [4,5,6], [7,8,9]])
Y = np.array([1, 2, 3])
s = sample(X, 2)

bootstrap_indices = s.bootstrap_sample(X)
print(bootstrap_indices)
print(s.idx_subspace)
X_sampled, y_sampled = s.get_subsample(X, Y, s.idx_subspace, bootstrap_indices)
print(X_sampled)
print(y_sampled)