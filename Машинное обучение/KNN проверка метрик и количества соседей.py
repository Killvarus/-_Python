import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
random_seed = 4238


np.random.seed(random_seed)
n_splits = 3

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

"""
  Здесь Вам предлагается написать тело цикла для подбора оптимального K
  Результаты оценки алгоритма при каждом отдельно взятом K рекомендуем записывать в список cv_scores
"""
cv_scores=np.empty((50,3))
for k in range(1, 51):
    clf = KNeighborsClassifier(n_neighbors=k) #Евклид
    scores = cross_val_score(clf, X, y, cv=n_splits)
    cv_scores[k-1][0] = np.mean(scores)

    clf = KNeighborsClassifier(n_neighbors=k,p=1) #Манхетон
    scores = cross_val_score(clf, X, y, cv=n_splits)
    cv_scores[k-1][1] = np.mean(scores)

    clf = KNeighborsClassifier(n_neighbors=k,metric='cosine') #кос
    scores = cross_val_score(clf, X, y, cv=n_splits)
    cv_scores[k-1][2] = np.mean(scores)

    pass
means = np.mean(cv_scores, axis=1)
idx = np.where(means == means.max())
print(cv_scores[idx[0][0]])
