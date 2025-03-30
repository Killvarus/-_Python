from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, roc_auc_score,precision_recall_curve
import numpy as np
"""
TODO: make additional imports here
"""
np.random.seed(42)
data = fetch_openml(data_id=42608)
X, y = data['data'].drop(columns='Outcome').values, data['data']['Outcome'].astype(int).values

X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

"""
In the following part of code specify algorithms with their own parameters by yourself
"""
tree = DecisionTreeClassifier()
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(probability=True)
sc = StandardScaler()
sc.fit(X_train)
X_train, x_test = sc.transform(X_train), sc.transform(x_test)
tree.fit(X_train,y_train)
lr.fit(X_train,y_train)
knn.fit(X_train,y_train)
svm.fit(X_train,y_train)
tree_pred = tree.predict(x_test)
lr_pred = lr.predict(x_test)
knn_pred = knn.predict(x_test)
svm_pred = svm.predict(x_test)
preds = [tree_pred,lr_pred,knn_pred,svm_pred]
roc_auc_score_= []
auc_ = []
models = [tree,lr,knn,svm]
for i in range(4):
    roc_auc_score_.append(roc_auc_score(y_test, preds[i]))
    knn_scores = models[i].predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, knn_scores)
    knn_auc_pr = auc(recall, precision)
    auc_.append( knn_auc_pr)

print(roc_auc_score_,auc_)
