from sklearn.datasets import fetch_openml
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
np.random.seed(42)
lr = LogisticRegression()
sc = StandardScaler()
mnist = fetch_openml('mnist_784')
X = mnist.data.to_numpy()
y = mnist.target.to_numpy()
X = X[:2000]
y = y[:2000]
X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
sc.fit(X_train)
X_train, x_test = sc.transform(X_train), sc.transform(x_test)
N_COMPONENTS = [1,3,5,10,15,20,30,40,50,60]
itog = []
for i in N_COMPONENTS:
    pca = PCA(n_components=i)
    X_tr = pca.fit_transform(X_train)
    x_te = pca.transform(x_test)
    lr.fit(X_tr, y_train)
    acc = accuracy_score(lr.predict(x_te), y_test)
    itog.append(acc)
print(itog)
print(max(itog),N_COMPONENTS[itog.index(max(itog))])