
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42) # Разбейте выборку на train и test
scaler = StandardScaler()
"""
Обучите и примените StandardScaler
"""
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def compare_svm_kernels(X_train, X_test, y_train, y_test):
    """
      Напишите функцию, определяющую наилучшее ядро для решения данной задачи
    """
    classifiers = ['linear', 'poly', 'rbf', 'sigmoid']
    compare = {}
    for i in classifiers:
       model = SVC(kernel=i, C=1.0).fit(X_train,y_train)
       y_pred = model.predict(X_test)
       score= accuracy_score(y_test,y_pred)
       compare[i] = score
    print(compare)
    
    

compare_svm_kernels(X_train, X_test, y_train, y_test)
