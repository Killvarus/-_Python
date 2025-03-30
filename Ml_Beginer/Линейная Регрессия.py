import numpy as np
np.random.seed(42)

class LinearRegression:
    def __init__(self, **kwargs):
        self.coef_ = None
        pass

    def fit(self, x: np.array, y: np.array):
        self.x_train = x  # Запоминаем обучающую выборку
        self.y_train = y # Запоминаем метки классов
        pass

    def predict(self, x: np.array):
        onesArr = np.ones((self.x_train.shape[0],1))
        x_1 = np.concatenate((self.x_train, onesArr), axis=1)
        x_t = x_1.transpose()
        x_t_x = np.dot(x_t,x_1)
        x_t_x_1 = np.linalg.inv(x_t_x)
        x_t_x_1_x_t = np.dot(x_t_x_1,x_t)
        self.coef_ = np.dot(x_t_x_1_x_t,y_train)
        predictions = np.dot(x,self.coef_[:-1]) + self.coef_[-1]
        return predictions
    
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True) #Этот датасет уже предобработан
# Разбиение датасета на тренировочную и тестовую часть
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
# Создание объекта Вашего класса и его обучение
LinReg = LinearRegression()
LinReg.fit(X_train, y_train)
# Прогноз на тестовом датасете
predictions = LinReg.predict(x_test)
print(predictions)
# Оценка качества прогноза

from sklearn.metrics import r2_score

print(r2_score(y_test, predictions))