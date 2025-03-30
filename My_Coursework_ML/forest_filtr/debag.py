from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import xgboost
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch 
import torch.optim as optim
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,Dataset
from sklearn.model_selection import KFold
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch 
import torch.optim as optim
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

def create_dataframe_score(n, r2_scores, mse_scores, mae_scores, pearson_scores):
    # Создание словаря с данными
    data = {
        'Мера оценки': ['R2', 'MSE', 'MAE', 'Pearson'],
        **{f'Оценка{i+1}': [r2_scores[i], mse_scores[i], mae_scores[i], pearson_scores[i]] for i in range(n)}
    }
    
    # Создание DataFrame
    df = pd.DataFrame(data)
    
    # Транспонируем DataFrame, чтобы меры оценки стали столбцами
    df = df.set_index('Мера оценки').T
    return df

def create_dataframe_error(n, r2_scores, mse_scores, mae_scores, pearson_scores):
    # Создание словаря с данными
    data = {
        'Мера оценки': ['R2', 'MSE', 'MAE', 'Pearson'],
        **{f'Ошибка{i+1}': [r2_scores[i], mse_scores[i], mae_scores[i], pearson_scores[i]] for i in range(n)}
    }
    
    # Создание DataFrame
    df = pd.DataFrame(data)
    
    # Транспонируем DataFrame, чтобы меры оценки стали столбцами
    df = df.set_index('Мера оценки').T
    return df
def create_dataframe_score_GB(r2_scores, mse_scores, mae_scores, pearson_scores):
    # Создание словаря с данными
    data = {
        'Мера оценки': ['R2', 'MSE', 'MAE', 'Pearson'],
            'Оценка 1': [r2_scores, mse_scores, mae_scores, pearson_scores]
    }
    
    # Создание DataFrame
    df = pd.DataFrame(data)
    
    # Транспонируем DataFrame, чтобы меры оценки стали столбцами
    df = df.set_index('Мера оценки').T
    return df

def create_dataframe_error_GB(r2_scores, mse_scores, mae_scores, pearson_scores):
    # Создание словаря с данными
    data = {
        'Мера оценки': ['R2', 'MSE', 'MAE', 'Pearson'],
        'Ошибка 1': [r2_scores, mse_scores, mae_scores, pearson_scores]
    }
    
    # Создание DataFrame
    df = pd.DataFrame(data)
    
    # Транспонируем DataFrame, чтобы меры оценки стали столбцами
    df = df.set_index('Мера оценки').T
    return df
def pearson_corr(estimator,X, y_true):
    y_pred = estimator.predict(X)
    corr_pear, _ = pearsonr(y_pred, y_true)
    return (corr_pear)
def GBT (X_train,y_train,X_valid,y_valid,X_test,y_test, n_estimators,max_depth,output_dim=1,seed=42,cv = 5):
    xgb_reg_0 = XGBRegressor(n_estimators=n_estimators,max_depth=max_depth,seed = seed)
    xgb_reg_0.fit(X_train, y_train)
    y_valid = y_valid.iloc[:, 0].to_numpy()
    pred1 = xgb_reg_0.predict(X_test)
    corr_pear, _ = pearsonr(pred1, y_test)
    corr_r2 = r2_score(pred1, y_test)
    corr_MSE = mean_squared_error(pred1, y_test)
    corr_MAE = mean_absolute_error(pred1, y_test)
    scoring_metrics = ['r2', 'neg_mean_squared_error'
                       ,'neg_mean_absolute_error']
    pearson_scorer = make_scorer(pearson_corr)
    Cv_list = []
    y_valid = y_valid[[0]].to_numpy()
    Cv_list.append(cross_val_score(estimator=xgb_reg_0,X=X_valid,y=y_valid,n_jobs=-1,cv = cv, scoring=pearson_corr))

    for i in scoring_metrics:
        Cv_list.append(cross_val_score(estimator=xgb_reg_0,X=X_valid,y=y_valid,n_jobs=-1,cv = cv, scoring=i))
    df_CV = create_dataframe_score_GB(np.mean(Cv_list[1]),np.mean(Cv_list[2]),
                                   np.mean(Cv_list[3]),np.mean(Cv_list[0]))
    df_CVE = create_dataframe_error_GB(np.std(Cv_list[1]),np.std(Cv_list[2]),
                                   np.std(Cv_list[3]),np.std(Cv_list[0]))
    # df_TE = create_dataframe_score(output_dim,corr_r2,corr_MSE,corr_MAE,corr_pear)
    return (df_CV,df_CVE)

train = pd.read_csv('D:/Desktop/Вся прога на питоне/Задача по курсовой/mtsgrvmgn_trn.csv')
test = pd.read_csv('D:/Desktop/Вся прога на питоне/Задача по курсовой/mtsgrvmgn_tst.csv')
valid = pd.read_csv('D:/Desktop/Вся прога на питоне/Задача по курсовой/mtsgrvmgn_vld.csv')
y_train = train[['H1_8']]
y_valid = valid[['H1_8']]
y_test = test[['H1_8']]
X_test = test.drop(['H1_8'], axis=1)
common_columns = X_test.columns.intersection(train.columns)
X_train = train[common_columns]
X_test = X_test[common_columns]
X_valid = valid[common_columns]

print(GBT(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
            X_valid=X_valid,y_valid=y_valid,n_estimators=256,max_depth=5))
