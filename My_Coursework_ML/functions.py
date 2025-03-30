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

def process_list(original_list, n):
    averages = []  # Список для средних значений
    std_devs = []  # Список для стандартных отклонений

    # Проходим по списку с шагом n
    for i in range(n):
        # Берем подсписок из n элементов
        sublist = original_list[i::n]
        
        # Вычисляем среднее и стандартное отклонение
        avg = np.mean(sublist)
        std_dev = np.std(sublist)/np.sqrt(len(sublist))
        
        # Добавляем результаты в соответствующие списки
        averages.append(avg)
        std_devs.append(std_dev)
    
    return averages, std_devs

def ONP (X_train,y_train,X_valid,y_valid,X_test,y_test,batch_size, 
         input_dim,output_dim,learning_rate,num_epochs,patience,graph=False):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_valid.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    batch_size = batch_size
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class SingleLayerPerceptron(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SingleLayerPerceptron, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.fc(x)

    input_dim = input_dim
    output_dim = output_dim 
    learning_rate = learning_rate
    num_epochs = num_epochs
    patience = patience

    model = SingleLayerPerceptron(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    counter = 0  

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for X_batch, y_batch in dataloader:
            # Прямой проход
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_dataloader:
                val_outputs = model(X_val_batch)
                val_loss += criterion(val_outputs, y_val_batch).item()

        val_loss /= len(val_dataloader)
        train_loss_history.append(epoch_loss / len(dataloader))
        val_loss_history.append(val_loss)

        # Логика Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Сброс счетчика, так как ошибка уменьшилась
        else:
            counter += 1  # Увеличиваем счетчик, если улучшения нет
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break  # Прерываем цикл обучения

        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}")

    model.eval()
    # with torch.no_grad():
    #     predictions = model(X_train_tensor)
    if graph:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss', marker='o')
        plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', marker='s')
        plt.title('Зависимость ошибки от количества эпох', fontsize=16)
        plt.xlabel('Эпохи', fontsize=14)
        plt.ylabel('Ошибка (Loss)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

    kf = KFold(n_splits=5)
    pearson_scores = []
    r2_scores = []
    MSE_scores = []
    MAE_scores =[]
    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Преобразуем в тензоры
        X_fold_train_tensor = torch.tensor(X_fold_train.values, dtype=torch.float32)
        y_fold_train_tensor = torch.tensor(y_fold_train.values, dtype=torch.float32)
        X_fold_val_tensor = torch.tensor(X_fold_val.values, dtype=torch.float32)
        y_fold_val_tensor = torch.tensor(y_fold_val.values, dtype=torch.float32)

        fold_dataset = TensorDataset(X_fold_train_tensor, y_fold_train_tensor)
        fold_dataloader = DataLoader(fold_dataset, batch_size=batch_size, shuffle=True)

        # Инициализируем модель заново
        model = SingleLayerPerceptron(input_dim, output_dim)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Обучаем модель на fold-е
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in fold_dataloader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Оцениваем на валидации
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_fold_val_tensor).numpy()
            for i in range(output_dim):
                corr_pear, _ = pearsonr(val_outputs[:, i], y_fold_val_tensor[:, i].numpy())
                pearson_scores.append(corr_pear)
                corr_r2 = r2_score(val_outputs[:, i], y_fold_val_tensor[:, i].numpy())
                r2_scores.append(corr_r2)
                corr_MSE = mean_squared_error(val_outputs[:, i], y_fold_val_tensor[:, i].numpy())
                MSE_scores.append(corr_MSE)
                corr_MAE = mean_absolute_error(val_outputs[:, i], y_fold_val_tensor[:, i].numpy())
                MAE_scores.append(corr_MAE)                
    mean_CV_r2,st_mean_CV_r2 = process_list(r2_scores,output_dim)
    mean_CV_Pear,st_mean_CV_Pear = process_list(pearson_scores,output_dim)
    mean_CV_MSE,st_mean_CV_MSE = process_list(MSE_scores,output_dim)
    mean_CV_MAE,st_mean_CV_MAE = process_list(MAE_scores,output_dim)

    df_CV = create_dataframe_score(output_dim,mean_CV_r2,mean_CV_MSE,mean_CV_MAE,mean_CV_Pear)
    df_error_CV = create_dataframe_error(output_dim,st_mean_CV_r2,st_mean_CV_MSE,st_mean_CV_MAE,st_mean_CV_Pear)
    pearson_scores = []
    r2_scores = []
    MSE_scores = []
    MAE_scores = []
    with torch.no_grad():
        test_outputs = model(X_test_tensor).numpy()
        for i in range(output_dim):
            corr_pear, _ = pearsonr(test_outputs[:, i], y_test_tensor[:, i].numpy())
            pearson_scores.append(corr_pear)
            corr_r2 = r2_score(test_outputs[:, i], y_test_tensor[:, i].numpy())
            r2_scores.append(corr_r2)
            corr_MSE = mean_squared_error(test_outputs[:, i], y_test_tensor[:, i].numpy())
            MSE_scores.append(corr_MSE)
            corr_MAE = mean_absolute_error(test_outputs[:, i], y_test_tensor[:, i].numpy())
            MAE_scores.append(corr_MAE)
    df_TE = create_dataframe_score(output_dim,r2_scores,MSE_scores,MAE_scores,pearson_scores)
    return (df_TE,df_CV,df_error_CV)

def ONP_Adagrad (X_train,y_train,X_valid,y_valid,X_test,y_test,batch_size, 
         input_dim,output_dim,learning_rate,num_epochs,patience,graph=False):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_valid.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    batch_size = batch_size
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class SingleLayerPerceptron(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SingleLayerPerceptron, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.fc(x)

    input_dim = input_dim
    output_dim = output_dim 
    learning_rate = learning_rate
    num_epochs = num_epochs
    patience = patience

    model = SingleLayerPerceptron(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    counter = 0  

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for X_batch, y_batch in dataloader:
            # Прямой проход
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_dataloader:
                val_outputs = model(X_val_batch)
                val_loss += criterion(val_outputs, y_val_batch).item()

        val_loss /= len(val_dataloader)
        train_loss_history.append(epoch_loss / len(dataloader))
        val_loss_history.append(val_loss)

        # Логика Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Сброс счетчика, так как ошибка уменьшилась
        else:
            counter += 1  # Увеличиваем счетчик, если улучшения нет
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break  # Прерываем цикл обучения

        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}")

    model.eval()
    if graph:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss', marker='o')
        plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', marker='s')
        plt.title('Зависимость ошибки от количества эпох', fontsize=16)
        plt.xlabel('Эпохи', fontsize=14)
        plt.ylabel('Ошибка (Loss)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

    kf = KFold(n_splits=5)
    pearson_scores = []
    r2_scores = []
    MSE_scores = []
    MAE_scores =[]
    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Преобразуем в тензоры
        X_fold_train_tensor = torch.tensor(X_fold_train.values, dtype=torch.float32)
        y_fold_train_tensor = torch.tensor(y_fold_train.values, dtype=torch.float32)
        X_fold_val_tensor = torch.tensor(X_fold_val.values, dtype=torch.float32)
        y_fold_val_tensor = torch.tensor(y_fold_val.values, dtype=torch.float32)

        fold_dataset = TensorDataset(X_fold_train_tensor, y_fold_train_tensor)
        fold_dataloader = DataLoader(fold_dataset, batch_size=batch_size, shuffle=True)

        # Инициализируем модель заново
        model = SingleLayerPerceptron(input_dim, output_dim)
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

        # Обучаем модель на fold-е
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in fold_dataloader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Оцениваем на валидации
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_fold_val_tensor).numpy()
            for i in range(output_dim):
                corr_pear, _ = pearsonr(val_outputs[:, i], y_fold_val_tensor[:, i].numpy())
                pearson_scores.append(corr_pear)
                corr_r2 = r2_score(val_outputs[:, i], y_fold_val_tensor[:, i].numpy())
                r2_scores.append(corr_r2)
                corr_MSE = mean_squared_error(val_outputs[:, i], y_fold_val_tensor[:, i].numpy())
                MSE_scores.append(corr_MSE)
                corr_MAE = mean_absolute_error(val_outputs[:, i], y_fold_val_tensor[:, i].numpy())
                MAE_scores.append(corr_MAE)                
    mean_CV_r2,st_mean_CV_r2 = process_list(r2_scores,output_dim)
    mean_CV_Pear,st_mean_CV_Pear = process_list(pearson_scores,output_dim)
    mean_CV_MSE,st_mean_CV_MSE = process_list(MSE_scores,output_dim)
    mean_CV_MAE,st_mean_CV_MAE = process_list(MAE_scores,output_dim)

    # data_CV = {
    #     'Мера оценки': ['r2', 'MSE', 'MAE','Pearson'],
    #     'Оценка': [mean_CV_r2,mean_CV_MSE,mean_CV_MAE,mean_CV_Pear],
    #     'Отклонение': [st_mean_CV_r2,st_mean_CV_MSE,st_mean_CV_MAE, st_mean_CV_Pear]
    # }
    df_CV = create_dataframe_score(output_dim,mean_CV_r2,mean_CV_MSE,mean_CV_MAE,mean_CV_Pear)
    df_error_CV = create_dataframe_error(output_dim,st_mean_CV_r2,st_mean_CV_MSE,st_mean_CV_MAE,st_mean_CV_Pear)
    pearson_scores = []
    r2_scores = []
    MSE_scores = []
    MAE_scores = []
    with torch.no_grad():
        test_outputs = model(X_test_tensor).numpy()
        for i in range(output_dim):
            corr_pear, _ = pearsonr(test_outputs[:, i], y_test_tensor[:, i].numpy())
            pearson_scores.append(corr_pear)
            corr_r2 = r2_score(test_outputs[:, i], y_test_tensor[:, i].numpy())
            r2_scores.append(corr_r2)
            corr_MSE = mean_squared_error(test_outputs[:, i], y_test_tensor[:, i].numpy())
            MSE_scores.append(corr_MSE)
            corr_MAE = mean_absolute_error(test_outputs[:, i], y_test_tensor[:, i].numpy())
            MAE_scores.append(corr_MAE)
    df_TE = create_dataframe_score(output_dim,r2_scores,MSE_scores,MAE_scores,pearson_scores)
    return (df_TE,df_CV,df_error_CV)

def to_exel (file_name,n_coloumns, n_iter,X_train,y_train,X_valid,y_valid,X_test,y_test,batch_size, 
         input_dim,output_dim,learning_rate,num_epochs,patience,graph=False):
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        for i in range(n_iter):
                    TE,CV,CVE = ONP(X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid,X_test=X_test,y_test=y_test,
                    batch_size=batch_size,input_dim=input_dim,output_dim=output_dim,
                    learning_rate=learning_rate,num_epochs=num_epochs,patience=patience,graph=graph)
                    if i == 0:
                        writer.book.create_sheet('TE')
                        TE.to_excel(writer, sheet_name='TE', index=False, startrow=0, startcol=i * n_coloumns)
                    else:
                        TE.to_excel(writer, sheet_name='TE', index=False, startrow=0, startcol=i * n_coloumns + 1)
                    # Записываем CV
                    if i == 0:
                        writer.book.create_sheet('CV')
                        CV.to_excel(writer, sheet_name='CV', index=False, startrow=0, startcol=i * n_coloumns)
                    else:
                        CV.to_excel(writer, sheet_name='CV', index=False, startrow=0, startcol=i * n_coloumns + 1)
                    # Записываем CVE
                    if i == 0:
                        writer.book.create_sheet('CVE')
                        CVE.to_excel(writer, sheet_name='CVE', index=False, startrow=0, startcol=i * n_coloumns)
                    else:
                        CVE.to_excel(writer, sheet_name='CVE', index=False, startrow=0, startcol=i * n_coloumns + 1)


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

def GBT (X_train,y_train,X_valid,y_valid,X_test,y_test, n_estimators,max_depth,subsample,output_dim=1,seed=42,cv = 5):
    xgb_reg_0 = XGBRegressor(n_estimators=n_estimators,max_depth=max_depth,seed = seed,subsample = subsample)
    xgb_reg_0.fit(X_train, y_train)
    y_valid = y_valid.iloc[:, 0].to_numpy()
    y_test = y_test.iloc[:, 0].to_numpy()
    pred1 = xgb_reg_0.predict(X_test)
    corr_pear, _ = pearsonr(pred1, y_test)
    corr_r2 = r2_score(pred1, y_test)
    corr_MSE = mean_squared_error(pred1, y_test)
    corr_MAE = mean_absolute_error(pred1, y_test)
    scoring_metrics = ['r2', 'neg_mean_squared_error',
                       'neg_mean_absolute_error']
    Cv_list = []
    Cv_list.append(cross_val_score(estimator=xgb_reg_0,X=X_valid,y=y_valid,n_jobs=-1,cv = cv, scoring=pearson_corr))

    for i in scoring_metrics:
        Cv_list.append(cross_val_score(estimator=xgb_reg_0,X=X_valid,y=y_valid,n_jobs=-1,cv = cv, scoring=i))
    df_CV = create_dataframe_score_GB(np.mean(Cv_list[1]),np.mean(Cv_list[2]),
                                   np.mean(Cv_list[3]),np.mean(Cv_list[0]))
    df_CVE = create_dataframe_error_GB(np.std(Cv_list[1]),np.std(Cv_list[2]),
                                   np.std(Cv_list[3]),np.std(Cv_list[0]))
    df_TE = create_dataframe_score_GB(corr_r2,corr_MSE,corr_MAE,corr_pear)
    return (df_TE,df_CV,df_CVE)

def to_exel_GB(file_name,n_coloumns, n_iter,X_train,y_train,X_valid,
               y_valid,X_test,y_test, n_estimators,max_depth,subsample,output_dim=1,cv = 5):
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        for i in range(n_iter):
                    TE,CV,CVE = GBT(X_train=X_train,y_train=y_train,
                                    X_valid=X_valid,y_valid=y_valid,
                                    X_test=X_test,y_test=y_test,
                                    n_estimators=n_estimators,max_depth=max_depth,
                                    output_dim=output_dim,seed=i,cv = cv,subsample = subsample)
                    if i == 0:
                        writer.book.create_sheet('TE')
                        TE.to_excel(writer, sheet_name='TE', index=False, startrow=0, startcol=i * n_coloumns)
                    else:
                        TE.to_excel(writer, sheet_name='TE', index=False, startrow=0, startcol=i * n_coloumns + 1)
                    # Записываем CV
                    if i == 0:
                        writer.book.create_sheet('CV')
                        CV.to_excel(writer, sheet_name='CV', index=False, startrow=0, startcol=i * n_coloumns)
                    else:
                        CV.to_excel(writer, sheet_name='CV', index=False, startrow=0, startcol=i * n_coloumns + 1)
                    # Записываем CVE
                    if i == 0:
                        writer.book.create_sheet('CVE')
                        CVE.to_excel(writer, sheet_name='CVE', index=False, startrow=0, startcol=i * n_coloumns)
                    else:
                        CVE.to_excel(writer, sheet_name='CVE', index=False, startrow=0, startcol=i * n_coloumns + 1)

def train_test_val(train,test,valid,columns):
    y_train = train[columns]
    y_valid = valid[columns]
    y_test = test[columns]
    X_test = test.drop(columns, axis=1)
    common_columns = X_test.columns.intersection(train.columns)
    return (train[common_columns],X_test[common_columns],valid[common_columns],y_train,y_test,y_valid)

def NN_weight (X_train,y_train,X_valid,y_valid,batch_size, 
         input_dim,output_dim,learning_rate,num_epochs,patience,graph=False):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_valid.values, dtype=torch.float32)

    batch_size = batch_size
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class SingleLayerPerceptron(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SingleLayerPerceptron, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.fc(x)

    input_dim = X_train.shape[1]
    output_dim = output_dim
    learning_rate = learning_rate
    num_epochs = num_epochs
    patience = patience

    model = SingleLayerPerceptron(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    counter = 0  

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for X_batch, y_batch in dataloader:
            # Прямой проход
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_dataloader:
                val_outputs = model(X_val_batch)
                val_loss += criterion(val_outputs, y_val_batch).item()

        val_loss /= len(val_dataloader)
        train_loss_history.append(epoch_loss / len(dataloader))
        val_loss_history.append(val_loss)

        # Логика Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Сброс счетчика, так как ошибка уменьшилась
        else:
            counter += 1  # Увеличиваем счетчик, если улучшения нет
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break  # Прерываем цикл обучения

        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}")

    model.eval()

    # После обучения модели
    model.eval()

    # Извлечение весов и смещений
    weights = model.fc.weight.data

    # Преобразование в numpy (опционально)
    weights_np = weights.numpy()

    # Анализ важности признаков
    feature_importance = np.sum(np.abs(weights_np), axis=0)
    if graph:
        plt.figure(figsize=(10, 6))
        plt.bar(range(input_dim), feature_importance)
        plt.xlabel("Номер признака")
        plt.ylabel("Важность признака")
        plt.title("Важность признаков на основе весов модели")
        plt.show()
    # Извлечение весов из модели
    weights = model.fc.weight.data.numpy()  # Размерность: (output_dim, input_dim)

    # Вычисление важности признаков (сумма абсолютных значений весов по всем выходным нейронам)
    feature_importance = np.sum(np.abs(weights), axis=0)

    # Сортировка признаков по убыванию важности
    sorted_indices = np.argsort(feature_importance)[::-1]

    # Выбор топ-750 признаков
    top_750_indices = sorted_indices[:750]
    X_train_top_750 = X_train.iloc[:, top_750_indices]
    return(X_train_top_750,feature_importance)



def get_top_features(X_train, y_train, X_valid, y_valid, 
                           batch_size, input_dim, output_dim, 
                           learning_rate, num_epochs, patience, graph=False,
                           runs=5,n_features = 750, top_n=750):
    # Список для хранения весов всех запусков
    all_weights = []
    
    # Запускаем модель 5 раз
    for _ in range(runs):
        t,weights = NN_weight(X_train, y_train, X_valid, y_valid, 
                          batch_size, input_dim, output_dim, 
                          learning_rate, num_epochs, patience, graph=False)
        all_weights.append(weights)
    
    # Усредняем веса (если размеры совпадают)
    averaged_weights = np.mean(all_weights, axis=0)
    
    # Выбираем топ-N признаков с наибольшими средними весами
    top_indices = np.argsort(-np.abs(averaged_weights))[:top_n]  # Сортируем по убыванию модуля веса
    top_weights = averaged_weights[top_indices]
    
    # Визуализация
    plt.figure(figsize=(15, 6))
    plt.bar(range(top_n), top_weights)
    plt.xlabel("Номер признака (топ-750)")
    plt.ylabel("Усредненный вес")
    plt.title(f"Топ-{top_n} признаков по усредненным весам (по {runs} запускам)")
    plt.axhline(y=np.mean(top_weights), color='r', linestyle='--', 
                label=f"Средний вес: {np.mean(top_weights):.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return averaged_weights, top_indices


def get_discrete_selected_features(X_train, y_train, X_valid, y_valid, 
                           batch_size, input_dim, output_dim, 
                           learning_rate, num_epochs, patience, graph=False,
                           runs=5,n_features = 750, top_n=750,threshold=3):
    # Словарь для подсчета, сколько раз каждый признак попадал в топ
    feature_counts = {}

    for _ in range(runs):
        # Получаем веса для текущего запуска
        t,weights = NN_weight(X_train, y_train, X_valid, y_valid,
                          batch_size, input_dim, output_dim,
                          learning_rate, num_epochs, patience, graph=False)
        
        # Выбираем топ-N признаков с наибольшими абсолютными весами
        top_indices = np.argsort(-np.abs(weights))[:top_n]
        
        # Увеличиваем счетчик для каждого отобранного признака
        for idx in top_indices:
            if idx in feature_counts:
                feature_counts[idx] += 1
            else:
                feature_counts[idx] = 1
    
    # Преобразуем в массив и сортируем
    features = np.array(list(feature_counts.keys()))
    counts = np.array(list(feature_counts.values()))
    
    # Выбираем признаки, которые попали в топ >= threshold раз
    selected_mask = counts >= threshold
    selected_features = features[selected_mask]
    selected_counts = counts[selected_mask]
    
    # Сортируем по частоте попадания (для наглядности)
    sort_order = np.argsort(-selected_counts)
    selected_features = selected_features[sort_order]
    selected_counts = selected_counts[sort_order]
    
    # Визуализация
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(selected_features)), selected_counts)
    plt.axhline(y=threshold, color='r', linestyle='--',
               label=f'Порог отбора: {threshold}')
    plt.xlabel("Номер признака")
    plt.ylabel("Количество попаданий в топ-750")
    plt.title(f"Дискретный отбор признаков (выбрано {len(selected_features)} признаков)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Итоговое число отобранных признаков: {len(selected_features)}")
    print(f"Индексы отобранных признаков: {selected_features}")
    
    return selected_features
