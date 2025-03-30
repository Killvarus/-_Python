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
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
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
# import sys
# sys.path.append('D:\Desktop\Вся прога на питоне\Задача по курсовой')
# from functions import ONP
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

    # input_dim = input_dim
    # output_dim = output_dim 
    # learning_rate = learning_rate
    # num_epochs = num_epochs
    # patience = patience

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
    with torch.no_grad():
        predictions = model(X_train_tensor)
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
    for train_index, val_index in kf.split(X_valid):
        X_fold_train, X_fold_val = X_valid.iloc[train_index], X_valid.iloc[val_index]
        y_fold_train, y_fold_val = y_valid.iloc[train_index], y_valid.iloc[val_index]

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
    # st_mean_CV_r2 = np.std(r2_scores)/np.sqrt(len(r2_scores))
    mean_CV_Pear,st_mean_CV_Pear = process_list(pearson_scores,output_dim)
    # st_mean_CV_Pear = np.std(pearson_scores)/np.sqrt(len(pearson_scores))
    mean_CV_MSE,st_mean_CV_MSE = process_list(MSE_scores,output_dim)
    # st_mean_CV_MSE = np.std(MSE_scores)/np.sqrt(len(MSE_scores))
    mean_CV_MAE,st_mean_CV_MAE = process_list(MAE_scores,output_dim)
    # st_mean_CV_MAE = np.std(MAE_scores)/np.sqrt(len(MAE_scores))

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
    # mean_TE_r2 = np.mean(r2_scores)
    #     # st_mean_TE_r2 = np.std(r2_scores)/np.sqrt(len(r2_scores))
    # mean_TE_Pear = np.mean(pearson_scores)
    #     # st_mean_TE_Pear = np.std(pearson_scores)/np.sqrt(len(pearson_scores))
    # mean_TE_MSE = np.mean(MSE_scores)
    #     # st_mean_TE_MSE = np.std(MSE_scores)/np.sqrt(len(MSE_scores)) 
    # mean_TE_MAE = np.mean(MAE_scores)
        # st_mean_TE_MAE = np.std(MAE_scores)/np.sqrt(len(MAE_scores)) 
    df_TE = create_dataframe_score(output_dim,r2_scores,MSE_scores,MAE_scores,pearson_scores)
    return (df_TE,df_CV,df_error_CV)


train = pd.read_csv('D:/Desktop/Вся прога на питоне/Задача по курсовой/mtsgrvmgn_trn.csv')
test = pd.read_csv('D:/Desktop/Вся прога на питоне/Задача по курсовой/mtsgrvmgn_tst.csv')
valid = pd.read_csv('D:/Desktop/Вся прога на питоне/Задача по курсовой/mtsgrvmgn_vld.csv')
y_train = train[['H1_8','H2_8','H3_8']]
y_valid = valid[['H1_8','H2_8','H3_8']]
y_test = test[['H1_8','H2_8','H3_8']]
X_test = test.drop(['H1_8','H2_8','H3_8'], axis=1)
common_columns = X_test.columns.intersection(train.columns)
X_train = train[common_columns]
X_test = X_test[common_columns]
X_valid = valid[common_columns]

TE,CV,CVE = ONP(X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid,X_test=X_test,y_test=y_test,
     batch_size=64,input_dim=X_train.shape[1],output_dim=3,learning_rate=0.001,num_epochs=50,patience=10,graph=False)
print(TE,CV,CVE)