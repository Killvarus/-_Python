import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('D:/Desktop/Python МФК/TRAIN.csv')
df = df.drop(df.columns[0],axis=1)
# print(df['cut'].unique())
# print(df['color'].unique())
# print(df['clarity'].unique())
# ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')
# ohetransform1 = ohe.fit_transform(df[['cut']])
# df=pd.concat([df,ohetransform1],axis=1).drop(columns=['cut'])
# ohetransform_color = ohe.fit_transform(df[['color']])
# df=pd.concat([df,ohetransform_color],axis=1).drop(columns=['color'])
# ohetransform_clarity = ohe.fit_transform(df[['clarity']])
# df=pd.concat([df,ohetransform_clarity],axis=1).drop(columns=['clarity'])
df = shuffle(df)
# 
le = LabelEncoder()
le.fit(df['cut'])
print(le.classes_)
df=pd.concat([df,pd.Series(le.transform(df['cut']))],axis=1).drop(columns=['cut'])
df = df.rename(columns={0: 'cut'})

le.fit(df['color'])
print(le.classes_)
df=pd.concat([df,pd.Series(le.transform(df['color']))],axis=1).drop(columns=['color'])
df = df.rename(columns={0: 'color'})

le.fit(df['clarity'])
print(le.classes_)
df=pd.concat([df,pd.Series(le.transform(df['clarity']))],axis=1).drop(columns=['clarity'])
df = df.rename(columns={0: 'cut'})

#X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['price']),df['price'],test_size=0.4, random_state=42)
reg = DecisionTreeRegressor(criterion='squared_error',max_depth=12)
scores = cross_val_score(reg,df.drop(columns=['price']), df['price'], cv=10, scoring='r2')
print(np.sum(scores)/10,'criterion=squared_error,max_depth=12')

reg = DecisionTreeRegressor(criterion='friedman_mse',max_depth=16)
scores = cross_val_score(reg,df.drop(columns=['price']), df['price'], cv=10, scoring='r2')
print(np.sum(scores)/10,'criterion=friedman_mse,max_depth=16')

reg = DecisionTreeRegressor(criterion='poisson',max_depth=22)
scores = cross_val_score(reg,df.drop(columns=['price']), df['price'], cv=10, scoring='r2')
print(np.sum(scores)/10,'criterion=poisson,max_depth=22')

reg = DecisionTreeRegressor(criterion='squared_error',max_depth=45)
scores = cross_val_score(reg,df.drop(columns=['price']), df['price'], cv=10, scoring='r2')
print(np.sum(scores)/10,'criterion=squared_error,max_depth=45')

reg = DecisionTreeRegressor(criterion='friedman_mse',max_depth=95)
scores = cross_val_score(reg,df.drop(columns=['price']), df['price'], cv=10, scoring='r2')
print(np.sum(scores)/10,'criterion=friedman_mse,max_depth=95')

reg = DecisionTreeRegressor(criterion='poisson',max_depth=33)
scores = cross_val_score(reg,df.drop(columns=['price']), df['price'], cv=10, scoring='r2')
print(np.sum(scores)/10,'criterion=poisson,max_depth=33')

# print(df.head(10))