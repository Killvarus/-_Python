import numpy as np

def mean_and_std(X):
    mean = np.sum(X, axis=0) / X.shape[0]
    std = np.sqrt(np.sum((X - mean) ** 2, axis=0) / X.shape[0])
    
    return mean, std
X = np.array([[1,2,3],[4,5,6]])
print(mean_and_std(X))

def procces(X):
    mean, std = mean_and_std(X)
    return (X - mean) / std

def scale(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

def normalize(X):
    return X/np.linalg.norm(X, axis=0, keepdims=True)

X = np.array([[1,2,3],[4,5,6]])
print(mean_and_std(X))
print(np.std(X,axis=0))
print(procces(X))
print(scale(X))
print(normalize(X))