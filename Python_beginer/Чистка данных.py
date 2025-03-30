import numpy as np
def clear(X):
    X=X[np.any(np.isnan(X),axis=1)==False] 
    return(X)
X = np.array([
    [0,         1,          2], 
    [1,         np.nan,     2], 
    [np.nan,    np.nan,     np.nan], 
    [-1,        -2,         -3]
])
print(clear(X)) # [[0, 1, 2], [-1, -2, -3]]