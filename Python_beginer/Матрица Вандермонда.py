import numpy as np

def vander(x):
    n = len(x)
    return x[:, None] ** np.arange(n)
x = np.array([1,2,3,4])
print(vander(x))