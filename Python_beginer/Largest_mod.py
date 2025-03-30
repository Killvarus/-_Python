import numpy as np

def largest_mod(x):
    x = np.array(x) 
    Y = np.where(x[:, None] != x, x[:, None] % x, 0)
    return np.max(Y)
x = [3, 7, 9, 15]
print(largest_mod(x))  # Ожидаемый результат 9