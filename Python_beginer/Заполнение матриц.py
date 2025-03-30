import numpy as np

def cross(n, m):
    matrix = np.ones((n, n))
    matrix[:m, :m] = 0
    matrix[:m, -m:] = 0
    matrix[-m:, :m] = 0
    matrix[-m:, -m:] = 0
    return matrix

def square(n, m):
    matrix = np.zeros((n, n), dtype=int)
    matrix[m:n-m, m:n-m] = 1
    return matrix
import numpy as np

def chess_board(n):
    matrix = np.zeros((n, n), dtype=int)
    matrix[::2, ::2] = 1 
    matrix[1::2, 1::2] = 1 
    return matrix

print(cross(8,4))
print(square(6,2))
print(chess_board(6))