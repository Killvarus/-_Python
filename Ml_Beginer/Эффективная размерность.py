import numpy as np
import matplotlib.pyplot as plt
data = np.load('D:/Desktop/Вся прога на питоне/Python МФК/PCA.npy')
u,s,vh = np.linalg.svd(data)
sum = np.sum(s)
scatter = []
sum_1 = []
M=500
for m in range(M):
    sum_1.append(np.sum(s[m+1:]))
    scatter.append(sum_1[m]/sum)
    if(abs(scatter[m])<0.2):
        print(m)
        M = m+1
        break
plt.figure(figsize=(10,6))
plt.plot(range(M),scatter)
plt.show()