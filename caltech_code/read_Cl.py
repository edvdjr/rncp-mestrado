import matplotlib.pyplot as plt
import numpy as np

out_path = "Out/"
y_0 = open(out_path + "0_Cl.csv").readlines()
y_0 = y_0[0].split(',')
y_0.pop()
y_0 = [float(i) for i in y_0]
x_0 = [i for i in range(len(y_0))]

y_1 = open(out_path + "1_Cl.csv").readlines()
y_1 = y_1[0].split(',')
y_1.pop()
y_1 = [float(i) for i in y_1]
x_1 = [i for i in range(len(y_1))]

y_2 = open(out_path + "2_Cl.csv").readlines()
y_2 = y_2[0].split(',')
y_2.pop()
y_2 = [float(i) for i in y_2]
x_2 = [i for i in range(len(y_2))]

plt.xlabel("Iterações por Camada")
plt.ylabel("Cl")

miny, maxy = min(min(y_0), min(y_1), min(y_2)), max(max(y_0), max(y_1), max(y_2))
plt.yticks(np.arange(miny, maxy, (maxy - miny)/10))
plt.plot(x_0, y_0, label='Conv 0')
plt.plot(x_1, y_1, label='Conv 1')
plt.plot(x_2, y_2, label='Conv 2')
plt.title('Índice de Convergência')
plt.legend()
plt.savefig(out_path + 'Cl.png')
plt.close()