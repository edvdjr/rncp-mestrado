import matplotlib.pyplot as plt
import numpy as np

out_path = "Out/"
y_0 = open(out_path + "0_weights.csv").readlines()
y_0 = y_0[0].split(',')
y_0.pop()
y_0 = [float(i) for i in y_0]
x_0 = [i for i in range(len(y_0))]

q25, q75 = np.percentile(y_0, [25, 75])
bin_width = 2 * (q75 - q25) * len(y_0) ** (-1/3)
bins = round((y_0.max() - y_0.min()) / bin_width)
print("Freedman–Diaconis number of bins:", bins)
plt.hist(y_0, bins=bins);

# plt.ylabel('Probability')
# plt.xlabel('Data');