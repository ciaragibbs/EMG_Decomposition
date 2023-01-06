import sys

import matplotlib.pyplot as plt
import numpy as np
import csv


emg = np.load('EMG_files/short_7_slow_4.npy')
print('done')




"""csv.field_size_limit(sys.maxsize)
list_emg = []
with open('EMG_files/1_5_fast.csv', newline='') as csvfile:
    emg = csv.reader(csvfile, delimiter=',')
    for row in emg:
        list_emg.append([float(element) for element in row])

plt.plot(list_emg[160])
plt.show()"""