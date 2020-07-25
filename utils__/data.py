import numpy as np
import scipy.io as scio
import matplotlib as mpl
import matplotlib.pyplot as plt

Theta = 91
Lamda = int((700-400)/20)
d1 = 4
d2 = 4
d3 = 4
n2 = 10
n3 = 10
n4 = 10
Parameters_nums = Lamda*d1*d2*d3*n2*n3*n4
index_hang = 0
Parameters = np.zeros((Parameters_nums, 7))

for a in np.arange(400, 700, 20):
    for b in np.arange(1, 5):
        for c in np.arange(1, 5):
            for d in np.arange(1, 5):
                for e in np.arange(1, 2, 0.1):
                    for f in np.arange(1, 2, 0.1):
                        for g in np.arange(1, 2, 0.1):
                            Parameters[index_hang, 0] = a
                            Parameters[index_hang, 1] = b
                            Parameters[index_hang, 2] = c
                            Parameters[index_hang, 3] = d
                            Parameters[index_hang, 4] = e
                            Parameters[index_hang, 5] = f
                            Parameters[index_hang, 6] = g
                            index_hang = index_hang+1

FileIn = open('outtest.dat', mode='r',  encoding="ascii")
count = 0
Reflects = []
for line in FileIn:
    temp = []
    print(count)
    count = count+1
    line = line[1:-2].split(',')
    for i in line:
        temp.append(float(i))
    Reflects.append(temp)
FileIn.close()
scio.savemat('testP.mat', {'Parameters': Parameters, 'Reflects': Reflects})
