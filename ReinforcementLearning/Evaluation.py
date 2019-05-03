import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data1 = pd.read_csv("GA_test_p.csv")
data2 = pd.read_csv("GA_test.csv")
steps1 = data1.iloc[:,2:33]
least1 = data1.iloc[:,32]
steps2 = data2.iloc[:,2:33]
least2 = data2.iloc[:,32]
least = []
for k in range(30):
    if least1[k] > least2[k]:
        least.append(least2[k])
    else:
        least.append(least1[k])
minimun = np.array(least)
percent1 = steps1.div(minimun,axis = 0)
percent2 = steps2.div(minimun,axis = 0)
result1 = []
for i in range(percent1.shape[1]):
    product = 1
    for j in range(percent1.shape[0]):
        product = product*percent1.iloc[j,i]
    mean = pow(product, 1/30)
    result1.append(mean)
average1 = np.array(result1)
result2 = []
for i in range(percent2.shape[1]):
    product = 1
    for j in range(percent2.shape[0]):
        product = product*percent2.iloc[j,i]
    mean = pow(product, 1/30)
    result2.append(mean)
average2 = np.array(result2)
plt.plot(average1, label = '2-opt - GA with D')
plt.plot(average2, label = '2-opt - GA')
plt.xlabel('Iteration') 
plt.ylabel('Scaled Distance')
plt.legend(loc=0,ncol=1)
plt.savefig("result.jpg")