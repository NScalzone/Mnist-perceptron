import matplotlib.pyplot as plt
import numpy as np

epochs = np.array([0,1,2,3,4,5,10])
testdata_results_001 = np.array([11.35,81.35,83.74,84.15,84.72,84.77,85.85])
testdata_results_01 = np.array([11.35,84.03,84.34,84.16,81.95,83.67,85.30])
testdata_results_1 = np.array([11.35,78.59,80.98,80.47,82.41,85.02,82.78])

# traindata_results_001 = np.array([7080,48243,49843,50380,50611,50739,51366])
# traindata_results_01 = np.array([7080,50582,50668,50843,49692,50549,51656])
# traindata_results_1 = np.array([7080,47200,48918,48278,49560,51273,49719])
traindata_results_001 = np.array([11.8,80.41,83.07,83.96,84.35,84.56,85.61])
traindata_results_01 = np.array([11.8,84.30,84.45,84.74,82.82,84.25,86.09])
traindata_results_1 = np.array([11.8,78.67,81.53,80.46,82.6,85.46,82.87])


plt.plot(epochs, traindata_results_1, 'o')
plt.title("Training data, learning rate 0.1")
plt.xlabel("Epochs")
plt.ylabel("Percent correct")
plt.show()