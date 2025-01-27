import csv
import numpy as np
from typing import List
import sys

testing_data_path = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Homework1/mnist_test_with_bias.csv"
weight_data_path = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Homework1/perceptron_weights.csv"
trained_weight_data_path = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Homework1/lr01updated_model_weights-5epochs-updated.csv"

testing_data = []
with open(testing_data_path) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        testing_data.append(row)
        
model_weights = []
with open(trained_weight_data_path) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        model_weights.append(row)

def run_test(test_vals:List[float], weights:List[float])->float:
    if len(test_vals) != len(weights):
        print(f"Array length mismatched, exiting. test_val_length: {len(test_vals)}, weights length: {len(weights)}")
        sys.exit(1)
    test = np.dot(test_vals, weights)
    # result = 0
    # if test > 0:
    #     result = 1
    return test

def evaluate_results(results:List[float])->int:
    max_val = max(results)
    predicted_value = results.index(max_val)
    return predicted_value
    
epochs = 10000
total_correct_runs = 0
run = 0

confusion_matrix = []
for i in range(10):
    temp = []
    confusion_matrix.append(temp)
    for j in range(10):
        confusion_matrix[i].append(0)

while run < epochs:
    
    target_val = (testing_data[run][0])
    mnist_val = testing_data[run][1:]
    
    run_results = []
    for i in model_weights:
        result = run_test(mnist_val, i)
        run_results.append(result)
    
    predicted_value = evaluate_results(run_results)
    
    confusion_matrix[int(target_val)][int(predicted_value)] += 1
    
    if predicted_value == target_val:
        total_correct_runs += 1
        # print(f"predicted value: {predicted_value}\ntarget value: {target_val}")
    run += 1
    
print("============================================================================================")
print(f"Total runs: {run}\nTotal correct predictions: {total_correct_runs}")
print("============================================================================================")
print("Confusion Matrix:")
print("Predicted Values:\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\trecall\n----------------------------------------------------------------------------------------------------")
precision = [0,0,0,0,0,0,0,0,0,0]
for k in range(10):
    print(f"Target value: {k}\t\t", end='')
    sum = 0
    for l in range(10):
        print(f"{confusion_matrix[k][l]}\t", end='')
        sum += confusion_matrix[k][l]
        precision[l] += confusion_matrix[k][l]
    recall = int(100 * ((confusion_matrix[k][k])/sum))
    print(recall)
    print('----------------------------------------------------------------------------------------------------')

print("Precision:\t\t",end='')
for m in range(10):
    precison_value = int(100*((confusion_matrix[m][m])/(precision[m])))
    print(f"{precison_value}\t",end='')
    
print('\n----------------------------------------------------------------------------------------------------')