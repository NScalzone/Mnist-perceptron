import csv
import numpy as np
from typing import List
import sys
from tqdm import tqdm

ETA = 0.001
epochs = 20
training_data_path = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Homework1/mnist_train_with_bias.csv"
weight_data_path = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Homework1/perceptron_weights.csv"
# weight_data_path = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Homework1/updated_model_weights-40epochs.csv"
save_path = f"/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Homework1/lr001updated_model_weights-{epochs}epochs.csv"


training_data = []
with open(training_data_path) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        training_data.append(row)
        
model_weights = []
with open(weight_data_path) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        model_weights.append(row)

def run_test(test_vals:List[float], weights:List[float])->float:
    if len(test_vals) != len(weights):
        print(f"Array length mismatched, exiting. test_val_length: {len(test_vals)}, weights length: {len(weights)}")
        sys.exit(1)
    test = np.dot(test_vals, weights)
    return test

def evaluate_results(results:List[float])->int:
    max_val = max(results)
    predicted_value = results.index(max_val)
    return predicted_value

def update_weights(weights, inputs, target, results)->List[float]:

    target = int(target)
    for i in range(len(weights)):
        y = 0
        t = 0
        if results[i] > 0:
            y = 1
        if i == target:

            t = 1

        for j in range(len(weights[i])):
            weights[i][j] = update(weight=weights[i][j], y=y, t=t, x=inputs[j])
    return weights

def update(weight:float, y:float, t:float, x:float)->float:
    updated_weight = weight + (ETA * (t - y) * x)
    return updated_weight


epoch = 0
for epoch in tqdm(range(epochs)):
    test_set_size = 60000
    total_correct_runs = 0
    run = 0
    while run < test_set_size:
        
        target_val = (training_data[run][0])
        mnist_val = training_data[run][1:]
        
        run_results = []
        for i in model_weights:
            result = run_test(mnist_val, i)
            run_results.append(result)
        
        predicted_value = evaluate_results(run_results)
        
        if predicted_value != target_val:
            model_weights = update_weights(model_weights, mnist_val, target_val, run_results)

        run += 1
    

with open(save_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(model_weights)
