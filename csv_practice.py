import csv
import random
import numpy as np

data_file = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Homework1/test_csv.csv"
training_data_path = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Homework1/mnist_train.csv"
test_data_path = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Homework1/mnist_test.csv"
save_path = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/Homework1/mnist_test_with_bias.csv"

# results = []
# with open(test_data_path) as csvfile:
#     reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
#     for row in reader: # each row is a list
#         results.append(row)

# for i in results:
#     i.insert(1,1)
#     for j in range(2,786):
#         i[j] = i[j]/255

# model_weights = []
# for i in range(10):
#     temp = []
#     for j in range(785):
#         temp.append(random.uniform(-0.5, 0.5))
#     model_weights.append(temp)

# print(model_weights)

# with open(save_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(results)


test1 = [1,2,3]
test2 = [4,5,6]

dot = np.dot(test1, test2)
print(dot)