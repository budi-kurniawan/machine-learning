from __future__ import division
import numpy as np
import sys
import math

if len(sys.argv) < 4:
    sys.argv = ['', 'X_train.csv', 'y_train.csv', 'X_test.csv']

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

def split_data_by_class(X, y):
    temp = {}
    for i in range(len(y)):
        klazz = int(y[i])
        if klazz not in temp:
            temp[klazz] = []
        temp[klazz].append(X[i])
    ## return a list instead of dict
    temp2 = []
    for i in range(len(temp)):
        temp2.append(temp[i])
    return temp2

def calculate_priors(y, num_classes):
    temp = [0] * num_classes
    for i in range(len(y)):
        temp[int(y[i])] += 1/len(y)
    return temp

def calculate_means_per_feature(X_by_class):
    temp = []
    for i in range(len(X_by_class)):
        class_data = X_by_class[i]
        means = []
        for attribute in zip(*class_data):
            means.append(np.mean(attribute))
        temp.append(means)
    return temp

def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def calculate_variances_per_feature(X_by_class):
    temp = []
    for i in range(len(X_by_class)):
        class_data = X_by_class[i]
        variances = []
        for attribute in zip(*class_data):
            avg = mean(attribute)
            variance = sum([pow(x-avg,2) for x in attribute])/float(len(attribute)-1)
            variances.append(variance)
        temp.append(variances)
    return temp
    
def predict(num_classes, X_test, priors, means_by_class, variances_by_class):
    temp = []
    for i in range(len(X_test)):
        probabilities = []
        test_data = X_test[i]
        for j in range(num_classes):
            means = means_by_class[j]
            x = test_data - means
            variance = np.diag(variances_by_class[j])
            exponent = -0.5 * np.dot(x.T.dot(np.linalg.inv(variance)), x)
            probability = priors[j] / math.sqrt(np.linalg.det(variance)) * math.exp(exponent)
            probabilities.append(probability)
        temp.append(probabilities)
    return temp

X_by_class = split_data_by_class(X_train, y_train)

num_classes = len(X_by_class)
priors_by_class = calculate_priors(y_train, num_classes)
means_per_feature_by_class = calculate_means_per_feature(X_by_class)
variances_per_feature_by_class = calculate_variances_per_feature(X_by_class)

print X_by_class
print "priors", priors_by_class
print "means", means_per_feature_by_class
print "variances", variances_per_feature_by_class

final_outputs = predict(num_classes, X_test, priors_by_class, means_per_feature_by_class, variances_per_feature_by_class) # assuming final_outputs is returned from function
# print final_outputs

# normalize
for i in range(len(final_outputs)):
    row = final_outputs[i]
    total = sum(row)
    for j in range(len(row)):
        row[j] = row[j]/total
        
np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file

# print final_outputs
# for i in range(len(final_outputs)):
#     class_0 = final_outputs[i][0]
#     class_1 = final_outputs[i][1]
#     if class_0 > class_1:
#         print i+1, 0
#     else:
#         print i+1, 1
