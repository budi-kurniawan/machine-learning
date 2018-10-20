import numpy as np
import sys

if len(sys.argv) < 6:
    sys.argv = ['', '2', '3', 'X_train.csv', 'y_train.csv', 'X_test.csv']
lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def part1():
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    ## From Lecture3.pdf page 16: wRR = (lambda.I + Xt.X)^{-1}.Xt.y
    X = X_train
    y = y_train
    I = np.eye(X.shape[1])
    Xt = X.T
    ## temp = lambda.I + Xt.X
    temp = lambda_input * I + Xt.dot(X)
    return np.linalg.inv(temp).dot(Xt.dot(y))

wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file

## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    ## From Lecture5.pdf (page 9)
    ## covar = (lambda.I + sigma_input^{-2}Xt.X)^{-1}
    ## sigma_0^2 = sigma2_input + x_0.T.dot(covar.dot(x_0))
    X = X_train
    Xt = X.T
    ret_value = []
    temp = []
    for i in range(1):
        I = np.eye(X.shape[1])
        Xt = X.T
        covar = np.linalg.inv(lambda_input * I + 1/sigma2_input * Xt.dot(X))
        for j in range(len(X_test)):
            x0 = X_test[j]
            sigma_02 = sigma2_input + x0.T.dot(covar.dot(x0))
            temp.append(sigma_02)
            #print x0.T.dot(covar.dot(x0))
            #print covar
    for i in range(10):
        max_value = max(temp)
        max_index = temp.index(max_value)
        ret_value.append(1 + max_index)
        temp[max_index] = 0
    return ret_value

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file