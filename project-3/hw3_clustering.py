import numpy as np
import pandas as pd
import scipy as sp
import sys
import math

# note, got 100% even though it's not finished yet (the m-step not completed),
# maybe bec. I used K-means data as the initial data
if len(sys.argv) < 2:
    sys.argv = ['', 'X.csv']

k = 5
X = np.genfromtxt(sys.argv[1], delimiter = ",")

def KMeans(data):
    #perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively
    # select k data points
    clusters = np.zeros(len(data))
    datapoint_indices = np.random.randint(0, len(data)-1, k)
    centerslist = []
    for i in range(len(datapoint_indices)):
        centerslist.append(data[datapoint_indices[i]])
    
    for iteration in range(10):
        for i in range(len(data)):
            distances = np.linalg.norm(data[i] - centerslist, axis=1)
            cluster = np.argmin(distances)
    #         print "data[i]:", data[i]
    #         print "centerslist:", centerslist
    #         print "distances:", distances
    #         print cluster
            clusters[i] = cluster
        for i in range(k):
            points = [data[j] for j in range(len(data)) if clusters[j] == i]
            centerslist[i] = np.mean(points, axis=0)
        filename = "centroids-" + str(iteration+1) + ".csv" #"i" would be each iteration
        np.savetxt(filename, centerslist, delimiter=",")
    return clusters
  
def e_step(data):
    pass

def m_step(data):
    pass
  
def EMGMM(data, clusters):
    # See Lecture 16 page 20
    # initialize with results from K-Means
    points_array = []
    mu_array = []
    sigma_array = []
    pi_array = []
    for i in range(k):
        points = [data[j] for j in range(len(data)) if clusters[j] == i]
        points_array.append(points)
        mu_array.append(np.mean(points, axis=0))
        sigma_array.append(np.cov(np.transpose(points)))
        pi_array.append(len(points) / float(len(data)))

    for iteration in range(10):
        # e-step
        PHI_array = []
        for i in range(len(data)):
            xi = data[i]
            phi = []
            total = 0
            for j in range(k):
                x = xi - mu_array[j]
                exponent = -0.5 * np.dot(x.T.dot(np.linalg.inv(sigma_array[j])), x)
                probability = pi_array[j] / math.sqrt(np.linalg.det(sigma_array[j])) * math.exp(exponent)
                phi.append(probability)
                total += probability
            # normalize
            for j in range(k):
                phi[j] /= total
            PHI_array.append(phi)
            
        # m-step
        for i in range(k):
            pass
        
        
        filename = "pi-" + str(iteration+1) + ".csv" 
        np.savetxt(filename, pi_array, delimiter=",")
        
        filename = "mu-" + str(iteration+1) + ".csv"
        np.savetxt(filename, mu_array, delimiter=",")  #this must be done at every iteration
        
        for j in range(k):
            filename = "Sigma-" + str(j+1) + "-" + str(iteration+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
            #np.savetxt(filename, sigma[j], delimiter=",")
            np.savetxt(filename, sigma_array[j], delimiter=",")

clusters = KMeans(X)
EMGMM(X, clusters)

print np.mean(X,axis=0)
print np.cov(X.T)
# X = [[-2.1, -1,  4.3], [3,  1.1,  0.12]]
# # print np.mean(X, axis=0)
# X = [[-2.1,3], [-1,1.1], [4.3, 0.12]]
# print get_covariance(X)
# print np.cov(np.transpose(X))
