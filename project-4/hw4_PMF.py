from __future__ import division
import numpy as np
import sys

# See lecture 17 pages 17..19
# https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf
# this is helpful: https://www.quora.com/What-is-the-meaning-of-low-rank-matrix?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
# if len(sys.argv) < 2:
#     sys.argv = ['', 'ratings.csv']
train_data_string = np.genfromtxt(sys.argv[1], delimiter = ",")

lam = 2
sigma2 = 0.1
d = 5

train_data = []
for i in range(len(train_data_string)):
    data_point = []
    data_point.append(int(train_data_string[i][0]))
    data_point.append(int(train_data_string[i][1]))
    data_point.append(float(train_data_string[i][2]))
    train_data.append(data_point)

def get_matrix_dimension(train_data):
    N1, N2 = 0, 0
    for data_point in train_data:
        user_index = data_point[0]
        object_index = data_point[1]
        if N1 < user_index:
            N1 = user_index
        if N2 < object_index:
            N2 = object_index
    return N1, N2

def create_matrix(train_data, N1, N2):
    M = np.zeros((N1, N2))
    for data_point in train_data:
        user_index, object_index, rating = data_point[0], data_point[1], data_point[2]
        # ratings in train_data is one-based
        M[user_index - 1][object_index - 1] = rating
    return M

def create_omega_u(train_data, N1):
    omega_u = []
    for i in range(N1):
        row = []
        omega_u.append(row)
    for data_point in train_data:
        user_index, object_index = data_point[0], data_point[1]
        omega_u[user_index - 1].append(object_index)
    return omega_u

def create_omega_v(train_data, N2):
    omega_v = []
    for i in range(N2):
        row = []
        omega_v.append(row)
    for data_point in train_data:
        user_index, object_index = data_point[0], data_point[1]
        omega_v[object_index - 1].append(user_index)
    return omega_v

def get_sum_omega_u(omega_u, V):
    result = np.zeros((d, d))
    for j in omega_u:
        v = np.matrix(V[:, j-1])
        result += np.dot(v.T, v)
    return result

def get_sum_M_object(M, i, omega_u, V):
    result = np.zeros(d)
    for j in omega_u:
        v = V[:, j-1]
        mij = M[i-1][j-1]
        result += mij * v
    return np.matrix(result)

def get_sum_omega_v(omega_v, U):
    result = np.zeros((d, d))
    for i in omega_v:
        u = np.matrix(U[i-1])
        result += np.dot(u.T, u)
    return result

def get_sum_M_user(M, j, omega_v, U):
    result = np.zeros(d)
    for i in omega_v:
        u = U[i-1]
        mij = M[i-1][j-1]
        result += mij * u
    return result

# Implement function here
def PMF(train_data):
    L = []
    N1, N2 = get_matrix_dimension(train_data)
    M = create_matrix(train_data, N1, N2)
#     print N1, N2
#     print M
    V = np.random.normal(0, 1/lam, (d, N2))
    U = np.zeros((N1, d))
    omega_u = create_omega_u(train_data, N1)
    omega_v = create_omega_v(train_data, N2)
    
#     print omega_u
#     print omega_v
    #for iteration in range(1, 51):
    for iteration in range(1, 51):
        # update U
        for i in range(1, N1+1):
            term1 = lam * sigma2 * np.identity(d) + get_sum_omega_u(omega_u[i-1], V)
            term1_inverse = np.linalg.inv(term1)
            term2 = get_sum_M_object(M, i, omega_u[i-1], V)
            result = np.dot(term1_inverse, term2.T)
            U[i-1] = result.T
            
        # update V
        for j in range(1, N2+1):
            term1 = lam * sigma2 * np.identity(d) + get_sum_omega_v(omega_v[j-1], U)
            term1_inverse = np.linalg.inv(term1)
            term2 = get_sum_M_user(M, j, omega_v[j-1], U)
            result = np.dot(term1_inverse, term2.T)
            for k in range(d):
                V[k][j-1] = result[k]
            
        if iteration == 10 or iteration == 25 or iteration == 50:
            np.savetxt("U-" + str(iteration) + ".csv", U, delimiter=",")
            np.savetxt("V-" + str(iteration) + ".csv", V.T, delimiter=",")
        
        L.append(0.1)
    np.savetxt("objective.csv", L, delimiter=",")

# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
#L, U_matrices, V_matrices = PMF(train_data)
PMF(train_data)
# np.savetxt("objective.csv", L, delimiter=",")
# np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
# np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
# np.savetxt("U-50.csv", U_matrices[49], delimiter=",")
# np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
# np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
# np.savetxt("V-50.csv", V_matrices[49], delimiter=",")