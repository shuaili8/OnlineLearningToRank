import numpy as np
from sklearn.preprocessing import normalize

def ExtractFeatures(num_users, num_users_in_train, num_items, d, filename):
    X = np.load(filename)

    Nx = np.shape(X)[0]
    Ny = np.shape(X)[1]

    idx = np.random.choice(Nx, num_users + num_users_in_train)
    X = X[idx, :]
    idy = np.random.choice(Ny, num_items)
    X = X[:, idy]

    A1 = X[:num_users_in_train, :]
    u, s, vt = np.linalg.svd(A1)

    vt = vt[:d-1, :]
    vt = normalize(vt, axis = 0, norm = 'l2')

    V = np.concatenate((vt, np.ones((1, num_items))), axis = 0) / np.sqrt(2)
    test_matrix = X[num_users_in_train:, :][:, :num_items]

    A = np.matmul(V, np.transpose(V, (1, 0)))
    b = np.dot(V, np.sum(test_matrix, axis=0)) / num_users
    theta = np.dot(np.linalg.pinv(A), b)

    V = np.transpose(V, (1, 0))

    return V, theta

