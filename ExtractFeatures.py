import numpy as np
import pickle
from sklearn.preprocessing import normalize
import time

def ExtractFeatures(num_users, num_users_in_train, num_items, d, filename):
	X = np.load(filename)

	Nx = np.shape(X)[0]
	Ny = np.shape(X)[1]

	idx = np.random.choice(Nx, num_users + num_users_in_train) #np.random.shuffle(np.arange(N))
	X = X[idx, :]
	idy = np.random.choice(Ny, num_items)  #np.random.shuffle(np.arange(N))
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

	# np.savez('M_'+str(num_items)+'_'+str(d)+'_'+str(seed), np.transpose(V, (1, 0)), test_matrix)
# 'movielens_best_pool_1k_1k.pickle'
# filename = 'yelp_1100user_1000item.npy' # 'ml_1k_1k.npy'
# ExtractFeatures(num_users=900,num_users_in_train=100,num_items=1000,d=10,filename=filename)
