import numpy as np
import pickle
import scipy as sp
import scipy.sparse as spp

def load_sparse_matrix(filename):
	t = 0
	f = open(filename, 'r')
	f.readline()

	user_list = []
	item_list = []

	for line in f:
		word = line.split(',')
		user_list.append(int(word[0]))
		item_list.append(int(word[1]))

		t += 1
		if t % 50000 == 0:
			print('.', end = '')
			if t % 2000000 == 0:
				print('%d m lines' %(t // 1000000))

	users = list(set(user_list))
	items = list(set(item_list))
	user2id = dict(zip(users, range(len(users))))
	item2id = dict(zip(items, range(len(items))))

	data_list = list(np.ones(t))
	user_id_list = [user2id[u] for u in user_list]
	item_id_list = [item2id[i] for i in item_list]

	result = spp.csr_matrix((data_list, (user_id_list, item_id_list)))
	return result

def extract_users(num_users, sparse_matrix):
	item_count = sparse_matrix.getnnz(axis = 1)
	user_sort = sorted(enumerate(item_count), key = lambda x: x[1], reverse = True)[:num_users]
	user_indices = [x[0] for x in user_sort]

	return spp.vstack([sparse_matrix.getrow(i) for i in user_indices])

def extract_items(num_items, sparse_matrix):
	user_count = sparse_matrix.getnnz(axis = 0)
	item_sort = sorted(enumerate(user_count), key = lambda x: x[1], reverse = True)[:num_items]
	item_indics = [x[0] for x in item_sort]

	return spp.hstack([sparse_matrix.getcol(j) for j in item_indics])

def get_reduced_matrix(num_users, num_items, filename):
	data = load_sparse_matrix(filename)
	data1 = extract_users(3 * num_users, data)
	data2 = extract_items(num_items, data1)
	data3 = extract_users(num_users, data2)
	return data3.toarray()

reduced_matrix = get_reduced_matrix(num_users = 1000, num_items = 1000, filename = 'MovieLens/ratings.csv')
print(reduced_matrix.shape)
np.save('ml_1000user_1000item', reduced_matrix)
X = np.load('ml_1000user_1000item.npy')
print(X.shape)

