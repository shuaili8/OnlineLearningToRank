import numpy as np
from sklearn.preprocessing import normalize
from functools import reduce

def genitems(L, d):
	"""return a list of tuples"""
	A = np.random.normal(0,1,(L,d-1))
	result = np.hstack(( normalize(A, axis=1)/np.sqrt(2), np.ones((L,1))/np.sqrt(2) ))
	return result # [tuple(list(x)) for x in result]

class Environment(object):
	"""docstring for Environment"""
	def __init__(self, L, d, K, synthetic=True):
		super(Environment, self).__init__()
		if synthetic:
			self.items = genitems(L, d)
			theta = genitems(1, d)[0]
			self.means = np.dot(self.items, theta) #[np.dot(x, theta) for x in self.items]
			self.breward = 0
			self.bestmeans = np.sort(self.means)[::-1][:K]

		self.K = K

	def feedback(self, A):
		pass

class CasEnv(Environment):
	"""docstring for CasEnv"""
	def __init__(self, L, d, K):
		super(CasEnv, self).__init__(L, d, K)
		# bestmeans = sorted(self.means, reverse=True)[:self.K]
		# bestmeans = sorted([np.dot(x, self.theta) for x in self.items], reverse=True)[:self.K]
		self.breward = self.or_func(self.bestmeans)

	def or_func(self, v):
		return 1 - np.prod(1 - v) #1 - reduce((lambda x, y: x * y), [1-x for x in A])
	
	def feedback(self, A):
		means = self.means[A] #[self.means[i] for i in A]
		# means = [np.dot(self.items[i], self.theta) for i in A]
		# print(means)
		x = np.random.binomial(1, means)
		if x.sum() > 1:
			first_click = np.flatnonzero(x)[0]
			x[first_click + 1 : ] = 0

		return x, self.or_func(means)
		# inds = np.flatnonzero(var) #list(np.random.binomial(1, means))
		# result = np.zeros(len(A))
		# if 1 in var:
		# 	result[var.index(1)] = 1

		# return result, self.orfun(means)

class PbmEnv(Environment):
	"""docstring for PbmEnv"""
	def __init__(self, L, d, K, beta):
		super(PbmEnv, self).__init__(L, d, K)
		self.beta = beta
		# bestmeans = sorted(self.means, reverse=True)[:self.K]
		# bestmeans = sorted([np.dot(x, self.theta) for x in self.items], reverse=True)[:self.K]
		self.breward = np.dot(self.beta, self.bestmeans)

	def feedback(self, A):
		means = self.means[A] * self.beta #[self.means[A[k]] * self.beta[k] for k in range(len(A))]
		# means = [np.dot(self.items[A[k]], self.theta) * self.beta[k] for k in range(len(A))]
		return np.random.binomial(1, means), sum(means)

# import os
from extract import extract
# import time

class MLEnv(Environment):
	"""docstring for MLEnv"""
	def __init__(self, L, d, K):
		super(MLEnv, self).__init__(L, d, K, synthetic=False)

		# seed = int(time.time() * 100) % 339
		V, self.fbmat = extract(num_users=900,num_users_in_train=100,num_items=L,d=d,filename='ml_1k_1k.npy')
		# file = np.load('M_'+str(L)+'_'+str(d)+'_'+str(seed)+'.npz')
		# V = file['arr_0']
		# self.fbmat = file['arr_1']
		self.items = V #[tuple(V[i,:]) for i in range(L)]

	def feedback(self, A):
		ui = np.random.choice(900)
		# print(A)
		r = self.fbmat[ui,A] #[self.fbmat[ui, j] for j in A]
		return r, sum(r)

class YelpEnv(Environment):
	"""docstring for MLEnv"""
	def __init__(self, L, d, K):
		super(YelpEnv, self).__init__(L, d, K, synthetic=False)

		# seed = int(time.time() * 100) % 339
		V, self.fbmat = extract(num_users=900,num_users_in_train=100,num_items=L,d=d,filename='yelp_1k_1k.npy')
		# file = np.load('M_'+str(L)+'_'+str(d)+'_'+str(seed)+'.npz')
		# V = file['arr_0']
		# self.fbmat = file['arr_1']
		self.items = V #[tuple(V[i,:]) for i in range(L)]

	def feedback(self, A):
		ui = np.random.choice(900)
		# print(A)
		r = self.fbmat[ui,A] #[self.fbmat[ui, j] for j in A]
		return r, sum(r)
		
		