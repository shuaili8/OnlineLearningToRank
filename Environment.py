import numpy as np
from sklearn.preprocessing import normalize

class Environment(object):
	def __init__(self, L, d, K, synthetic=True):
		super(Environment, self).__init__()
		if synthetic:
			self.items = self.genitems(L, d)
			theta = self.genitems(1, d)[0]
			self.means = np.dot(self.items, theta)
			self.bestmeans = np.sort(self.means)[::-1][:K]

		self.K = K

	def genitems(self, L, d):
		# Return an array of L * d, where each row is a d-dim feature vector with last entry of 1/sqrt{2}
		A = np.random.normal(0, 1, (L,d-1))
		result = np.hstack(( normalize(A, axis=1)/np.sqrt(2), np.ones((L,1))/np.sqrt(2) ))
		return result

class CasEnv(Environment):
	def __init__(self, L, d, K):
		super(CasEnv, self).__init__(L, d, K)
		self.breward = self.or_func(self.bestmeans)

	def or_func(self, v):
		return 1 - np.prod(1 - v)
	
	def feedback(self, A):
		means = self.means[A]
		x = np.random.binomial(1, means)
		if x.sum() > 1:
			first_click = np.flatnonzero(x)[0]
			x[first_click + 1 : ] = 0

		return x, self.or_func(means)

class PbmEnv(Environment):
	def __init__(self, L, d, K, beta):
		super(PbmEnv, self).__init__(L, d, K)
		self.beta = beta
		self.breward = np.dot(self.beta, self.bestmeans)

	def feedback(self, A):
		means = self.means[A] * self.beta
		return np.random.binomial(1, means), sum(means)

from ExtractFeatures import ExtractFeatures

class RealDataEnv(Environment):
	def __init__(self, L, d, K, filename='ml_1k_1k.npy'):
		super(RealDataEnv, self).__init__(L, d, K, synthetic=False)
		self.items, self.fbmat = ExtractFeatures(num_users=900, num_users_in_train=100, num_items=L, d=d, filename=filename)

	def feedback(self, A):
		ui = np.random.choice(900)
		r = self.fbmat[ui,A]
		return r, sum(r)
		
		