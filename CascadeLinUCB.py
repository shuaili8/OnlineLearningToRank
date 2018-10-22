import numpy as np
import sys

class CascadeLinUCB(object):
	# Current version is for fixed item set
	def __init__(self, K, env, T):
		super(CascadeLinUCB, self).__init__()
		self.K = K
		self.env = env
		self.T = T

		self.items = self.env.items
		self.d = len(self.items[0])
		self.beta = np.sqrt(self.d * np.log((1 + self.T/self.d) + 2*np.log(4*self.T))) + 1
		
		# L = len(self.items)
		self.S = np.zeros((self.d, self.d))
		self.b = np.zeros(self.d)

		self.rewards = np.zeros(self.T)

	def run(self):
		for t in range(self.T):
			if np.linalg.cond(self.S) < 1 / sys.float_info.epsilon:
				Sinv = np.linalg.inv(self.S)
				theta = np.dot(Sinv, self.b)

				At = np.argsort(np.dot(self.items, theta) + self.beta * (np.matmul(self.items, Sinv) * self.items).sum(axis = 1))[:: -1][: self.K]
			else:
				At = np.random.permutation(len(self.items))[: self.K]

			x, r = self.env.feedback(At)
			self.rewards[t] = r

			first_click = self.K
			if x.sum() > 0:
				first_click = np.flatnonzero(x)[0]

			A = self.items[At[:first_click]]
			x = x[:first_click]
			self.S += np.matmul(A.T, A)
			self.b += np.dot(x, A)

		return self.rewards

