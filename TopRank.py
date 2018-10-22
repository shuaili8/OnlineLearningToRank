import numpy as np

class TopRank(object):
	def __init__(self, K, env, T):
		super(TopRank, self).__init__()
		self.K = K
		self.env = env
		self.T = T

		self.L = len(env.items)
		self.N = np.ones((self.L, self.L))
		self.S = np.zeros((self.L, self.L))

		self.G = np.ones((self.L, self.L), dtype = bool)
		self.P = np.zeros(self.L) # partition index for items
		self.Peq = np.ones((self.L, self.L)) # if indexes for (i,j) are the same

		self.rewards = np.zeros(self.T)

	def rerank(self):
		c = 3.43
		Gt = (self.S - np.sqrt(2 * self.N * np.log(c * np.sqrt(self.T) * np.sqrt(self.N)))) > 0
		
		if not np.array_equal(Gt, self.G):
			self.G = np.copy(Gt)

			Pid = 0
			self.P = - np.ones(self.L)
			while (self.P == -1).sum() > self.L - self.K:
				items = np.flatnonzero(Gt.sum(axis = 0) == 0)
				self.P[items] = Pid
				Gt[items, :] = 0
				Gt[:, items] = 1
				Pid += 1

			items = np.flatnonzero(self.P == -1)
			self.P[items] = Pid

			self.Peq = (np.tile(self.P[np.newaxis], (self.L, 1)) == np.tile(self.P[np.newaxis].T, (1, self.L))).astype(float)

	def update(self, t, At, x, r):
		self.rewards[t] = r

		clicks = np.zeros(self.L)
		clicks[np.asarray(At)] = x

		M = np.outer(clicks, 1 - clicks) * self.Peq
		self.N += M + M.T
		self.S += M - M.T

		if t % 1000 == 0:
			self.rerank()

	def get_action(self):
		return np.argsort(self.P + 1e-6 * np.random.rand(self.L))[: self.K]

	def run(self):
		for t in range(self.T):
			At = self.get_action()
			x, r = self.env.feedback(At)
			self.update(t, At, x, r)

		return self.rewards
