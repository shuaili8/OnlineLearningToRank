import numpy as np

# class Partition(object):
# 	"""docstring for Partition"""
# 	def __init__(self, items, K1, K2):
# 		super(Partition, self).__init__()
# 		self.items = items
# 		self.K1 = K1
# 		self.K2 = K2

class TopRank(object):
	"""docstring for TopRank"""
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

		# self.S = np.zeros((self.L, self.L))
		# self.N = np.zeros((self.L, self.L))
		# self.G = set()

		# self.partitions = {0:Partition(items=list(np.arange(self.L)), K1=0, K2=self.K-1)}
		# self.recommend_list = list(np.ones(self.K))
		self.rewards = np.zeros(self.T)

	# def criterion(self, S, N):
	# 	c = 3.43
	# 	return S >= np.sqrt(2 * N * np.log(c * np.sqrt(self.T) * np.sqrt(N)))

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

		self.rerank()

	def get_action(self):
		return np.argsort(self.P + 1e-6 * np.random.rand(self.L))[: self.K]

	def run(self):
		for t in range(self.T):
			At = self.get_action()
			x, r = self.env.feedback(At)
			self.update(t, At, x, r)

			# for c in self.partitions:
			# 	partition = self.partitions[c]
			# 	partition.items = np.random.permutation(partition.items)
			# 	self.recommend_list[partition.K1:partition.K2+1] = partition.items[:partition.K2-partition.K1+1]

			# # receive feedback
			# x, r = self.env.feedback(self.recommend_list)
			# self.rewards[tau] = r
			# rewards = np.zeros(self.L)
			# for k in range(self.K):
			# 	a = self.recommend_list[k]
			# 	rewards[a] = x[k]

			# update statistics
			# updateG = False
			# for c in self.partitions:
			# 	partition = self.partitions[c]
			# 	for i in range(partition.K2-partition.K1+1):
			# 		a = partition.items[i]
			# 		for j in range(i+1, len(partition.items)):
			# 			b = partition.items[j]
			# 			self.S[a, b] += rewards[a] - rewards[b]
			# 			self.N[a, b] += np.abs(rewards[a] - rewards[b])
			# 			self.S[b, a] += rewards[b] - rewards[a]
			# 			self.N[b, a] += np.abs(rewards[b] - rewards[a])

			# 			# add pair if needed
			# 			if self.N[a, b] > 0:
			# 				if self.criterion(self.S[a, b], self.N[a, b]):
			# 					self.G.add((b, a))
			# 					updateG = True
			# 			elif self.N[b, a] > 0:
			# 				if self.criterion(self.S[b, a], self.N[b, a]):
			# 					self.G.add((a, b))
			# 					updateG = True

			# # update partitions
			# if updateG:
			# 	self.partitions = {}
			# 	c = 0
			# 	K1 = 0
			# 	itemset = set(np.arange(self.L))
			# 	while K1 < self.K:
			# 		A = set([x[0] for x in self.G if x[1] in itemset and x[0] in itemset])
			# 		B = itemset-A
			# 		self.partitions[c] = Partition(items=list(B),K1=K1,K2=min(K1+len(B)-1, self.K-1))

			# 		K1 += len(B)
			# 		itemset = A
			# 		c += 1
			# 	# print(tau, [self.partitions[x].items for x in self.partitions])

			# tau += 1

		return self.rewards




