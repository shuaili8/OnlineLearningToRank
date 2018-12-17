import numpy as np
from utlis import is_power2

class Partition:
	def __init__(self, items, k, m):
		self.items = items
		self.k = k
		self.m = m

class TopRank:
	def __init__(self, K, env, T):
		self.K = K
		self.env = env
		self.T = T

		self.L = len(env.items)
		self.S = np.zeros((self.L, self.L)) # S[i,j] = \sum_t U(t,i,j)
		self.N = np.zeros((self.L, self.L)) # N[i,j] = \sum_t |U(t,i,j)| = \sum_t |C(t,i) - C(t,j)| 1{i,j in the same partition}

		self.G = np.zeros((self.L, self.L), dtype = bool) # G[i,j] = 1 iff i is better than j
		self.partitions = {0:Partition(items=range(self.L), k=0, m=self.K)}
		
		self.rewards = np.zeros(self.T)

	def _criterion(self, S, N):
		c = 3.43
		return S >= np.sqrt(2 * N * np.log(c * np.sqrt(self.T) * np.sqrt(N)))

	def update(self, t, At, x, r):
		self.rewards[t] = r

		clicks = np.zeros(self.L)
		clicks[np.asarray(At)] = x

		# update S and N
		for c in self.partitions:
			partition = self.partitions[c]
			for i in range(partition.m):
				a = partition.items[i]
				for j in range(i+1, len(partition.items)):
					b = partition.items[j]
					x = clicks[a] - clicks[b]
					self.S[a, b] += x
					self.N[a, b] += np.abs(x)
					self.S[b, a] -= x
					self.N[b, a] += np.abs(x)

		if True:#t % 2000 == 0  or is_power2(t):
			updateG = False
			for c in self.partitions:
				partition = self.partitions[c]
				for i in range(partition.m):
					a = partition.items[i]
					for j in range(i+1, len(partition.items)):
						b = partition.items[j]

						if self.N[a, b] > 0:
							if self._criterion(self.S[a, b], self.N[a, b]): # a is better than b
								self.G[a, b] = True
								updateG = True
							elif self._criterion(self.S[b, a], self.N[b, a]):
								self.G[b, a] = True
								updateG = True
			
			if updateG:
				self.partitions = {}
				c = 0
				k = 0
				remain_items = set(np.arange(self.L))
				while k < self.K:
					bad_items = set(np.flatnonzero(np.sum(self.G[np.asarray(list(remain_items)),:], axis=0)))
					good_items = remain_items - bad_items
					self.partitions[c] = Partition(items=list(good_items),k=k,m=min(len(good_items), self.K-k))

					k += len(good_items)
					remain_items = remain_items.intersection(bad_items)
					c += 1
				# print(t, [self.partitions[x].items for x in self.partitions])

	def select(self):
		A = [0] * self.K
		for c in self.partitions:
			partition = self.partitions[c]
			partition.items = np.random.permutation(partition.items)
			A[partition.k:partition.k+partition.m] = partition.items[:partition.m]
		return A

	def run(self):
		for t in range(self.T):
			At = self.select()
			x, r = self.env.feedback(At)
			self.update(t, At, x, r)

		return self.rewards
