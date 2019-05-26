import numpy as np
import random
from functools import reduce

def gopt(A, C = 2):
    """return g-optimal design: a list of tuples"""
    def gram(A):
        return sum([np.outer(x,x) for x in A])

    d = len(A[0])
    A = [tuple(x) for x in A]
    result = set()

    while len(A) > C * d * np.log(d):
        Vinv = np.linalg.pinv(gram(A))

        p = [1/(2* len(A)) + 1/(2*d) * np.dot(x, np.dot(Vinv, x)) for x in A]
        p = np.divide(p, sum(p))

        s = set()
        sampling = True
        while sampling:
            samples = set(np.random.choice(range(len(A)), size=int(C*d*np.log(d)), p=p))
            s = set([A[i] for i in samples])

            Vinv = np.linalg.pinv(gram(s))
            index = [np.dot(x, np.dot(Vinv, x)) <= 1 for x in A]

            if sum(index) >= len(A) / 2:
                result = result | s
                sampling = False
                A = [A[i] for i in range(len(A)) if index[i] == 0]

    result = result | set(A)
    return list(result)


class Part(object):
    """partition structure, items = a list of lists"""
    def __init__(self, l, items, k, m, K, T):
        super(Part, self).__init__()
        self.l = l
        self.items = items
        self.k = k
        self.m = m

        self.spanner = gopt(self.items)

        d = len(self.items[0])
        self.round = 0
        self.budget = int(4 ** self.l * d * np.log(2 * len(self.items) * K * l * (l+1) * np.sqrt(T)) )

class RecurRank(object):
    def __init__(self, K, env, T):
        super(RecurRank, self).__init__()
        self.K = K
        self.env = env
        self.T = T

        self.items = [tuple(list(x)) for x in self.env.items] #self.env.items
        
        L = len(self.items)
        self.ind = {tuple(self.items[i]):i for i in range(L)}

        self.Si = {i:np.outer(self.items[i], self.items[i]) for i in range(L)}
        d = len(self.items[0])
        self.S = {i:np.zeros((d, d)) for i in range(L)}
        self.b = {i:np.zeros(d) for i in range(L)}

        A = np.random.permutation(L)
        self.parts = {0:Part(l=1, items=[self.items[i] for i in A], k=0, m=self.K, K=self.K, T=self.T)}

        self.recommend_list = list(np.ones(self.K))
        self.rewards = np.zeros(self.T)

    def run(self):
        tau = 0
        while tau < self.T:
            for c in self.parts:
                i = np.random.choice(len(self.parts[c].spanner))
                a = self.parts[c].spanner[i]
                self.recommend_list[self.parts[c].k] = self.ind[tuple(a)] #tuple(a)

                A = self.parts[c].items[:]
                if tuple(a) in A:
                    i = A.index(a)
                    A = A[:i] + A[i+1:]
                for k in range(self.parts[c].m-1):
                    self.recommend_list[self.parts[c].k+k+1] = self.ind[tuple(A[k])]

                self.parts[c].round += 1

            x, r = self.env.feedback(self.recommend_list)
            self.rewards[tau] = r

            for c in self.parts:
                k = self.parts[c].k
                i = self.recommend_list[k]
                self.S[i] += self.Si[i]
                self.b[i] += np.dot(self.items[i], x[k])

            # check partitions
            C = set(self.parts)
            for c in C:
                if self.parts[c].round == self.parts[c].budget:
                    Sc = reduce((lambda x, y: x+y), [self.S[self.ind[tuple(a)]] for a in self.parts[c].spanner])
                    bc = reduce((lambda x, y: x+y), [self.b[self.ind[tuple(a)]] for a in self.parts[c].spanner])
                    theta = np.dot(np.linalg.pinv(Sc), bc)

                    sorted_items = sorted([(a, np.dot(theta, a)) for a in self.parts[c].items], key = lambda x:x[1], reverse = True)

                    K1 = self.parts[c].k
                    M = self.parts[c].m
                    l = self.parts[c].l

                    # check for elimination
                    a = sorted_items[M-1][0]
                    for k in range(M, len(self.parts[c].items)):
                        b = sorted_items[k][0]

                        if sorted_items[M-1][1] - sorted_items[k][1] >= 1 / (2 ** (l-1)):
                            sorted_items = sorted_items[:k]
                            break

                    self.parts[c] = Part(l=l+1, items=[x[0] for x in sorted_items],k=K1, m=M, K=self.K, T=self.T)

                    # check for split
                    lastk = 0
                    for k in range(1, M):
                        if sorted_items[k][1] - sorted_items[k-1][1] >= 1 / (2 ** (l-1)):
                            c1 = max(set(self.parts)) + 1
                            self.parts[c1] = Part(l=l+1, items=[sorted_items[i][0] for i in range(lastk, k)], k=K1+lastk, m=k-lastk+1, K=self.K, T=self.T)
                            self.parts[c] = Part(l=l+1, items=[sorted_items[i][0] for i in range(k, len(sorted_items))], k=K1+k, m=len(sorted_items)-k+1, K=self.K, T=self.T)
                            lastk = k

                            print(tau, [(self.parts[cc].k, len(self.parts[cc].items)) for cc in self.parts])

            tau += 1

        return self.rewards
