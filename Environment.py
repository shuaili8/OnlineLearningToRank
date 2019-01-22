import numpy as np
from sklearn.preprocessing import normalize

from ExtractFeatures import ExtractFeatures

class Environment(object):
    def __init__(self, L, d, synthetic=True, tabular=False, filename=''):
        super(Environment, self).__init__()
        if synthetic:
            if tabular:
                self.items = np.eye(L)
                # self.means = np.random.rand(L)
                self.means = np.arange(1,0,-1/L)
            else:
                self.items = self.genitems(L, d)
                theta = self.genitems(1, d)[0]
                self.means = np.dot(self.items, theta)
        else:
            self.items, theta = ExtractFeatures(num_users=1000, num_users_in_train=100, num_items=L, d=d, filename=filename)
            self.means = np.dot(self.items, theta)
            
        # self.K = K

    def genitems(self, L, d):
        # Return an array of L * d, where each row is a d-dim feature vector with last entry of 1/sqrt{2}
        A = np.random.normal(0, 1, (L,d-1))
        result = np.hstack(( normalize(A, axis=1)/np.sqrt(2), np.ones((L,1))/np.sqrt(2) ))
        return result

class CasEnv(Environment):
    def __init__(self, L, d, synthetic=True, tabular=False, filename=''):
        super(CasEnv, self).__init__(L, d, synthetic=True,tabular=tabular,filename=filename)

    def or_func(self, v):
        return 1 - np.prod(1 - v)
    
    def feedback(self, A):
        means = self.means[A]
        x = np.random.binomial(1, means)
        if x.sum() > 1:
            first_click = np.flatnonzero(x)[0]
            x[first_click + 1 : ] = 0

        return x, self.or_func(means)

    def get_best_reward(self, K):
        bestmeans = np.sort(self.means)[::-1][:K]
        breward = self.or_func(bestmeans)
        return breward

class PbmEnv(Environment):
    def __init__(self, L, d, beta, synthetic=True, tabular=False, filename=''):
        super(PbmEnv, self).__init__(L, d, synthetic=True, tabular=tabular, filename=filename)
        self.beta = beta

    def feedback(self, A):
        means = self.means[A] * self.beta[:len(A)]
        return np.random.binomial(1, means), sum(means)

    def get_best_reward(self, K):
        beta = self.beta[:K]
        bestmeans = np.sort(self.means)[::-1][:K]
        breward = np.dot(beta, bestmeans)
        return breward

# from ExtractFeatures import ExtractFeatures

# class RealDataEnv(Environment):
#     def __init__(self, L, d, K, filename='yelp_1100user_1000item.npy'):  #'yelp_1100user_1000item.npy'
#         super(RealDataEnv, self).__init__(L, d, K, synthetic=False)
#         self.items, theta = ExtractFeatures(num_users=1000, num_users_in_train=100, num_items=L, d=d, filename=filename)
#         self.means = np.dot(self.items, theta)

#     def feedback(self, A):
#         ui = np.random.choice(900)
#         r = self.fbmat[ui,A]
#         return r, sum(r)
        
        