import numpy as np
import sys
from utlis import is_power2

class CascadeLinTS:
    # Current version is for fixed item set
    def __init__(self, K, env, T):
        super(CascadeLinTS, self).__init__()
        self.K = K
        self.env = env
        self.T = T

        self.items = self.env.items
        self.d = len(self.items[0])
        
        self.S = np.zeros((self.d, self.d))
        self.b = np.zeros(self.d)
        self.Sinv = np.zeros((self.d, self.d))
        self.theta = np.zeros(self.d)

        self.rewards = np.zeros(self.T)

    def run(self):
        for t in range(self.T):
            if True:#t % 5000 == 0 or is_power2(t):
                self.Sinv = np.linalg.pinv(self.S)
                theta = np.dot(self.Sinv, self.b)
                self.theta = np.random.multivariate_normal(theta, self.Sinv)

            At = np.argsort(np.dot(self.items, self.theta))[:: -1][: self.K]

            x, r = self.env.feedback(At)
            self.rewards[t] = r

            first_click = self.K - 1
            if x.sum() > 0:
                first_click = np.flatnonzero(x)[0]

            A = self.items[At[: first_click + 1]]
            x = x[: first_click + 1]
            self.S += np.matmul(A.T, A)
            self.b += np.dot(x, A)

        return self.rewards

