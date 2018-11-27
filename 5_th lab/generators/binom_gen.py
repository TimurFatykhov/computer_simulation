import math
import numpy as np
from .uniform.uni_gen import Uni_generator

class Binom_gen():
    """
    Generate binomial distribution
    """

    def __init__(self, m, p):
        """
        Inputs:
        - m: Integer count of experiments
        - p: Float probability of success
        """
        self.m = m 
        self.p = p


    def _combination_(self, n, k):
        """
        Calculate combination from 'n' by 'k'
        """
        res = 1
        for i in range(n - k + 1, n + 1):
            res *= i
        res /= math.factorial(k) 
        return res


    def _prob(self, k):
        """
        Calculate probability for appropriate k
        """
        C = self._combination_(self.m, k)
        res = C * self.p**k * (1-self.p)**(self.m-k)
        return res

    def generate(self, N):
        """
        Generate binomial sequence 
        """
        # create probability row
        self._prob_row = np.array([])
        for k in range(self.m + 1):
            self._prob_row = np.append(self._prob_row, self._prob(k))
        self._prob_cumsum = np.cumsum(self._prob_row)
        
        # create seq belong to uni(0, 1)
        uni = Uni_generator(1, 2)
        uni_seq = uni.generate(N, normalized=True)
        uni_seq = np.array(uni_seq)
        
        # calculation elements belong to binomial destribution
        
        # fast calculation via numpy:
        #
        # sub = uni_seq - self._prob_cumsum.reshape((self._prob_cumsum.shape[0], -1))
        # sub[sub >= 0] = -2
        # res = np.argmax(sub, axis=0)

        # vanila algorithm
        steps = 0
        res = []
        for i in range(len(uni_seq) - 1, -1, -1):
            j = 0
            el = uni_seq[i]
            while el - self._prob_cumsum[j] > 0:
                j += 1
                steps += 1
            res.append(j) 
            
        return np.array(res), steps


    