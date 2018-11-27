import math
import numpy as np
from .uniform.uni_gen import Uni_generator

class Poisson_gen():
    def __init__(self, alpha, k=None, verbose=False):
        """
        Inputs:
        - alpha: Integer most popular element (aka lambda)
        - k: Integer size of probability row; if set to None
        then will be calculate durin generating process
        - verbose: Boolean; if set to true then print logs
        """
        self.alpha = alpha
        self.k = k
        self._verbose = verbose


    def _get_prob_row_(self):
        """
        Return probability row of poisson distribution
        """
        prob_row = []
        if self.k is not None:
            for i in range(self.k + 1):
                p = self.alpha**i * math.exp(-self.alpha) / math.factorial(i)
                prob_row.append(p)
        else:
            if self._N is None:
                raise ValueError('k is None, but desirable size of sequence is not defined! Try to set self._N manualy.')
            i = 0
            p = 1
            while self._N * p > 1:
                p = self.alpha**i * math.exp(-self.alpha) / math.factorial(i)
                prob_row.append(p)
                i += 1
            self._k = i

        return np.array(prob_row)


    def _print(self, string):
        """
        Print logs if self.verbose is equal true
        """
        if self._verbose:
            print(string)
    

    def set_verbose_to(self, verbose):
        """
        self.verbose setter
        """
        if type(verbose) is not bool:
            raise ValueError('verbose must be bool, not ' + str(type(verbose)))
        self._verbose = verbose


    def generate(self, N, set_verbose_to=None):
        """
        Inputs:
        - N: Integer length of sequence
        - set_verbose_to: Boolean; if set to None then
        do not change set value

        Outputs:
        - res: Array; generated sequence
        - steps: Integer number of steps required for genearation
        """
        # we have to know value of N if self.k is set to None
        self._N = N
        if set_verbose_to is not None:
            self._verbose = set_verbose_to

        self._uni_seq = np.random.uniform(size=N)

        self._Q = 0
        self._prob_row = self._get_prob_row_()
        for i in range(self.alpha + 1):
            self._Q += self._prob_row[i]

        steps = 0
        res = []
        self._print('. Q = % 5.2f\n'%(self._Q))
        for u in self._uni_seq:
            # calculate subtraction between generated probability
            # and cumulative summ of probabilities of first self._alpha 
            # elements
            s = u - self._Q 
            self._print('.. u = % 5.2f'%(u))
            self._print('.. u - Q = % 5.2f'%(s))

            # determine the direction of seek
            if s > 0:
                j = 0
                # self._print('...... s > 0')
                while s  > 1e-7:
                    j += 1 
                    s -= self._prob_row[self.alpha + j]
                    # self._print('...... j = %2d | p = % 5.2f | s = % 5.2f'%(j, self._prob_row[self.alpha + j], s))
                    steps += 1
                res.append(self.alpha + j)
                self._print('...... generated = %d | j = %d '%(self.alpha + j, j))
            elif s < 0:
                j = -1
                # self._print('...... s < 0')
                while s <= -1e-7:
                    j += 1 
                    s += self._prob_row[self.alpha - j]
                    # self._print('...... j = %2d | p = % 5.2f | s = % 5.2f'%(j, self._prob_row[self.alpha - j], s))
                    steps += 1
                res.append(self.alpha - j)
                self._print('...... generated = %d | j = %d '%(self.alpha - j, j))
            else:
                res.append(self.alpha)
            self._print('')

        return np.array(res), steps