import numpy as np 
from scipy.optimize import fmin
from scipy.special import erf

class Lognormal_gen():
    """
    Generate sequence from lognormal distribution
    """

    def __init__(self, mu, sigma, capacity = 0.98):
        """
        Inputs:
        - mu: Float parameter of distribution
        - sigma: Float parameter of distribution
        - capacity: Float value from interval [0, 1);
        what percent of cdf to use
        """
        self.mu = mu 
        self.sigma = sigma
        
        if capacity is not None and capacity >= 0 and capacity < 1:
            self.capacity = capacity 
        else:
            ValueError('capacity have to be from interval [0, 1)')

        self.a, self.b = self._get_bounds_(capacity)

        # find point where pdf is maximum
        self.max_x = fmin(lambda x: -self._pdf_(x), self.a, disp=False)
        self.max_y = self._pdf_(self.max_x)


    def _pdf_(self, x):
        """ 
        Probability density function of log-normal distribution
        Inputs:
        - x: Array, list or single value
        """
        if type(x) == list:
            x = np.array(x)

        res = None
        if type(x) is np.ndarray:
            # if x is equal to 0, then put 0 to result vector;
            # define mask of non-zero elements
            mask = x > 1e-6
            res = np.empty_like(x) 

            res[~mask] = 0
            e = np.exp(-(np.log(x[mask]) - self.mu)**2/(2*self.sigma**2))
            denominator = x[mask] * self.sigma * np.sqrt(2 * np.pi)
            res[mask] = e / denominator
        else: 
            if x == 0:
                res = 0
            else:
                e = np.exp(-(np.log(x) - self.mu)**2/(2*self.sigma**2))
                denominator = x * self.sigma * np.sqrt(2 * np.pi)
                res = e / denominator

        return res 
    
    def _cdf_(self, x):
        """
        Cummulative distribution function
        """
        return 1/2 * (1 + erf((np.log(x) - self.mu) / (self.sigma * np.sqrt(2))) )


    def _get_bounds_(self, capacity=None, dx=1e-3):
        """
        Calculate bounds according to capacity of generetor

        Inputs:
        - capacity: Float value from [0, 1)
        - dx: Float step for numerical integral calculation

        Outputs:
        - a, b: Tupple of bounds of x
        """
        capacity = self.capacity if capacity is None else capacity
        if capacity is not None and capacity >= 0 and capacity < 1:
            self.capacity = capacity 
        else:
            ValueError('capacity have to be from interval [0, 1)')

        offset = (1 - capacity) / 2

        x = 0
        cumm = 0
        while cumm < offset:
            s = self._pdf_(x) * dx
            cumm += s 
            x += dx 
        a = x 

        while cumm < 1 - offset:
            s = self._pdf_(x) * dx
            cumm += s
            x += dx
        b = x 

        return a, b 


    def generate(self, N):
        """
        Generate sequence from log-normal distribution

        Inputs:
        - N: Integer size of sequence

        Outputs:
        - sequence
        """
        self._N = N
        seq = []
        seq_size = 0
        iters = 0
        while seq_size < N:
            iters += 1
            x_value = self.a + np.random.rand(1)[0] * (self.b - self.a)
            y_value = np.random.rand(1) * self.max_y

            if y_value <= self._pdf_(x_value):
                seq.append(x_value)
                seq_size += 1

        self._iters = iters
        return np.array(seq) 