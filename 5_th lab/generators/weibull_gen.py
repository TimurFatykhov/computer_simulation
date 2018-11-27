from .generator import Generator
import numpy as np 

class Weibull_gen(Generator):
    def __init__(self, m, v):
        """
        Initialize Weibull generator

        Inputs:
        - m: Float first parameter (the most frequent element)
        - v: Float second parameter
        """
        self.v = v 
        self.m = m 


    def _pdf(self, x):
        """
        Probability density function of Weibull distribution (aka pdf)

        Inputs: 
        - x: Float x-point

        Outputs:
        - y: Float y-point of pdf
        """
        numerator = self.v * np.power(x, self.v - 1) * np.exp( -np.power(x/self.m, self.v) )
        denominator = self.m ** self.v 
        return numerator / denominator
    

    def _cdf(self, x):
        """
        Cummulative density function of Weibull distribution (aka cdf)

        Inputs: 
        - x: Float x-point

        Outputs:
        - y: Float y-point of cdf
        """
        return 1 - np.exp(-(x/self.m) ** self.v)


    def generate(self, N):
        """
        Generate sequence 

        Inputs:
        - N: Integer size of sequence

        Outputs:
        - seq: Array of elements from Weibull distribution
        """
        self._N = N
        self._uni_seq = np.random.uniform(size=N)

        # apply inverse distribution function (aka PPF)
        ln = np.log(1 - self._uni_seq)
        root = (-ln) ** (1 / self.v)
        return self.m * root
