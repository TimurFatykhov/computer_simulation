from scipy.stats import chisquare
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate as integrate
from scipy.special import gamma

import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir) 
# from empirical_tests import _freqs 

def _freqs(seq, lower_bound, upper_bound, n, normalized=False):
    """
    Return frequences of sequence values

    Inputs:
    - seq: Array of integers. Observable sequence
    - lower_bound: Integer lower bound of the domain
     of generator values
    - upper_bound: Integer upper bound of the domain
     of generator values
    - n: Integer number of regions

    Outputs:
    - freqs: Array of occurences of values in each region 
    with region_width
    - region_width: Float width of regions
    """
    freqs = []
    region_width = (upper_bound - lower_bound) / n 

    for i in range(n):
        low = lower_bound + i * region_width
        high = lower_bound + i * region_width + region_width
        freqs.append( np.logical_and(seq >= low, seq < high).sum() )

    # because last interval has '[a;b]' - bounds, not '[a,b)'
    freqs[-1] += 1
    
    if normalized:
        freqs = np.array(freqs) / len(seq)

    return np.array(freqs), region_width

def chisquare(empiric_freqs, probs, n, alpha):
    """
    Inputs:
    - empiric: Empirical frequencies
    - probs: Theoretical probabilities
    - n: Size of sequence
    """
    k = len(empiric_freqs)
    r = k - 1
    s = (np.square(empiric_freqs - probs) / probs).sum() * n 
    p = integrate.quad( lambda x: x**(r/2 - 1) * np.exp(-x/2) , s, np.inf)[0] \
                                                / (2**(r/2) * gamma(r/2))
    is_rejected = p <= alpha

    return is_rejected, s, p



def chisquare_uniform(seq, 
                    lower_bound, 
                    upper_bound, 
                    alpha=0.05, 
                    n = None, 
                    verbose=True):
    """
    Statistical test applied to sets of categorical data to 
    evaluate how likely it is that any observed difference 
    between the sets arose by chance.

    Inputs:
    - seq: Array of integers
    - lower_bound: Integer lower bound of the domain of 
    generator values
    - upper_bound: Integer upper bound of the domain of 
    generator values
    - alpha: Float desirible level of significance
    - n: Integer number of regions (default is None)
    - verbose: Boolean; If set to true then print logs

    Outputs:
    - Boolean; If hypothesis is ejected
    """
    seq = np.array(seq)
    if n is None:
        n = len(seq)
    else:
        seq = np.random.choice(seq, n)

    k = int(5 * np.log(n))
    r = k - 1 

    freqs, _ = _freqs(seq, lower_bound, upper_bound, k)

    freqs = np.array(freqs) / n 
    p = 1 / k
    s = (np.square(freqs - p) / p).sum() * n 

    p = integrate.quad( lambda x: x**(r/2 - 1) * np.exp(-x/2) , s, np.inf)[0] \
                                                / (2**(r/2) * gamma(r/2))
    is_rejected = p <= alpha

    # list for x-values for barplot
    region_size = upper_bound / k 
    left_borders = []
    for i in range(k):
        low = i * region_size
        high = i * region_size + region_size
        left_borders.append(low)

    if verbose:
        plt.ylim(-0.0, 0.15)
        plt.bar(left_borders, height=freqs, align='edge', width=upper_bound/k,
                    color='blue', label='frequences')
        plt.hlines(1 / k, lower_bound, upper_bound, 'r', label='n / k')
        plt.title(f'Frequency bar plot (1/k = {1/k})') 
        plt.legend()
        plt.show() 

        print(f'S = {s}')
        print(f'Number of interval k = {k}')
        print(f'Sequence length n = {n}')
        print('P = %f' % p)

        # show bar plot
        # plt.ylim(-0.1, 0.3)
        # plt.bar(left_borders, height=freqs, align='edge', 
        #             color='blue', label='frequences')
        # plt.title(f'Frequency bar plot (1/k = {1/k})')  
        # plt.legend()
        # plt.show()

    return is_rejected 
