import numpy as np 
from scipy.stats import anderson
from scipy.special import gamma as g 
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def anderson_darlin(seq, F, alpha=.05, n = None, verbose = True):
    """
    Statistical test of whether a given sample of data is drawn from 
    a given probability distribution. 

    Inputs:
    - seq: Array of integers
    - F: Cumulative distribution function
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

    seq_sorted = sorted(seq)

    s = 0
    for i in range(1, n + 1):
        f = F(seq_sorted[i - 1])
        s += (2*i - 1) * np.log(f) / (2*n) + \
        (1 - (2*i - 1) / (2*n)) * np.log(1 - f) 
    s = -n - 2*s 

    a2 = 0
    for j in range(15):
        a2 += (-1)**j * g(j + .5) * (4*j + 1) / g(.5) * g(j + 1) * \
        np.exp((4*j + 1)**2 * np.pi**2 / (-8 * s)) * \
        integrate.quad(lambda y: np.exp(s / (8 * (y**2 + 1)) - (4*j + 1)**2 \
                                * np.pi**2 * y**2 / (8 * s)), 0, np.inf)[0]

    a2 *= np.sqrt(2 * np.pi) / s 
    p = 1 - a2 

    is_rejected = p <= alpha

    if verbose:
        print(f'Significance level S = {s}')
        print(f'a2 = {a2}')
        print(f'p = {p}')
    return is_rejected


def kolmogorov(seq, F, alpha=.05, k = None, verbose = True):
    """
    Statistical test of whether a given sample of data is drawn from 
    a given probability distribution. 

    Inputs:
    - seq: Array of values from distribution
    - F: Cumulative distribution function (aka cdf)
    - alpha: Float desirible level of significance
    - k: Integer number of regions (default is None)
    - verbose: Boolean; If set to true then print logs

    Outputs:
    - Boolean; If hypothesis is rejected
    - s; Value of statistic
    - p; Probability
    """
    # build empirical probability row
    n = len(seq)
    if k is None:
        k = int(5 * np.log(n)) 
    lower_bound = 0
    upper_bound = max(seq)
    interval_width = (upper_bound - lower_bound) / k
    sorted_seq = sorted(seq)
    left_bounds = np.arange(k) * interval_width

    # calculate value of statistic
    all_bounds = np.append(left_bounds, [(k + 1)*interval_width], -1)
    d_minus = []
    d_plus = []
    i = 1
    for el in sorted_seq:
        d_plus.append( float(i) / n - F(el) )
        d_minus.append( F(el) - float(i - 1) / n )
        i += 1
    
    d = max(d_minus + d_plus)
    s = (6 * k * d + 1) / (6 *np.sqrt(k) )
    if verbose:
        print('...\n... Dmax = %f\n...' % (d)) 

    p = 1 - _K(s)

    is_rejected = p <= alpha

    return is_rejected, s, p


def _freqs(seq, lower_bound, upper_bound, k, normalized=False):
    """
    Return frequences of sequence values

    Inputs:
    - seq: Array of integers. Observable sequence
    - lower_bound: Integer lower bound of the domain
        of generator values
    - upper_bound: Integer upper bound of the domain
        of generator values
    - k: Integer number of intervals

    Outputs:
    - freqs: Array of occurences of values in each region 
    with region_width
    - region_width: Float width of regions
    """
    freqs = []
    region_width = (upper_bound - lower_bound) / k 

    for i in range(k):
        low = lower_bound + i * region_width
        high = lower_bound + i * region_width + region_width
        freqs.append( np.logical_and(seq >= low, seq < high).sum() )

    # because last interval has '[a;b]' - bounds, not '[a,b)'
    freqs[-1] += 1

    if normalized:
        freqs = np.array(freqs) / len(seq)

    return np.array(freqs), region_width
    
    
def _K(s):
    """
    Auxiliary function for integral calculating within Kolmogorov's
    statistical test

    Inputs:
    - s: Float value of statistic

    Outputs:
    - p: Float probability of Kolmogorov distribution
    """
    p = 0
    for k in range(-100, 100 + 1, 1):
        p += (-1)**k * np.exp(-2 * k**2 * s**2)
    return p 

    


    

    