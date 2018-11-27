import numpy as np 
from scipy.stats import anderson
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from math import exp, sqrt
from scipy.special import iv, gamma

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
        a2 += (-1)**j * gamma(j + .5) * (4*j + 1) / gamma(.5) * gamma(j + 1) * \
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
    empiric_freqs, interval_width = _freqs(seq, lower_bound, upper_bound, k, normalized=True)
    empiric_prob_row = np.cumsum(empiric_freqs)

    left_bounds = np.arange(k) * interval_width

    # calculate value of statistic
    all_bounds = np.append(left_bounds, [(k + 1)*interval_width], -1)
    d_minus = []
    d_plus = []
    i = 0
    for x in left_bounds:
        if i > 0:
            d_minus.append(F(x) - empiric_prob_row[i - 1])
            d_plus.append(empiric_prob_row[i] - F(x))
        else:
            d_minus.append(F(x)) 
            d_plus.append(empiric_prob_row[i] - F(x))
        if verbose:
            print('... i = %2d | x = % 5.3f | F(x) = % 5.3f | freq = % 5.3f | d_plus = % 5.3f | d_minus = % 5.3f' \
            % (i, x, F(x), empiric_prob_row[i] if i > 0 else 0, d_plus[-1], d_minus[-1]))
        i += 1
    
    d = max(d_minus + d_plus)
    s = (6 * n * d + 1) / (6 *np.sqrt(n) )
    if verbose:
        print('...\n... Dmax = %f\n...' % (d)) 

    p = 1 - _K(s)

    is_rejected = p <= alpha

    return is_rejected, s, p

def kramer_mizes(seq, F, alpha=.05):
    """
    Statistical test of whether a given sample of data is drawn from 
    a given probability distribution.

    Inputs:
    - seq: Array
    - F: Cummulative distribution function (a.k.a cdf)
    - alpha: Level of significance

    Outputs:
    - is_rejected: Boolean; True if rejected
    - s: Value of statistic
    - p: Value of probability
    """
    sort_seq = sorted(seq)
    n = len(sort_seq)
    s = _getKramerStatistic_(sort_seq, n, F)
    p = _getKramerProbability_(s)
    is_rejected = s <= alpha
    return is_rejected, s, p


def _getKramerStatistic_(seq, n, F):
    """
    Calculate statistic for Kramer-Mizes criteria
    """
    S = 0.0
    for i in range(n):
        S += (F(seq[i]) - (2 * i - 1) / (2 * n)) ** 2
    return 1 / (12 * n) + S


def _getKramerProbability_(stat):
    """
    Calculate probability for Kramer-Mizes criteria
    """
    a1 = 0.0
    j = 0
    part1 = part2 = part3 = 1.0

    while j < 5 and part1 != 0.0 and part2 != 0.0 and part3 != 0.0:
        part1 = (gamma(j + 1 / 2) * sqrt(4 * j + 1)) / (gamma(1 / 2) * gamma(j + 1))
        part2 = exp(-(4 * j + 1) ** 2 / (16 * stat))
        part3 = iv(-1 / 4, (4 * j + 1) ** 2 / (16 * stat)) - iv(1 / 4, (4 * j + 1) ** 2 / (16 * stat))
        a1 += part1 * part2 * part3
        j += 1
    a1 *= 1 / sqrt(2 * stat)
    return 1 - a1


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
    for k in range(-10, 10, 1):
        p += (-1)**k * np.exp(-2 * k**2 * s**2)
    return p 

    


    

    