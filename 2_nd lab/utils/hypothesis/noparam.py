import numpy as np 
from scipy.stats import anderson
from scipy.special import gamma as g 
import scipy.integrate as integrate

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
