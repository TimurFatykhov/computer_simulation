import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt


def permutation_test(seq, alpha = 0.05, n = None, verbose=True):
    """
    Randomness test via number of sign permutations.
    
    Inputs:
    - seq: Array of integer
    - alpha:
    
    Outputs:
    - boolean: If permutation test is passed then true, 
    otherwise false
    """
    if n is None:
        n = len(seq)
    elif len(seq) < n:
        raise ValueError(f'Length of sequence is smaller than \
        n = {n}')
    
    seq = seq[:n]
    
    # count number of sign permutations in sequence
    q = 0
    for i in range(len(seq) - 1):
        if seq[i] > seq[i + 1]:
            q += 1
            
    # compute confidence interval
    u_a = stats.norm.ppf(1 - alpha)
    
    eps = u_a * np.sqrt(n) / 2
    left_bound = q - eps
    right_bound = q + eps
    
    if verbose:
        print('Eps: %.3f \nq: %d \nn/2: %.1f \nBounds: [%.3f, %.3f]\n' \
            % (eps, q, n/2, left_bound, right_bound))
    
    return n/2 <= right_bound and n/2 >= left_bound 


def frequency_test(seq, 
    m = None, 
    k = 20, 
    n = None, 
    verbose=True, 
    alpha = 0.05):
    """
    Randomness test via bar chart
    
    Inputs:
    - seq: Array of integer
    - m: Integer modulus of generator
    - k: Integer number of regions
    - n: Integer length of sequence
    - verbose: Boolean; If set to true then print along the process
    - alphs: Float level of significance
    
    Outputs:
    - Boolean; If all tests passed successfully, then return true 
    and false otherwise
    If verbose is set to true then print logs and bar plots
    """
    if n is None:
        n = len(seq)
    
    if len(seq) < n:
        raise ValueError(f'Length of sequence is smaller than n = {n}')
    seq = np.array(seq[:n])
    
    if m is None:
        m = max(seq) + 1e-5
    
    # mean, var
    mean = seq.mean()
    var = seq.var()
        
    # frequency vector 
    left_borders, widths = [], []
    freq = []
    region_size = m / k 
    for i in range(k):
        low = i * region_size
        high = i * region_size + region_size
        freq.append( np.logical_and(seq >= low, seq < high).sum()  )
        
        left_borders.append(low)
        widths.append(high - low)
    
    freq = np.array(freq) / n
    
    # freq interval tests
    freq_int_test = False

    u_a = stats.norm.ppf(1 - alpha/2)
    eps = (u_a / k) * np.sqrt((k-1)/n)
    freq_low = freq - eps
    freq_high = freq + eps
    freq_int_test = np.prod((1/k >= freq_low) * (1/k <= freq_high))

    if verbose:
        if freq_int_test:
            print('-------- (PASSED) --->\
             Frequency interval test passed successfully!\n')
        else:
            print('-------- (FAILED) --->\
             Frequency interval test failed! (Take a look on bar plot)\n')

    # expectation interval tests
    mean_int_test = False

    eps = u_a * np.sqrt(var/n)
    mean_low = mean - eps
    mean_high = mean + eps
    mean_int_test = m/2 >= mean_low and m/2 <= mean_high 

    if verbose:
        print(f'-------- m/2 = {m/2}')
        print(f'-------- expectation interval: [{mean_low} ; {mean_high}]')
        if mean_int_test:
            print('-------- (PASSED) --->\
             Expectation interval test passed successfully!\n')
        else:
            print('-------- (FAILED) --->\
             Expectation interval test failed!\n')

    # variance interval tests
    var_int_test = False 

    var_low = var * (n - 1) / stats.chi2.ppf(1 - alpha, n - 1)
    var_high = var * (n - 1) / stats.chi2.ppf(alpha, n - 1)
    var_int_test = m*m/12 >= var_low and m*m/12 <= var_high
    
    # print logs
    if verbose:
        print(f'-------- m^2/12 = {m*m/12}')
        print(f'-------- variance interval: [{var_low} ; {var_high}]')
        if var_int_test:
            print('-------- (PASSED) --->\
             Variance interval test passed successfully!')
        else:
            print('-------- (FAILED) --->\
             Variance interval test failed!')

    # show plot
    if verbose: 
        plt.ylim(-0.1, 0.3)
        plt.bar(left_borders, height=freq, 
                    width=widths, align='edge', 
                    color='blue', label='frequences')
        for i, (l, h) in enumerate(zip(freq_low, freq_high)):
            x = [left_borders[i], left_borders[i] + widths[i]]
            plt.plot(x, [l, l], 'r')
            plt.plot(x, [h, h], 'r') 
            plt.plot([x[0], x[0]], [l, h], 'r--') 
            plt.plot([x[1], x[1]], [l, h], 'r--') 
        plt.title(f'Frequency bar plot (1/k = {1/k})') 
        plt.plot([left_borders[0], left_borders[-1] + widths[-1]], \
                    [1/k, 1/k], 'c-', markersize=20, label='1 / k') 
        plt.legend()
        plt.show()

    return freq_int_test and mean_int_test and var_int_test


def complex_test(seq, r=4, K=8, low_cost=True, verbose=1):
    """
    Perform frequency and permutation tests for subsequences

    Inputs:
    - seq: Array of integers
    - r: Integer coefficient
    - K: Integer coefficient
    - verbose: Integer 0, 1, 2 or 3; level of verbose
    - low_cost: Boolean;

    Otputs:
    - Boolean; If tests for all subsequences passed successfully, 
    then return true and false otherwise
    If verbose is set to true then print logs
    """
    seq = np.array(seq)
    n = len(seq)
    t = (n - r) // r 
    mask = np.arange(t+1) * r 

    # if verbose == 0
    _verbose = [None, None]

    if verbose == 1:
        _verbose = [False, False]
    elif verbose == 2:
        _verbose = [False, True]
    elif verbose == 3:
        _verbose = [True, True]

    for i in range(r):
        subseq = seq[mask]

        # permutation test
        if verbose > 0:
            print(f'- Permutation testing subsequence #{i + 1}/{r}') 
        if permutation_test(subseq, verbose=_verbose[0], n=len(subseq)):
            if verbose > 0:
                print(f'- (PASSED) --->\
                Permutation test passed succesfully for \
                subsequence #{i + 1}/{r}') 
        else:
            if verbose > 0:
                print(f'- (FAILED) --->\
                Permutation test failed for subsequence #{i + 1}/{r}') 
            if low_cost:
                return False 

        # frequency test
        if verbose > 0:
            print(f'- Frequency testing subsequence #{i + 1}/{r}') 
        if frequency_test(subseq, verbose=_verbose[1], k=K):
            if verbose > 0:
                print(f'- (PASSED) --->\
                Frequency test passed succesfully for \
                subsequence #{i + 1}/{r}') 
        else:
            if verbose > 0:
                print(f'- (FAILED) --->\
                 Frequency test failed for subsequence #{i + 1}/{r}') 
            if low_cost:
                return False 

        # print delimiter
        print(80*'-'+'\n')

        mask += 1
    
    return True


def _freqs(seq, lower_bound, upper_bound, n):
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
    
    return freqs, region_width
