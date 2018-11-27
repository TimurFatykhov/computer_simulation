class Uni_generator():
    """
    Adaptive pseudorandom generator
    """
    def __init__(self, x0, x1, N = 600, m =100):
        """
        Inputs:
        - N: Integer size of generated sequence
        - m: Integer modulus of generator
        - x0: Integer first element of our sequence 
        - x1: Integer second element of our sequence
        - params: Dictionary of ranges for generator's parameters
            - a: range of a-values
            - b: range of b-values
            - c: range of c-values
        - low_cost: Boolean; if set to false then perform grid search,
        otherwise find first suitable parameters for receiving 
        desired_period
        - desired_perion: None, Integer; if low_cost is set to false 
        then have to be integer value
        - verbose: Boolean; if set to true then print T for each set 
        of params
            
        Outputs:
        - T: size of period
        - best_param: parameters for received size of period
        - seq: sequence with best parameters and biggest period
        """
        self.N = N
        self.m = m
        self.x0 = x0
        self.x1 = x1
        self._gen_params = {'a': 157, 'b': 246, 'c': 149, 'm': m}
        

    def _gen(self, a, b, c, N, m = None):
        """
        Generate pseudorandom sequence

        Inputs:
        - a: Integer first parameter of generator
        - b: Integer second parameter of generator
        - c: Integer third parameter of generator
        - N: Integer size of sequence
        - m: Integer modulus of generator

        Outputs:
        - seq: pseudorandom sequence
        """
        if m is None:
            m = self.m
        x0 = self.x0 
        x1 = self.x1 

        seq = [x0, x1]
        for _ in range(N - 2):
            x2 = (a * x1**3 + b * x1 + c * x0**2) % m 
            seq.append(x2)
            x0 = x1 
            x1 = x2
        return seq 


    def _period_of_seq(self, seq, w_size, verbose=False):
        """
        Calculate period of sequence
        
        Inputs:
        - seq: Array of int
        - w_size: Integer window size. 
        - verbose: Boolean; if set to true then print logs
        
        Outputs:
        - size: Integer size of period
        """
        window = seq[-w_size:]
        T = 0
        print(f'window : {window}')
        for i in range(len(seq) - w_size - 1, 0, -1):
            T += 1
            if i > len(seq) - w_size - 20 and verbose:
                print(f'our seq: {seq[i:i + w_size]}')
            if window == seq[i:i + w_size]:
                return T
        return len(seq)

    def set_params(self, params):
        """
        Explicitly set parameters of generator

        Inputs: 
        - params:
            - a: Integer first parameter of generator
            - b: Integer second parameter of generator
            - c: Integer third parameter of generator
        """
        self._gen_params = params


    def fit(self, params, low_cost=False, desired_period=None, verbose=False):
        """
        Train our generator (select best parameters from random grid)

        Inputs:
        - params:
            - a: Array of integers; possible values of first parameter 
            - b: Array of integers; possible values of second parameter 
            - c: Array of integers; possible values of third parameter 
        - low_cost: Boolean; If set to true then stop parameters selection 
        when desired period is achieved
        - desired_period: Integer size of period that is desired
        - verbose: Boolean; If set to true then print logs

        Outputs:
        - nothing, but print logs
        """

        if low_cost and desired_period is None:
            raise ValueError('Check low_cost and desired_period values.')

        num_of_sets = len(params['a']) * len(params['b']) * len(params['c'])
        
        self._best_res = 0
        self._gen_params = {}
        self._best_seq = None
        self._iter_num = 0
        T = 0
        for ia in params['a']:
            for ib in params['b']:
                for ic in params['c']:
                    self._iter_num += 1
                    seq = self._gen(ia, ib, ic, self.N)
                    T = self._period_of_seq(seq, w_size=2)
                    
                    if verbose:
                        print(f'ia: {ia} | ib: {ib} | ic: {ic} |---> T: {T}')
                    if T > self._best_res:
                        self._best_res = T
                        self._best_seq = seq
                        self._gen_params = {
                            'a' : ia,
                            'b' : ib,
                            'c' : ic,
                            'm' : self.m,
                        }
                    
                    if low_cost and T >= desired_period:
                        print(f'Desired perios is reached \
                        by {self._iter_num} of {num_of_sets} iterations')
                        print(f'T equal to {T}\n')
                        return
        if low_cost:
            print('Desired period is not reached!')
        return


    def generate(self, N = None, m = None, normalized = False):
        """
        Generate pseudorandom sequence via fitted generator.

        Inputs:
        - N: Integer length of desirible sequence
        - m: Integer modulus
        - normalized: Bool; If set to true then return sequence where each
        element belong to interval [0, 1], otherwise [0, m]

        Outputs:
        - seq: Integer array; Generated pseudorandom sequences
        """
        if self._gen_params is None:
            raise ValueError('Parameters of generator is not set. \
                                Use .train or .set_params before.')

        if m is None:
            m = self.m 

        a = self._gen_params['a']
        b = self._gen_params['b']
        c = self._gen_params['c']

        if N is None:
            N = self.N

        seq = self._gen(a, b, c, N + 2, m)[2:]

        if normalized:
            for i in range(len(seq)):
                seq[i] /= self.m  

        return seq