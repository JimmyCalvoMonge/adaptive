import numpy as np

class MDP():
    
    """
    Implements a simple Markov Decision Process.
    With discrete actions and states spaces.
    Finding optimal policies for the process.

    References:
    - https://www.math.leidenuniv.nl/~kallenberg/Lecture-notes-MDP.pdf
    Markov Decision Processes
    Lodewijk Kallenberg
    University of Leiden

    - http://stanford.edu/~ashlearn/RLForFinanceBook/book.pdf
    Foundations of Reinforcement Learning with Applications in Finance
    Ashwin Rao, Tikhon Jelvis
    Stanford University

    """

    def __init__(self,
    S: list,
    A: list,
    rewards: list,
    trans_probs: list,
    horizon: int,
    delta: float,
    **kwargs):

        self.A = A # Actions space
        self.S = S # States space
        self.N = len(self.S)
        self.rewards = rewards # Immediate Reward
        self.trans_probs = trans_probs # Transition probabilities
        self.horizon = horizon # Planning horizon
        self.delta = delta # Discount factor
        self.logger = kwargs.get('logger', None)

        """
        A a list of actions.
        S a list of states.
        Define N = len(S)
        rewards: a list of self.horizon vectors in R^N, whose entries are functions that depend on a in A.
        trans_probs: a list of self.horizon matrices in R^{NxN}, whose entries are functions that depend on a in A.
        horizon: an integer.
        """
        self.verbose = kwargs.get('verbose', None)

    def _vprint(self, char):
        if self.logger:
            self.logger.info(char)
        else:
            if self.verbose:
                print(char)

    def fit_optimal_values(self, **kwargs):

        policies = []

        if 'init_point' in kwargs:
            x = kwargs.get('init_point')
        else:
            x = np.array([0]*self.N)

        x_history = [[x[h]] for h in self.S]
        # Backwards induction algorithm:

        self._vprint(f"""

        ##############################################
        ##### START DECISION PROCESS #################
        ##############################################


        Init point: 

        {x}

        """)


        for t in range(self.horizon):

            self._vprint("""
            ----- New iteration through horizon ----
            ----------------------------------------
            """)

            self._vprint(f"""
                Values of x (Value function vector) at this point:
                {x}                
            """)

            reward_step = self.rewards[t] #This is the vector with entries u_h^t(a) for h in S
            probs_step = self.trans_probs[t] # This is matrix with entries P_{hk}^t(a) for h,k in S

            all_prob_matrices = []
            for a in self.A:
                probs_matrix = probs_step[0,1](a)
                all_prob_matrices.append(probs_matrix)

            policies_step = {}
            vals = []
            
            for h in range(self.N):

                self._vprint(f"""
                >>>>>>>>>>>>>>>>>> h:{h} <<<<<<<<<<<<<<<<<<
                """)

                rwrds = [reward_step[h](a) for a in self.A]
                val_funcs = [self.delta*sum([probs_step[h,k](a)*x[k] for k in range(self.N)]) for a in self.A]

                self._vprint(f"""
                Rewards: 
                {rwrds}
                Delta*sums:
                {val_funcs}
                """)

                values = [reward_step[h](a) + self.delta*sum([probs_step[h,k](a)*x[k] for k in range(self.N)]) for a in self.A]
                self._vprint(f"""
                values: {values} <==================
                """)

                max_val = np.nanmax(np.array(values))
                max_arg = self.A[values.index(max_val)]

                self._vprint(f" max val: {max_val}")
                self._vprint(f" max val arg: {max_arg}")

                vals.append(max_val)
                policies_step[h] = max_arg

            x = np.array(vals)
            for h in self.S:
                x_history[h].insert(0,x[h])

            policies.append(policies_step)

        self.values = x
        self.values_history = x_history
        policies.reverse() # We reverse the policies, because the for loop started in the last time step in the horizon.
        self.policies = policies

        self._vprint(f"""
        First Policy:
        {self.policies[0]}
        """)

        self._vprint("""

        ##############################################
        ##### ENDED DECISION PROCESS #################
        ##############################################

        """)
