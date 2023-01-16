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
        if 'logger' in kwargs:
            self.logger = kwargs.get('logger')

        """
        A a list of actions.
        S a list of states.
        Define N = len(S)
        rewards: a list of self.horizon vectors in R^N, whose entries are functions that depend on a in A.
        trans_probs: a list of self.horizon matrices in R^{NxN}, whose entries are functions that depend on a in A.
        horizon: an integer.
        """

    def fit_optimal_values(self, **kwargs):

        policies = []

        if 'init_point' in kwargs:
            x = kwargs.get('init_point')
        else:
            x = np.array([0]*self.N)

        x_history = [[x[h]] for h in self.S]
        # Backwards induction algorithm:

        for t in range(self.horizon):

            reward_step = self.rewards[t] #This is the vector with entries u_h^t(a) for h in S
            probs_step = self.trans_probs[t] # This is matrix with entries P_{hk}^t(a) for h,k in S

            all_prob_matrices = []
            for a in self.A:
                probs_matrix = probs_step[0,1](a)
                all_prob_matrices.append(probs_matrix)

            policies_step = {}
            vals = []
            
            for h in range(self.N):
                values = [reward_step[h](a) + self.delta*sum([probs_step[h,k](a)*x[k] for k in range(self.N)]) for a in self.A]
                #print(values)
                max_val = np.max(np.array(values))
                vals.append(max_val)
                policies_step[h] = self.A[np.argmax(values)]
            
            x = np.array(vals)
            for h in self.S:
                x_history[h].insert(0,x[h])

            policies.append(policies_step)

        #Css
        # self.logger.info(f"Css obtained in backwards induction: (with {self.horizon} steps)")
        css = []
        for pol in policies:
            css.append(pol[0])
        # self.logger.info(css)

        # if verbose:
        # for h in self.S:
            #print(f"Values V_{{{h}}}(t) for t=0,1,...,{self.horizon}+1:")
            #print(x_history[h])

        self.values = x
        self.policies = policies
        self.first_policies = policies[0]
