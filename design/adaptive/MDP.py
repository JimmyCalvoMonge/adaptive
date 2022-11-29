import numpy as np

class MDP():

    def __init__(self, S: list, A: list, rewards: list, trans_probs: list, horizon: int, delta: float):

        self.A = A # Actions space
        self.S = S # States space
        self.N = len(self.S)
        self.rewards = rewards # Immediate Reward
        self.trans_probs = trans_probs # Transition probabilities
        self.horizon = horizon # Planning horizon
        self.delta = delta # Discount factor

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

        verbose = False
        if 'verbose' in kwargs:
            if isinstance(kwargs.get('verbose'), bool) and kwargs.get('verbose'):
                verbose = True

        # Backwards induction algorithm:

        for t in range(self.horizon):

            reward_step = self.rewards[t] #This is the vector with entries u_h^t(a) for h in S
            probs_step = self.trans_probs[t] # This is matrix with entries P_{hk}^t(a) for h,k in S

            policies_step = {}
            vals = []
            
            for h in self.S:

                values = [reward_step[h](a) + self.delta*sum([probs_step[h,k](a)*x[k] for k in range(self.N)]) for a in self.A]
                max_val = np.max(np.array(values))
                if verbose:
                    print(f"max val at step {t}: {max_val} and state {h}")
                vals.append(max_val)
                policies_step[h] = self.A[np.argmax(values)]
            
            x = np.array(vals)
            policies.append(policies_step)

        self.values = x
        self.policies = policies
        self.first_policies = policies[0]
