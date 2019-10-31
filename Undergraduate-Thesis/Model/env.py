import gym
import numpy as np
from scipy.stats import binom, beta

def _make_theta():
    biased = np.random.rand() < 0.5
    return beta(1, 1).rvs() if biased else 0.5
    # return beta(0.5, 0.5).rvs() if biased else 0.5
    

class CoinGame(gym.Env):
    """Remembering coin flips.

    rounds: number of rounds (observations)
    memory_cost: cost of remembering an observaton (adding to likelihood)
    error_cost: cost of incorrectly predicting whether or not the coin is biased
    n: number of flips per observation
    make_theta: a function that returns the coin probability
    """
    def __init__(self, rounds=20, memory_cost=1, error_cost=50, n=6, make_theta=None):
        super().__init__()
        self.rounds = rounds
        self.memory_cost = abs(memory_cost)
        self.error_cost = abs(error_cost)
        self.n = n
        self.make_theta = make_theta or _make_theta
        
        self.grid_size = 1001
        self.theta_grid = np.linspace(0, 1, self.grid_size)[1:-1]
        self.log_likelihood = binom(self.n, self.theta_grid).logpmf
        assert self.theta_grid[self.grid_size // 2 - 1] == 0.5
        # For now, we assume a flat prior for h1
        # self.h1_prior = np.ones(self.grid_size) / self.grid_size
        self.h1_prior = beta(*prior).pdf(self.theta_grid)
        self.h1_prior /= self.h1_prior.sum()
        self.reset()

    def _reset(self):
        self.round = 0
        self.theta = self.make_theta()
        self.coin = binom(self.n, self.theta)
        # Track binomial likelihood for many possible theta.
        # Posterior for each possible theta, only including remembered observations.
        self.belief = np.zeros_like(self.theta_grid)
        return self._observe()

    def _observe(self):
        self._obs = self.coin.rvs()
        return self._obs

    def _step(self, action):
        info = {}

        if action == 0:  # don't remember
            reward = 0

        elif action == 1:  # remember
            reward = - self.memory_cost
            self.belief += self.log_likelihood(self._obs)

        assert self.round < self.rounds
        self.round += 1
        done = self.round == self.rounds
        if done:
            # On the final round, agent guesses between H0 and H1,
            # receiving a penalty if incorrect.
            r, guess_info = self._guess()
            info.update(guess_info)
            reward += r

        obs = self._observe()
        info['obs'] = obs
        return obs, reward, done, info

    def _guess(self):
        h1_likelihood = (self.h1_prior * np.exp(self.belief)).sum()
        h0_likelihood = np.exp(self.belief[self.grid_size // 2 - 1])
        if np.allclose([h1_likelihood], [h0_likelihood]):
            guess = np.random.rand() > 0.5
        else:
            guess = h1_likelihood > h0_likelihood
        truth = self.theta != 0.5
        reward = 0 if guess == truth else - self.error_cost
        return reward, {'guess': guess, 'truth': truth}

