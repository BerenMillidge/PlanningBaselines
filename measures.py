import numpy as np 


class RewardMeasure(object):
    def __init__(self, env, scale=1.0):
        self._env = env
        self.scale = scale

    def __call__(self, states, actions=None):
        return self._env.reward_function(states, actions) * self.scale