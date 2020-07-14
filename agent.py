import os
import time
from copy import deepcopy
import copy
import math
import json
import random
from datetime import datetime
import itertools
import pprint
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
matplotlib.rcParams["axes.linewidth"] = 1.1

class Agent(nn.Module):
    """ Parent class for `Agent` objects """

    def forward(self, state):
        raise NotImplementedError

    def sample_episodes(self, env, num_episodes=1):
        return self._sample_episodes(env, num_episodes=num_episodes)

    def sample_record_episodes(self, env, buffer, num_episodes=1):
        return self._sample_episodes(env, num_episodes=num_episodes, buffer=buffer)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def _sample_episodes(self, env, num_episodes=1, buffer=None):
        rewards, steps, times = [], [], []

        for _ in range(num_episodes):
            epi_reward = 0
            epi_steps = 0
            epi_time = time.time()

            state = env.reset()
            done = False

            while not done:
                action = self(state)
                next_state, reward, done, _ = env.step(action)
                epi_reward += reward
                epi_steps += 1

                if buffer is not None:
                    mask = 1 if epi_steps == env.max_episode_steps else float(not done)
                    buffer.add(state, action, reward, next_state, mask)

                state = deepcopy(next_state)

            rewards.append(epi_reward)
            steps.append(epi_steps)
            times.append(time.time() - epi_time)

        return {"rewards": rewards, "steps": steps, "times": times}

    def __str__(self):
        return "<{}>".format(type(self).__name__)


class RandomAgent(Agent):
    def __init__(self, env, device):
        super().__init__()
        self._env = env
        self.to(device)

    def forward(self, state):
        return self._env.sample_action()

class RewardMeasure(object):
    def __init__(self, env, scale=1.0):
        self._env = env
        self.scale = scale

    def __call__(self, states, actions=None):
        return self._env.reward_function(states, actions) * self.scale