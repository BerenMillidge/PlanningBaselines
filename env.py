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
from envs.planar_env import *
from envs.piche_env import *
from envs.lunar_lander import LunarLander, LunarLanderContinuous
from envs.double_pendulum import SparseDoublePendulum
from envs.mountain_car import MountainCar
from envs.pendulum import PendulumEnv
from envs.acrobot import AcrobotEnv

class Env(object):
    def __init__(self, env_name, max_episode_steps, action_repeat=1, seed=None,device="cpu"):
        gym.logger.set_level(40)
        self.device = device
        self._env = self._get_env_object(env_name)
        self.max_episode_steps = max_episode_steps
        self.action_repeat = action_repeat
        if seed is not None:
            self._env.seed(seed)
        self._t = 0

    def reset(self):
        self._t = 0
        return self._env.reset()

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            state, reward_k, done, info = self._env.step(action)
            reward += reward_k
            self._t += 1
            if self._t == self.max_episode_steps:
                done = True

            if done:
                break

        return state, reward, done, info

    def render(self, mode="human"):
        self._env.render(mode)

    def close(self):
        self._env.close()

    def sample_action(self):
        return self._env.action_space.sample()

    def reward_function(self, new_state, old_state):
        return self._env.reward_function(new_state, old_state)

    def dynamics_functions(self, states, actions):
        return self._env.batch_dynamics(states, actions)

    def set_state(self, state):
        return self._env.set_state(state)

    def get_env_names(self):
        return ["toy", "piche","pendulum","double_pendulum","lunar_lander","lunar_lander_discrete","mountain_car"]

    def _get_env_object(self, env_name):
        if env_name == "toy":
            return PlanarEnv(self.device)
        elif env_name == "piche":
            return PicheEnv(self.device)
        elif env_name == "pendulum":
            return PendulumEnv()
        elif env_name == "double_pendulum":
            return SparseDoublePendulum()
        elif env_name == "lunar_lander":
            return LunarLanderContinuous()
        elif env_name == "lunar_lander_discrete":
            return LunarLander()
        elif env_name == "mountain_car":
            return MountainCar()
        else:
            try: 
                return gym.make(env_name)
            except:
                raise ValueError("Env name not recognised. You inputted: " + str(env_name))
            

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def unwrapped(self):
        return self._env