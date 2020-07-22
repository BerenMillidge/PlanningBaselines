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
import scipy.signal as signal
from agent import Agent

class PIPlanner(Agent):
    def __init__(
        self,
        env,
        action_size,
        ensemble_size,
        num_candidates,
        plan_horizon,
        lambda_,
        noise_mu,
        noise_sigma,
        reward_measure = None,
        device="cpu"
    ):
        super().__init__()
        self.action_size = action_size
        self.num_candidates = num_candidates
        self.plan_horizon = plan_horizon
        self.ensemble_size = ensemble_size
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma=noise_sigma
        self.reward_measure = reward_measure
        self.device = device
        self.action_trajectory= torch.zeros([self.plan_horizon, self.action_size]).to(self.device)
        self.env=copy.deepcopy(env)
        self.times_called = 0

    def set_action_noise(self, action_noise):
      self.action_noise = action_noise

    def _add_action_noise(self, action):
        if self.action_noise is not None and self.action_noise > 0.0:
            action += np.random.normal(0, self.action_noise, action.shape)
        return action

    def real_env_rollout(self, current_state, noise):
      current_state = current_state.cpu()
      noise = noise.cpu()
      costs = torch.zeros([self.ensemble_size, self.num_candidates,self.action_size])
      for j in range(self.ensemble_size):
          for k in range(self.num_candidates):
              s = self.env.reset()
              self.env._env.state = self.state_from_obs(current_state.numpy())
              for t in range(self.plan_horizon):
                  action = self.action_trajectory[t].cpu() + noise[j,k,t,:]
                  s, reward, _ = self.env.step(action)
                  costs[j,k,:] += reward.cpu()
      return None, costs.to(self.device)

    def _perform_rollout(self, current_state, actions):
      state_size = current_state.shape[0]
      T = self.plan_horizon + 1
      states = [torch.empty(0)] * T

      returns = torch.zeros(self.num_candidates).float().to(self.device)

      current_state = current_state.unsqueeze(dim=0)
      current_state = current_state.repeat(self.num_candidates, 1)
      states[0] = current_state
      for t in range(self.plan_horizon):
            new_state,rewards, dones = self.env.dynamics_functions(states[t], actions[t])
            states[t + 1] = new_state
            
            _new_states = states[t + 1].view(-1, state_size)
            _states = states[t].view(-1, self.num_actions)
            if rewards is None:
                rewards = self.reward_measure(_new_states, _states)
                returns += rewards
      states = torch.stack(states[1:], dim=0)
      return states, -returns


    def SG_filter(self, action_trajectory):
      WINDOW_SIZE = 5
      POLY_ORDER = 3
      return torch.tensor(signal.savgol_filter(action_trajectory, WINDOW_SIZE,POLY_ORDER,axis=0))

    def forward(self, current_state):
      if not torch.is_tensor(current_state):
          current_state = torch.from_numpy(state).float().to(self.device)
      noise = torch.randn([self.plan_horizon, self.num_candidates,self.action_size]) * self.noise_sigma
      noise = noise.to(self.device)
      """ costs: [Ensemble_size, num_candidates] """
      states, costs = self._perform_rollout(current_state,noise)
      costs = costs /torch.mean(torch.abs(costs))
      """ beta is for numerical stability. Aim is that all costs before negative exponentiation are small and around 1 """
      beta = torch.min(costs)
      costs = torch.exp(-(1/self.lambda_ * (costs - beta)))
      eta = torch.mean(costs) + 1e-10
      """ weights: [Ensemble_size, num_candidates] """
      weights = (1/eta) * costs
      print("weights: ", weights)
      weights = weights.unsqueeze(1).repeat(1,self.action_size)
      """ Multiply weights by noise and sum across time dimension """
      add = torch.stack([torch.sum(weights * noise[t,:,:],dim=0) for t in range(self.plan_horizon)])
      self.action_trajectory += self.SG_filter(add.cpu()).to(self.device)
      action = self.action_trajectory[0] * 0.02
      """ Move forward action trajectory by 1 in preparation for next time-step """
      self.action_trajectory = torch.roll(self.action_trajectory,-1)
      self.action_trajectory[self.plan_horizon-1] = 0
      a =  self._add_action_noise(action.detach().cpu().numpy())
      print("final action: ", a)
      return a
