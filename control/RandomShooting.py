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
from agent import Agent

class RandomShootingPlanner(Agent):
    def __init__(self, env,ensemble_size, action_size,plan_horizon, num_candidates, action_noise_sigma, reward_measure, exploration_measure =None, discount_factor=1,device='cpu'):
        super().__init__()
        self.env = env
        self.ensemble_size = ensemble_size
        self.action_size = action_size
        self.plan_horizon = plan_horizon
        self.num_candidates = num_candidates
        self.action_noise_sigma = action_noise_sigma
        self.reward_measure = reward_measure
        self.exploration_measure = exploration_measure
        self.discount_factor = discount_factor
        self.device=device
        self.to(self.device)
        if self.discount_factor <1:
            self.discount_factor_matrix = self._initialize_discount_factor_matrix()


    def _add_action_noise(self, action):
        if self.action_noise is not None and self.action_noise > 0.0:
            action += np.random.normal(0, self.action_noise, action.shape)
        return action

    def set_action_noise(self, action_noise):
      self.action_noise = action_noise

    def _initialize_discount_factor_matrix(self):
        discounts = np.zeros([self.plan_horizon,1,1,1])
        for t in range(self.plan_horizon):
            discounts[t,:,:,:] = self.discount_factor ** self.plan_horizon
        discounts=torch.from_numpy(discounts).repeat(1,self.ensemble_size, self.num_candidates, 1).to(self.device)
        return discounts

    def _perform_rollout(self, current_state, actions):
      state_size = current_state.size(0)
      T = self.plan_horizon + 1
      states = [torch.empty(0)] * T

      returns = torch.zeros(self.num_candidates).float().to(self.device)

      current_state = current_state.unsqueeze(dim=0).to(self.device)
      current_state = current_state.repeat(self.num_candidates, 1)
      print(current_state.shape)
      states[0] = current_state

      # actions = actions.unsqueeze(0)
      # actions = actions.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)

      for t in range(self.plan_horizon):
        new_state,rewards, dones = self.env.dynamics_functions(states[t], actions[t])
        states[t + 1] = new_state
        
        _new_states = states[t + 1].view(-1, state_size)
        _states = states[t].view(-1, self.num_actions)
        if rewards is None:
            rewards = self.reward_measure(_new_states, _states)
            returns += rewards
      states = torch.stack(states[1:], dim=0)
      return states, returns

    def forward(self, state):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().to(self.device)
        action_mean = torch.zeros(self.plan_horizon, 1, self.action_size).to(self.device)
        action_std_dev = torch.ones(self.plan_horizon, 1, self.action_size).to(self.device) * self.action_noise_sigma
        actions = action_mean + action_std_dev * torch.randn(self.plan_horizon, self.num_candidates, self.action_size, device=self.device)
        states, returns = self._perform_rollout(state, actions)
      

        if self.discount_factor <1:
            returns *= self.discount_factor_matrix

        returns = torch.where(
            torch.isnan(returns), torch.zeros_like(returns), returns
        )

        _, topk = returns.topk(
            1, dim=0, largest=True, sorted=True
        )

        best_actions = actions[:, topk.view(-1)].reshape(
            self.plan_horizon, 1, self.action_size
        )
        print("actions: ", best_actions.shape)
        return best_actions[0,:,:].squeeze(dim=0).cpu().detach().numpy()