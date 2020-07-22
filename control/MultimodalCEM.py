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



class MultiModalCEMPlanner(Agent):
    def __init__(self, env,ensemble_size, action_size,plan_horizon, num_candidates, optimisation_iters, top_candidates,num_modes,action_noise_sigma, reward_measure,device='cpu'):
      super().__init__()
      self.env = env
      self.ensemble_size = ensemble_size
      self.action_size = action_size
      self.plan_horizon = plan_horizon
      self.num_candidates = num_candidates
      self.optimisation_iters = optimisation_iters
      self.top_candidates = top_candidates
      self.action_noise_sigma = action_noise_sigma
      self.reward_measure = reward_measure
      self.num_modes = num_modes
      self.device=device

    def set_action_noise(self, action_noise):
        self.action_noise = action_noise

    def _add_action_noise(self, action):
        if self.action_noise is not None and self.action_noise > 0.0:
            action += np.random.normal(0, self.action_noise, action.shape)
        return action

    def _perform_rollout(self, current_state, actions):
      state_size = current_state.size(0)
      T = self.plan_horizon + 1
      states = [torch.empty(0)] * T

      returns = torch.zeros(self.num_candidates).float().to(self.device)

      current_state = current_state.unsqueeze(dim=0).to(self.device)
      current_state = current_state.repeat(self.num_candidates, 1)
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

    def sample_action(self):
      idx = Categorical(self.pis).sample()
      action_dist = Normal(loc=self.action_means[idx],scale=self.action_vars[idx])
      action = action_dist.sample()
      #print("sampled action", action.shape)
      return action

    def forward(self, state):
      if not torch.is_tensor(state):
        print("state: ", state)
        state = torch.from_numpy(state).float().to(self.device)
      self.action_means = torch.zeros([self.num_modes,self.plan_horizon,1,self.action_size]).to(self.device)
      self.action_vars = torch.ones([self.num_modes,self.plan_horizon,1,self.action_size]).to(self.device) * self.action_noise_sigma
      self.pis = torch.ones([self.num_modes]).to(self.device) / self.num_modes
      self.candidates_per_mode = self.num_candidates // self.num_modes
      #will be slightly less if rounding issues
      self.num_candidates = self.candidates_per_mode * self.num_modes
      for i in range(self.optimisation_iters):
        #initialize actions
        actions = self.action_means[0] + self.action_vars[0] * torch.randn(self.plan_horizon, self.candidates_per_mode, self.action_size, device=self.device)
        for n in range(1,self.num_modes):
          act = self.action_means[n] + self.action_vars[n] * torch.randn(self.plan_horizon, self.candidates_per_mode, self.action_size, device=self.device)
          actions = torch.cat((actions, act),dim=1)
        # perform rollout
        #print("actions: ", actions.shape)
        states, returns = self._perform_rollout(state, actions)
        returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)
        topk_vals, topk_idxs = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False)
        # assign the topk values for each mode and fit the gaussian
        topk_idxs = topk_idxs.view(-1)
        for n in range(self.num_modes):
          #compute the mode indices and then fit the pi, mean, and variance
          upper_idxs = topk_idxs[topk_idxs > (n*self.candidates_per_mode)].clone()
          mode_idxs = upper_idxs[upper_idxs < (n+1)*self.candidates_per_mode].clone()
          mode_actions = actions[:,mode_idxs,:].clone()
          print("mode actions: ",mode_actions.shape )
          self.pis[n] = mode_actions.shape[1] / self.num_candidates
          mean_actions = torch.mean(mode_actions,dim=1,keepdim=True)
          #print("mean actions: ", mean_actions.shape)
          self.action_means[n] = mean_actions
          self.action_vars[n] = torch.std(mode_actions,dim=1,unbiased=False,keepdim=True)

        print("pis: ", self.pis)
      #finally sample action
      action = self.sample_action()[0].squeeze(dim=0).cpu().detach().numpy()
      action = self._add_action_noise(action)
      #print(action.shape)
      #print("final action: ", action)
      return action


        
        





