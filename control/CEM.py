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


class CEMAgent(Agent):
    def __init__(
        self,
        env,
        num_actions,
        ensemble_size,
        device,
        sac_buffer=None,
        action_noise=None,
        plan_horizon=12,
        optimisation_iters=10,
        num_candidates=1000,
        top_candidates=100,
        cem_alpha=0.0,
        reward_measure=None,
        expl_measure=None,
        use_mean=True,
        logger=None,
    ):
        super().__init__()
        self.env = env
        self.num_actions = num_actions
        self.ensemble_size = ensemble_size
        self.sac_buffer = sac_buffer
        self.action_noise = action_noise

        self.plan_horizon = plan_horizon
        self.optimisation_iters = optimisation_iters
        self.num_candidates = num_candidates
        self.top_candidates = top_candidates
        self.cem_alpha = cem_alpha

        self.reward_measure = reward_measure
        self.expl_measure = expl_measure
        self.use_exploration = False if expl_measure is None else True
        self.use_reward = False if reward_measure is None else True
        self.use_mean = use_mean
        self.logger = logger
        self.device = device
        self.to(device)

    def set_use_reward(self, use_reward):
        self.use_reward = use_reward

    def set_use_exploration(self, use_exploration):
        self.use_exploration = use_exploration

    def set_action_noise(self, action_noise):
        self.action_noise = action_noise

    def forward(self, state):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().to(self.device)
        action_mean = torch.zeros(self.plan_horizon, 1, self.num_actions).to(self.device)
        action_std = torch.ones(self.plan_horizon, 1, self.num_actions).to(self.device)
        action_mean, _ = self._run_cem(state, action_mean, action_std)
        action = action_mean[0].squeeze(dim=0).cpu().detach().numpy()
        action = self._add_action_noise(action)
        return action

    def forward_init(self, state, action_mean, action_std, return_values=False):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().to(self.device)
        if return_values:
            action_mean, action_stds, states = self._run_cem(state, action_mean, action_std, return_values=return_values)
        else:
            action_mean, _ = self._run_cem(state, action_mean, action_std, return_values=return_values)
        action = action_mean[0].squeeze(dim=0).cpu().detach().numpy()
        action = self._add_action_noise(action)
        if return_values:
            
            return action, action_stds, states
        else:
            return action

    def _run_cem(self, state, action_mean, action_std, return_values=False):
        if self.logger is not None:
            action_means = []
            action_stds = []
            states = []
            d_stds = []

        prev_action_std = action_std
        state_size = state.size(0)

        for _ in range(self.optimisation_iters):
            actions = action_mean + action_std * torch.randn(
                self.plan_horizon, self.num_candidates, self.num_actions, device=self.device
            )
            states, returns = self._perform_rollout(state, actions)

            """
            returns = torch.zeros(self.num_candidates).float().to(self.device)
            if self.expl_measure is not None and self.use_exploration:
                expl_bonus = self.expl_measure(delta_means, delta_vars)
                returns += expl_bonus

            if self.reward_measure is not None and self.use_reward:
                _states = states.view(-1, state_size)
                _actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1)
                _actions = _actions.view(-1, self.num_actions)

                rewards = self.reward_measure(_states, _actions)
                rewards = rewards.view(self.plan_horizon, self.ensemble_size, self.num_candidates)
                rewards = rewards.mean(dim=1).sum(dim=0)
                returns += rewards
            """

            new_action_mean, new_action_std = self._fit_gaussian(actions, returns)
            action_mean = self.cem_alpha * action_mean + (1 - self.cem_alpha) * new_action_mean
            action_std = self.cem_alpha * action_std + (1 - self.cem_alpha) * new_action_std

            d_std = torch.abs(action_std - prev_action_std).mean(0).mean(0)
            prev_action_std = action_std

            if self.logger is not None:
                action_means.append(action_mean)
                action_stds.append(action_std)
                d_stds.append(d_std)

        if self.logger is not None:
            action_means = torch.stack(action_means)
            action_stds = torch.stack(action_stds)
            d_stds = torch.stack(d_stds)
            self.logger.log_cem_stats(action_means, action_stds, d_stds)

        if return_values:
            return action_mean, action_std, states
        else:
            return action_mean, action_std

    def _perform_rollout(self, current_state, actions):
        state_size = current_state.size(0)
        T = self.plan_horizon + 1
        states = [torch.empty(0)] * T
 
        returns = torch.zeros(self.num_candidates).float().to(self.device)

        current_state = current_state.unsqueeze(dim=0)
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

    def _fit_gaussian(self, actions, returns):
        returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)
        _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False)
        best_actions = actions[:, topk.view(-1)].reshape(
            self.plan_horizon, self.top_candidates, self.num_actions
        )
        action_mean, action_std_dev = (
            best_actions.mean(dim=1, keepdim=True),
            best_actions.std(dim=1, unbiased=False, keepdim=True),
        )
        return action_mean, action_std_dev

    def _add_action_noise(self, action):
        if self.action_noise is not None and self.action_noise > 0.0:
            action += np.random.normal(0, self.action_noise, action.shape)
        return action

    @staticmethod
    def _kl_divergence(q_mu, q_std, p_mu, p_std):
        return kl_divergence(Normal(q_mu, q_std), Normal(p_mu, p_std))

