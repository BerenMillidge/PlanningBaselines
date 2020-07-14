import pprint
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
import torch
class PicheEnv(gym.Env):
  def __init__(self,device="cpu"):
    self.device = device
    self.min_action = -0.05
    self.max_action = 0.05
    self.init_x = 0.0
    self.init_y = 0.0
    self.min_state_x = 0.0
    self.min_state_y = 0.0
    self.max_state_x = 1.0
    self.max_state_y = 1.0
    self.goal_state_x = 1.0
    self.goal_state_y = 0.5
    # TODO hard version, need to move away from goal
    # self.goal_state_y = 0.0

    self.goal_state = np.array([self.goal_state_x, self.goal_state_y])
    self.low_state = np.array([self.min_state_x, self.min_state_y], dtype=np.float32)
    self.high_state = np.array([self.max_state_x, self.max_state_y], dtype=np.float32)
    self.low_action = np.array([self.min_action, self.min_action], dtype=np.float32)
    self.high_action = np.array([self.max_action, self.max_action], dtype=np.float32)
    print(self.low_action)
    print(self.high_action)
    print(self.low_action.shape)
    print(self.high_action.shape)
    self.action_space = spaces.Box(
        low=self.low_action, high=self.high_action, dtype=np.float32
    )
    self.observation_space = spaces.Box(low=self.low_action, high=self.high_state, dtype=np.float32)
    self.line = np.array([0.2, 0.8])

    self.seed()
    self.reset()

  def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]

  def reset(self):
      self.state = np.array([self.init_x, self.init_y], dtype=np.float32)
      return self.state

  def step(self, actions):
      new_state = self.dynamics(self.state, actions)
      reward = self.reward(new_state, self.state)
      self.state = new_state
      done = False
      return new_state, reward, done, {}

  def reward(self, new_state, state):
      new_dis = ((new_state - self.goal_state) ** 2).mean()
      dis = ((state - self.goal_state) ** 2).mean()
      # @TODO avoid division by zero
      r = 1 - (new_dis / (dis + 1e-8))
      # @TODO
      if new_dis < 0.05:
          r = 10.0
      return r

  def reward_function(self, new_state, state):
      if torch.is_tensor(state):
          state = state.detach().cpu().numpy()
          new_state = new_state.detach().cpu().numpy()
      new_dis = ((new_state - self.goal_state) ** 2).mean(axis=1)
      dis = ((state - self.goal_state) ** 2).mean(axis=1)
      # @TODO avoid division by zero
      r = 1 - (new_dis / (dis + 1e-8))
      # @TODO
      idxss = np.where(new_dis < 0.05)
      r[idxss] = 10
      r = torch.from_numpy(r).float().to(self.device)
      return r

  def batch_dynamics(self, states, actions):
      """ (batch_size, state_size) / (batch_size, action_size) """
      if torch.is_tensor(states):
          states = states.detach().cpu().numpy()
          actions = actions.detach().cpu().numpy()
      batch_size = states.shape[0]
      new_states = np.zeros((batch_size, self.observation_space.shape[0]))
      for i in range(batch_size):
          _states = states[i, :]
          _actions = actions[i, :]
          new_state = self.dynamics(_states, _actions)
          new_states[i, :] = new_state
      # TODO
      new_states = torch.from_numpy(new_states).float().to(self.device)
      return new_states

  def dynamics(self, state, actions):
      vel_x = actions[0]
      vel_y = actions[1]
      vel_x = np.clip(vel_x, self.min_action, self.max_action)
      vel_y = np.clip(vel_y, self.min_action, self.max_action)
      x_state_old = state[0]
      y_state_old = state[1]
      x_state_new = x_state_old + vel_x
      x_state_new = np.clip(x_state_new, self.min_state_x, self.max_state_x)
      y_state_new = y_state_old + vel_y
      y_state_new = np.clip(y_state_new, self.min_state_y, self.max_state_y)

      if state[0] <= 0.5 and x_state_new >= 0.5:
          if y_state_new >= self.line[0] and y_state_new <= self.line[1]:
              x_state_new = state[0] - 1e-3
      if state[0] >= 0.5 and x_state_new <= 0.5:
          if y_state_new >= self.line[0] and y_state_new <= self.line[1]:
              x_state_new = state[0] + 1e-3

      return np.array([x_state_new, y_state_new])
  def draw_walls(self):
    #l = mlines.Line2D([0.5, 0.5], self.line)
    plt.axvline(0.5, self.line[0],self.line[1], lw=10, color="gray", alpha=0.7)

  def sample_action(self):
      return self.action_space.sample()

  def draw_env(self,states):
    states = np.stack(states)
    increment = 1/states.shape[0]
    alpha = 0.0
    for i in range(states.shape[0]):
        plt.scatter(states[i, 0], states[i, 1], color="b", alpha=min(1, alpha))
        alpha += increment
    self.draw_walls()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    plt.plot()
    plt.show()

if __name__ == '__main__':
    env = PicheEnv()
    states = []
    state = env.reset()
    for i in range(100):
        action = env.sample_action()
        state, reward,done,info = env.step(action)
        states.append(state)
    env.draw_env(states)

