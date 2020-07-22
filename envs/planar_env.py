import pprint
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
import torch

class PlanarEnv(gym.Env):
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
        self.goal_state_y = 1.0
        # TODO hard version, need to move away from goal
        # self.goal_state_y = 0.0

        self.goal_state = np.array([self.goal_state_x, self.goal_state_y])
        self.low_state = np.array([self.min_state_x, self.min_state_y], dtype=np.float32)
        self.high_state = np.array([self.max_state_x, self.max_state_y], dtype=np.float32)
        self.low_action = np.array([self.min_action, self.min_action], dtype=np.float32)
        self.high_action = np.array([self.max_action, self.max_action], dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.low_action, high=self.high_action, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=self.low_action, high=self.high_state, dtype=np.float32)
        self.wall_x_left = 0.45
        self.wall_x_right = 0.55
        self.wall_y_top = 0.7
        self.wall_y_bottom = 0.6

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
        return new_states, None, None

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

        if vel_x > 0:
            if x_state_old <= self.wall_x_left and x_state_new >= self.wall_x_left:
                if y_state_new >= self.wall_y_top or y_state_new <= self.wall_y_bottom: 
                    x_state_new = x_state_old

        if vel_x < 0:
            if x_state_old >= self.wall_x_right and x_state_new <= self.wall_x_right:
                if y_state_new >= self.wall_y_top or y_state_new <= self.wall_y_bottom: 
                    x_state_new = x_state_old

        return np.array([x_state_new, y_state_new])

    def draw_walls(self):
      plt.axvline(0.5, self.unwrapped.wall_y_top + 0.01, 1, lw=10, color="gray", alpha=0.7)
      plt.axvline(0.5, 0, self.unwrapped.wall_y_bottom - 0.01, lw=10, color="gray", alpha=0.7)

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


if __name__ =='__main__':
    env = PlanarEnv()
    states = []
    state = env.reset()
    done = False
    while not done:
        action = env.sample_action()
        state, reward, done, _ = env.step(action)
        states.append(state)

    states = np.stack(states)
    increment = 1/states.shape[0]
    alpha = 0.0
    for i in range(states.shape[0]):
        plt.scatter(states[i, 0], states[i, 1], color="b", alpha=min(1, alpha))
        alpha += increment

    env.draw_walls()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    plt.plot()