import math
import gym
import numpy as np
import torch

class MountainCar(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, goal_velocity=0, no_penalty=True,device="cpu"):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.goal_velocity = goal_velocity
        self.power = 0.0015
        self.no_penalty = no_penalty

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = gym.spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        self.max_reward = 1
        self.min_reward = 0

        self.seed()
        self.reset()
        self.device = device

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0
        if done:
            reward = 1.0

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def reward_function(self,states, actions):
        position, velocity = self.state 
        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = 0 
        if done:
            reward = 1.0
        return reward

    def batch_reward_function(self, states, actions):
        dones = bool(self.states[:,0] >= self.goal_position and self.states[:,1] >= self.goal_velocity)
        rewards = np.zeros(self.states.size(0),1)
        rewards[dones] = 1.0
        return rewards

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)
    
    def state_from_obs(self,obs):
        return obs
    
    def set_state(self, state):
        self.state = state

    def batch_dynamics(self, states, actions):
        """ (batch_size, state_size) / (batch_size, action_size) """
        if torch.is_tensor(states):
            states = states.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()
        batch_size = states.shape[0]
        new_states = np.zeros((batch_size, self.observation_space.shape[0]))
        rewards = np.zeros((batch_size,1))
        dones = np.zeros((batch_size,1))
        for i in range(batch_size):
            _states = states[i, :]
            _actions = actions[i, :]
            self.set_state(_states)
            new_state,r,done,info = self.step(_actions)
            new_states[i, :] = new_state
            rewards[i,:] = r
            dones[i,:] = done
        # TODO
        new_states = torch.from_numpy(new_states).float().to(self.device)
        return new_states,rewards,dones

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
