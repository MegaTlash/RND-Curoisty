import gym
import cv2

import numpy as np

from abc import abstractmethod
from collections import deque
from copy import copy

from gym.spaces import Box
from gym import Wrapper

#Super mario bros
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from render_browser import render_browser
#Legend of Zelda
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT

from torch.multiprocessing import Pipe, Process

from model import *
from config import *
from PIL import Image


train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())

class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if self.is_render:
                self.env.render()
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        
def create_train_env(world, stage, actions, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = JoypadSpace(env, actions)
    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    return env
class LegendOfZeldaEnvironment(Process):
    def __init__(
        self,
        env_id,
        is_render,
        env_idx,
        child_conn,
        history_size=4,
        life_done=True,
        h=84,
        w=84,
        movement= MOVEMENT,
        sticky_action=True,
        p=0.25):

        super(LegendOfZeldaEnvironment, self).__init__()
        self.daemon = True
        self.env = JoypadSpace(
            gym_zelda_1.make(env_id), MOVEMENT
        )
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.life_done = life_done
        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(LegendOfZeldaEnvironment, self).run()
        while True:
            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()
        
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action
            

            obs, reward, done, info = self.env.step(action)


            #When Link dies, change the state to terminal state
            if self.life_done:
                if self.hearts > info['hearts'] and info['hearts'] > 0:
                    force_done = True
                    self.hearts = info['hearts']
                else:
                    force_done = done
                    self.hearts = info['hearts']
            else:
                force_done = done

            log_reward = reward / 15
            self.rall += log_reward

            r = log_reward
            
            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(obs)

            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Heart Containters: {} killed_enemies :{}   current_level:{}".format(
                            self.episode,
                            self.env_idx,
                            self.steps,
                            self.rall,
                            np.mean(
                                self.recent_rlist),
                            info['heart_containers'],
                            info['killed_enemies'],
                            info['current_level']))

                self.history = self.reset()

            self.child_conn.send([self.history[:, :, :], r, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.hearts = 3
        self.rall = 0
        self.get_init_state(self.env.reset())
        return self.history[:, :, :]


    def pre_proc(self, X):
        # grayscaling
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        # resize
        x = cv2.resize(x, (self.h, self.w))

        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)

class MarioEnvironment(Process):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            life_done=True,
            h=84,
            w=84, movement=COMPLEX_MOVEMENT, sticky_action=True,
            p=0.25):
        super(MarioEnvironment, self).__init__()
        self.daemon = True
        self.env = JoypadSpace(
            gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)

        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.life_done = life_done
        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(MarioEnvironment, self).run()
        while True:
            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action

            obs, reward, done, info = self.env.step(action)

            # when Mario loses life, changes the state to the terminal
            # state.
            if self.life_done:
                if self.lives > info['life'] and info['life'] > 0:
                    force_done = True
                    self.lives = info['life']
                else:
                    force_done = done
                    self.lives = info['life']
            else:
                force_done = done

            # reward range -15 ~ 15
            log_reward = reward / 15
            self.rall += log_reward

            r = log_reward

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(obs)

            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print(
                    "[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Stage: {} current x:{}   max x:{}".format(
                        self.episode,
                        self.env_idx,
                        self.steps,
                        self.rall,
                        np.mean(
                            self.recent_rlist),
                        info['stage'],
                        info['x_pos'],
                        self.max_pos))

                self.history = self.reset()

            self.child_conn.send([self.history[:, :, :], r, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.lives = 3
        self.stage = 1
        self.max_pos = 0
        self.get_init_state(self.env.reset())
        return self.history[:, :, :]

    def pre_proc(self, X):
        # grayscaling
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        # resize
        x = cv2.resize(x, (self.h, self.w))

        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)


def create_train_env(world, stage, actions, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = JoypadSpace(env, actions)
    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    return env