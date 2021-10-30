import gym
import os
import random
from itertools import chain

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2
import time
import datetime

#This is from the network we created 
from PPO import PPOModel

import torch.optim as optim
from torch.multiprocessing import Pipe, Process

from collections import deque

#Tensor board
from tensorboardX import SummaryWriter
from torch.distributions.categorical import Categorical

#Importing super mario bros 
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT 

from utils import global_grad_norm_


class PPO(object):

    def __init__(
        self,
        input_size, 
        output_size,
        num_env,
        num_step,
        gamma,
        lam = 0.95,
        learning_rate = 1e-4,
        entropy_coef = 0.01,
        clip_grad_norm = 0.5,
        epoch = 3,
        batch_size = 128,
        ppo_eps = 0.1,
        use_gae = True,
        use_cuda = False,
    ):

        self.model = PPOModel(
            input_size, output_size
        )

        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.clip_grad_norm = clip_grad_norm
        self.epoch = epoch
        self.batch_size = batch_size
        self.ppo_eps = ppo_eps
        self.use_gae = use_gae
        self.optimizer = optim.Adam(
            self.model.parameters(), lr = self.learning_rate
        )

        #Printing out the device
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        print("DEVICE: " ,self.device)

        self.model = self.model.to(self.device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value = self.model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(action_prob)


        return action, value.data.cpu().numpy().squeeze(), policy.detach()

    
    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis = axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)


    def forward_transition(self, agent, state, next_state):
        state = torch.from_numpy(state).to(self.device)
        state = state.float()
        policy, value = agent.model(state)

        next_state = torch.from_numpy(next_state).to(self.device)
        next_state = next_state.float()
        _, next_value = agent.model(next_state)

        value = value.data.cpu().numpy().squeeze()
        next_value = next_value.data.cpu().numpy().squeeze()

        return value, next_value, policy

    

    def train_model(self, s_batch, next_s_batch, target_batch, y_batch, adv_batch, next_obs_batch, old_policy):

        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
        target_batch = torch.FloatTensor(target_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)


        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction = "none")


        with torch.no_grad():
            #for multiply advantage
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(self.device)
            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)

        for i in range(self.epoch):
            #Doing minibatches of training
            np.random.shuffle(sample_range)

            for j in range(int(len(s_batch)/ self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j+1)]


                policy, value = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_pro = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_pro - log_prob_old[sample_idx])

                #print(f'adv_batch Shape: {adv_batch.shape}')
                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps
                ) * adv_batch[sample_idx]

                #Calculate actor loss
                actor_loss = -torch.min(surr1, surr2).mean()

                #Calculate critic loss
                critic_loss = F.mse_loss(
                    value.sum(1), target_batch[sample_idx]
                )

                entropy = m.entropy().mean()


                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                loss.backward()
                global_grad_norm_(self.model.parameters())
                self.optimizer.step()





