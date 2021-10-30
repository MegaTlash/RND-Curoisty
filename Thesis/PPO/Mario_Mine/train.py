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

#Imports
from agents import PPO  
from env import MarioEnvironment
from utils import make_train_data, RunningMeanStd, RewardForwardFilter
from config import *


import torch.optim as optim
from torch.multiprocessing import Pipe, Process

from collections import deque

#Tensor board
from tensorboardX import SummaryWriter

#Importing super mario bros 
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT 





def main():

    #Prinintg out the config hpyer parameterss
    print({section: default_config[section] for section in default_config})

    #Select the tarining environment
    env_id = default_config['EnvID']


    #Select the env_type
    env_type = default_config['EnvType']


    #Get the config hyperparameters
    use_cuda = default_config.getboolean('UseGPU')

    #GAE hyperparameters
    use_gae = default_config.getboolean('UseGAE')
    
    lam = float(default_config['Lambda'])
    
    #Number of different environments to run in parallel
    num_worker = int(default_config['NumEnv'])
    num_step = int(default_config['NumStep'])

    #PPO epsilion (aka what will help )
    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    stable_eps = float(default_config['StableEps'])

    #Gradient normalization clip
    clip_grad_norm = float(default_config['ClipGradNorm'])
    
    #Use stick action
    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb'])
    
    
    life_done = default_config.getboolean('LifeDone')
    

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(gamma)


    if env_type == 'mario':
        env = JoypadSpace(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'zelda':
        env = JoypadSpace(gym_zelda_1.make(env_id), MOVEMENT)

    
    input_size = env.observation_space.shape #4
    output_size = env.action_space.n # 2

    env.close()

    writer = SummaryWriter("new_runs/ext-int_rewards/" + env_id)

    
    model_path = 'models/{}/{}.model'.format(env_id, env_id)

    #Set the writer (Tensorboard)

    is_load_model = False
    is_render = False

    agent = PPO(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam = lam,
        learning_rate= learning_rate,
        entropy_coef= entropy_coef,
        clip_grad_norm = clip_grad_norm,
        epoch = epoch,
        batch_size = batch_size,
        ppo_eps = ppo_eps,
        use_gae= use_gae,
        use_cuda= use_cuda,
    )

    if is_load_model:
        if use_cuda:
            agent.model.load_state_dict(torch.load(load_model_path))
        else:
            agent.model.load_state_dict(
                torch.load(
                    load_model_path,
                    map_location='cpu'
                )
            )


    
    #Starting the different parallel programming links
    print("Creating different parralel environments")
    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = MarioEnvironment(env_id, is_render, idx, child_conn, life_done=life_done)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)


    states = np.zeros([num_worker, 4, 84, 84])


    sample_episode = 0
    sample_rall = 0
    sample_i_rall = 0
    sample_step = 0
    sample_env_idx = 0
    global_step = 0
    recent_prob = deque(maxlen=10)

    #normalize obs
    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    for step in range(num_step * pre_obs_norm_step):
        actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd, lr = parent_conn.recv()
            next_obs.append(s[3, :, :].reshape([1, 84, 84]))

        if len(next_obs) % (num_step * num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []
    print('End to initalize...')
    print("In the training Loop")
    while True:
        
        total_state, total_reward, total_done, total_next_state, total_action, total_values, total_policy, total_policy_np, total_next_obs = [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)

        for _ in range(num_step):
            #agent.model.eval()
            actions, value, policy = agent.get_action(np.float32(states) / 255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)
            

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            #print("Getting Info from environments")
            for parent_conn in parent_conns:
                s, r, d, rd, lr = parent_conn.recv()
                next_states.append(s)
                #print(f'reward in parent conn = {r}')
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                next_obs.append(s[3, :, :].reshape([1, 84, 84]))

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)
            
            total_state.append(states)
            #print(f'total_state = {total_state}')
            total_next_state.append(next_states)
            #print(f'reward before appending= {rewards}')
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_values.append(value)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())
            total_next_obs.append(next_obs) 

            states = next_states[:, :, :, :]
            sample_rall += log_rewards[sample_env_idx]
            
            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward' , sample_rall, sample_episode)
                writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                writer.add_scalar('data/step' , sample_step, sample_episode)
                sample_rall = 0
                sample_i_rall = 0
                sample_step = 0

    
        total_state = np.stack(total_state).transpose(
            [1, 0, 2, 3, 4]
        ).reshape([-1, 4, 84, 84])

        total_next_state = np.stack(total_next_state).transpose(
            [1, 0, 2, 3, 4]
        ).reshape([-1, 4, 84, 84])

        #Calculate last next vaue
        _, value, _ = agent.get_action(np.float32(states) / 255.)
        total_values.append(value)


        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_values = np.stack(total_values).transpose()
        total_logging_policy = np.vstack(total_policy_np)
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])


        #logging output to see how it converges
        policy = policy.detach()
        m = F.softmax(policy, dim=-1)
        recent_prob.append(m.max(1)[0].mean().cpu().numpy())
        writer.add_scalar(
            'data/max_prob',
            np.mean(recent_prob),
            sample_episode
        )

        #total_target = []
        #total_adv = []

        
        #for idx in range(num_worker):
        target, adv = make_train_data(total_reward, 
                                    total_done,
                                    total_values,
                                    gamma,
                                    num_step,
                                    num_worker
                                    )
        
        
        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------


        #print("Training the model")
        agent.train_model(
            np.float32(total_state)/ 255.,
            total_next_state,
            target,
            total_action,
            adv,
            ((total_next_obs - obs_rms.mean)/ np.sqrt(obs_rms.var)).clip(-5, 5),
            total_policy
        )
        '''
        #adjust learning rate
        if lr_schedule:
            new_learning_rate = learning_rate - (global_step / max_step) * learning_rate

            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = new_learning_rate
                writer.add_scalar(
                    'data/lr', new_learning_rate, sample_episode
                )
        '''

        
        if global_step % (num_worker * num_step * 100) == 0:
            print("Num Step: ", num_step)
            print('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), model_path)



if __name__ == '__main__':
    main()
