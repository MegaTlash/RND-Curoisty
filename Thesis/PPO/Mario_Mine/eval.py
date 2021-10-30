from agents import *
from env import * 
from utils import * 
from config import *
from torch.multiprocessing import Pipe
from gym.wrappers.monitoring import video_recorder

from tensorboardX import SummaryWriter

import numpy as np
import pickle

def main():
    print({section: dict(config[section]) for section in config.sections()})
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        env = gym_super_mario_bros.make(env_id)
        #vid = video_recorder.VideoRecorder(env, path="./recordings/vid.mp4")
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
    else:
        raise NotImplementedError
    
    input_size = env.observation_space.shape #4
    output_size = env.action_space.n #2

    
    

    is_render = True
    model_path = 'models/{}/{}.model'.format(env_id, env_id)
    predictor_path = 'models/{}/{}.pred'.format(env_id, env_id)
    target_path = 'models/{}/{}.target'.format(env_id, env_id)



    use_cuda = True
    use_gae = default_config.getboolean('UseGAE')
    

    lam = float(default_config['Lambda'])
    num_worker = 1


    num_step = int(default_config['NumStep'])


    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])


    sticky_action = False
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    if default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    else:
        raise NotImplementedError

    agent = PPO(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        clip_grad_norm= clip_grad_norm,
        epoch= epoch,
        batch_size= batch_size,
        ppo_eps=  ppo_eps,
        use_cuda=  use_cuda,
        use_gae= use_gae,
    )

    print('Loading Pre-trained model ....')
    if use_cuda:
        agent.model.load_state_dict(torch.load(model_path))

    else:
        agent.model.load_state_dict(torch.load(model_path, map_loaction='cpu'))
    print('End load ..')
    
    works = []
    parent_conns = []
    child_conns = []
    

    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    steps = 0
    rall = 0
    rd = False
    intrinsic_reward_list = []
    while not rd:
        
        steps += 1
        actions, value, policy = agent.get_action(np.float32(states)/ 255.)

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)
        
        next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
        for parent_conn in parent_conns:
            
            s, r, d, rd, lr = parent_conn.recv()
            rall += r
            next_states = s.reshape([1, 4, 84, 84])
            next_obs = s[3, :, :].reshape([1, 1, 84, 84])
    
        #total reward = in reward + ext reward
        states = next_states[:,:,:,:]


        if rd:
            intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(intrinsic_reward_list)

            with open('int_reward', 'wb') as f:
                pickle.dump(intrinsic_reward_list, f)
            steps = 0
            rall = 0
    
    env.close()

    
    
if __name__ == '__main__':
    main()