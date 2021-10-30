from agents import *
from envs import * 
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
        #env = wrappers.Monitor(env, './recordings/', video_callable=False, force=True)
        vid = video_recorder.VideoRecorder(env, path="./recordings/vid.mp4")
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
    else:
        raise NotImplementedError
    
    input_size = env.observation_space.shape #4
    output_size = env.action_space.n #2

    

    is_render = True
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)



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

    agent = RNDAgent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
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
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
        agent.rnd.target.load_state_dict(torch.load(target_path))

    else:
        agent.model.load_state_dict(torch.load(model_path, map_loaction='cpu'))
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
        agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
    print('End load ..')



    states = np.zeros([1, 4, 84, 84])
    env.reset()
    steps = 0
    rall = 0
    rd = False
    intrinsic_reward_list = []
    
    while not rd:
        steps += 1
        actions, value_ext, value_int, policy = agent.GetAction(np.float32(states)/ 255.)

        next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
        rall = 0
        for action in actions:
            #print(action)
            s, r, d, info = env.step(action)
            rall += r
            env.render()
            vid.capture_frame()
        
        #nextstates = s.reshape([1, 4, 84, 84])
        #next_obs = s[3, :, :].reshape([1, 1, 84, 84])
        
    
        #total reward = in reward + ext reward
        #intrinsic_reward = agent.ComputeIntrinisicReward(next_obs)
        #intrinsic_reward_list.append(intrinsic_reward)
        #states = next_states[:,:,:,:]

        

        if info["flag_get"]:
            print("World Completed")
            break
    
    vid.close()
    env.close()
    
if __name__ == '__main__':
    main()