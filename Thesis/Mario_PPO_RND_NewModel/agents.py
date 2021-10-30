import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import PPOModel, RNDModel, Generator, Discriminator
from utils import global_grad_norm_


class RNDAgent(object):
    
    def __init__(
        self,
        input_size,
        output_size,
        num_env,
        num_step,
        gamma,
        lam=0.95,
        learning_rate=1e-4,
        ent_coef=0.01,
        clip_grad_norm=0.5,
        epoch=3,
        batch_size=128,
        ppo_eps=0.1,
        update_proportion=0.25,
        use_gae=True,
        use_cuda=False):
        
        #Creating a PPO Model
        self.model  =  PPOModel(input_size, output_size)
        
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.learning_rate = learning_rate
        self.beta1 = 0.5
        self.ent_coef = ent_coef
        self.clip_grad_norm = clip_grad_norm
        self.epoch = epoch
        self.batch_size = batch_size
        self.ppo_eps = ppo_eps
        self.update_proportion = update_proportion
        self.use_gae = use_gae
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        print("DEVICE: ", self.device)
        
        
        #Creating a RND Model
        self.rnd = RNDModel(input_size, output_size)
        
        #Creating GAN model
        self.netD = Discriminator(input_size, output_size)
        self.netG = Generator(input_size, output_size)

        #Using an optimizer (Adam)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.rnd.predictor.parameters()), lr=self.learning_rate)
        
        #Variables for Gans ------------------------------------------------
        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(64, 100, 1, 1, device = self.device)

        #Distinguishing Real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        #(GANS) optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))

        #---------------------------------------------------------------------
        
        self.rnd = self.rnd.to(self.device)
        self.model = self.model.to(self.device)
        self.netD = self.netD.to(self.device)
        self.netG = self.netG.to(self.device)
    
    def GetAction(self, state):
        
        #Transform the state into a float 32 tensor
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        
        
        #Getting the policy, value_ext, value _int
        policy, value_ext, value_int = self.model(state)
        
        #Get action probability distrubiton
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        #select action
        action = self.RandomChoiceProbIndex(action_prob)

        return action, value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(), policy.detach() #, action_int , policy_int.detach()
    
    
    
    @staticmethod
    def RandomChoiceProbIndex(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)
    
    #Calculate Intrinsic reward
    def ComputeIntrinisicReward(self, next_obs):
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        
        
        #Get target feature
        target_next_feature = self.rnd.target(next_obs)
        
        #Get prediction feature
        #predict_next_feature = self.rnd.predictor(next_obs)

        #Getting the generator feature
        fake_next_features = self.netG(next_obs)
        
        
        #Calculate intrinisc reward
        intrinsic_reward = (target_next_feature - fake_next_features).pow(2).sum(1) / 2
        
        return intrinsic_reward.data.cpu().numpy()
    
    
    
    def TrainModel(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        #target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        
        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction = "none")
        
        
        #Getting old policy
        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1,0,2).contiguous().view(-1, self.output_size).to(self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)

            
        for i in range(self.epoch):
            #Doing minibatches of training
            np.random.shuffle(sample_range)
            
            for j in range(int(len(s_batch)/ self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                
                torch.autograd.set_detect_anomaly(True)
                
                #--------------------------------------------------------------------------------------
                #Curiosity driven calcuation (RND-GAN)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################               
                
                #Used for Real Data for D
                predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch[sample_idx])
                #Real Data for GAN
                self.optimizerD.zero_grad()

                label = torch.full((self.batch_size,), self.real_label, dtype=torch.float, device=self.device)
                output = self.netD(target_next_state_feature).view(-1)
                errD_real = self.criterion(output, label)
                D_x = output.mean().item()
                errD_real.backward()

                #Fake data for GAN
                fake = self.netG(next_obs_batch[sample_idx])
                label.fill_(self.fake_label)
                output = self.netD(fake.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                D_G_z1 = output.mean().item()
                errD_fake.backward()

                #Getting total error
                #errD = errD_real + errD_fake
                #errD.backward()
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.optimizerG.zero_grad()
                label.fill_(self.real_label)
                output = self.netD(fake).view(-1)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                #Update G
                self.optimizerG.step()


                #Calling Getting fake value
                fake = self.netG(next_obs_batch[sample_idx])
                #Using the Generator as the predictor
                forward_loss = forward_mse(fake, target_next_state_feature.detach()).mean(-1)
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                
                #--------------------------------------------------------------------------------------------------
                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])
                
                
                ratio = torch.exp(log_prob - log_prob_old[sample_idx])
                
                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps) * adv_batch[sample_idx]
                
                #Calculate actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                #Calcualate critic loss
                #critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])
                
                
                #Critic loss = critic E loss + critic I loss
                #critic_loss = critic_ext_loss + critic_int_loss
                critic_loss = critic_int_loss
                #Calculate the entropy
                # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
                entropy = m.entropy().mean()
                
                #Reset the gardients
                self.optimizer.zero_grad()

                #Calculate the loss
                #Total loss = Policy gradient loss - entropy * entropy coefficent + Value coefficent * value loss + foward_loss
                loss  = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + forward_loss
                
                
                #Backpropagation
                loss.backward()
                
                global_grad_norm_(list(self.model.parameters())+list(self.rnd.predictor.parameters()))

                #Optimizer Step
                self.optimizer.step()


