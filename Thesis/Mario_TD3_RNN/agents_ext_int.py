import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import PPOModel, ICMModel
from utils import global_grad_norm_


class ICMAgent(object):
    def __init__(
        self,
        input_size,
        output_size,
        num_env,
        num_step,
        gamma,
        lam=0.95,
        learning_rate= 1e-4,
        ent_coef=0.01,
        clip_grad_norm=0.5,
        epoch=3,
        batch_size = 128,
        ppo_eps=0.1,
        eta=0.01,
        use_gae= True,
        use_cuda=False,
        use_noisy_net=False):

        self.model = PPOModel(input_size, output_size)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.eta = eta
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.device = torch.device('cuda' if use_cuda else 'cpu')


        
        self.icm = ICMModel(input_size, output_size, use_cuda)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.icm.parameters()),
                                    lr=learning_rate)

                                    
        self.model = self.model.to(self.device)
        self.icm = self.icm.to(self.device)


    def GetAction(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value_ext, value_int = self.model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()


        action = self.RandomChoiceProbIndex(action_prob)

        return action, value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(),policy.detach()

    @staticmethod
    def RandomChoiceProbIndex(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)


    def ComputeIntrinsicReward(self, state, next_state, action):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)


        action_onehot= torch.FloatTensor(len(action), self.output_size).to(self.device)
        
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), -1)


        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
            [state, next_state, action_onehot]
        )
        intrinsic_reward = self.eta * F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(-1)
        return intrinsic_reward.data.cpu().numpy()

    
    def TrainModel(self, s_batch, next_s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, old_policy):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
        #target_batch = torch.FloatTensor(target_batch).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        ce = nn.CrossEntropyLoss()
        forward_mse = nn.MSELoss()

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven
                action_onehot = torch.FloatTensor(self.batch_size, self.output_size).to(self.device)
                action_onehot.zero_()
                action_onehot.scatter_(1, y_batch[sample_idx].view(-1, 1), 1)
                real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
                    [s_batch[sample_idx], next_s_batch[sample_idx], action_onehot])

                inverse_loss = ce(
                    pred_action, y_batch[sample_idx])

                forward_loss = forward_mse(
                    pred_next_state_feature, real_next_state_feature.detach())
                # ---------------------------------------------------------------------------------

                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])
                
                #Getting Total Critic Loss
                critic_loss = critic_int_loss + critic_ext_loss

                
                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = (actor_loss + 0.5 * critic_loss - 0.001 * entropy) + forward_loss + inverse_loss
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()