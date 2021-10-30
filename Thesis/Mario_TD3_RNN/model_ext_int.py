import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

#PPO MODEL
class PPOModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PPOModel, self).__init__()
        
        linear = nn.Linear
        
        #Shared network (CNN Part)
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
             nn.ELU(),
            
            Flatten(),
            linear(7 * 7 * 64, 256),
            nn.ELU(),

            linear(256, 448),
            nn.ELU()
        )
        
        self.actor = nn.Sequential(
            linear(448, 448),
            nn.ELU(),
            linear(448, output_size)
        )

        
        # The layer before having 2 value head
        self.common_critic_layer = nn.Sequential(
            linear(448, 448),
            nn.ELU()
        )

        self.critic_ext = linear(448, 1)
        self.critic_int = linear(448, 1)
        
        
        
        #Initialize the weights
        for p in self.modules():
            #Need to initialize the weights because it will return an error saying ELU does not have weights
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
            
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
                
        
        #Initialize critics
        init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()
        
        init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()
        

         #Intialize actor
        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()
        
        #Init value common layer
        for i in range(len(self.common_critic_layer)):
            if type(self.common_critic_layer[i]) == nn.Linear:
                init.orthogonal_(self.common_critic_layer[i].weight, 0.1)
                self.common_critic_layer[i].bias.data.zero_()
                
                
    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value_ext = self.critic_ext(self.common_critic_layer(x) + x)
        value_int = self.critic_int(self.common_critic_layer(x) + x)
        return policy, value_ext, value_int



#ICM Model
class ICMModel(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=True):
        super(ICMModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        feature_output = 7 * 7 * 64
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.ELU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.ELU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512)

        )

        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        self.residual = [nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(self.device)] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, inputs):
        state, next_state, action = inputs

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action