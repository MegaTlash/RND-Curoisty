import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from torch.nn import init

from torch.distributions.categorical import Categorical


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class PPOModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PPOModel, self).__init__()
        
        #if use_noisy_net:
        #    print('use NoisyNet')
        #    linear = NoisyLinear
        #else:
        linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels = 4,
                out_channels = 32,
                kernel_size = 8,
                stride=4
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 4,
                stride=2
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels = 64,
                out_channels = 64,
                kernel_size = 3,
                stride = 1
            ),
            nn.ELU(),
            Flatten(),
            linear(7 * 7 * 64, 256),
            nn.ELU(),
            linear(256, 448),
            nn.ELU(), 
        )

        self.actor = nn.Sequential(
            linear(448, 448),
            nn.ELU(),
            linear(448, output_size)
        )

        self.common_critic_layer = nn.Sequential(
            linear(448, 448),
            nn.ELU()
        )

        self.critic = linear (448, 1)

        #Initialize the weights
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()


            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        #Initialize critics
        init.orthogonal_(self.critic.weight, 0.01)
        self.critic.bias.data.zero_()

        # Intiailize actor
        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        # Init value common layer
        for i in range(len(self.common_critic_layer)):
            if type(self.common_critic_layer[i]) == nn.Linear:
                init.orthogonal_(self.common_critic_layer[i].weight, 0.1)
                self.common_critic_layer[i].bias.data.zero_()



    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(self.common_critic_layer(x) + x)
        return policy, value




