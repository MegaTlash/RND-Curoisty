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

        '''
        self.actor_next_int = nn.Sequential(
            linear(448, 448),
            nn.ELU()
            linear(448, output_size)
        )
        '''
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
        
        '''
        #Intialize actor_int
        for i in range(len(self.actor_next_int)):
            if type(self.actor_next_int[i]) == nn.Linear:
                init.orthogonal_(self.actor_next_int[i].weight, 0.01)
                self.actor_next_int[i].bias.data.zero_()
        '''

        #Init value common layer
        for i in range(len(self.common_critic_layer)):
            if type(self.common_critic_layer[i]) == nn.Linear:
                init.orthogonal_(self.common_critic_layer[i].weight, 0.1)
                self.common_critic_layer[i].bias.data.zero_()
                
        
    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        #policy_next_int = self.actor_next_int(x)
        value_ext = self.critic_ext(self.common_critic_layer(x) + x)
        value_int = self.critic_int(self.common_critic_layer(x) + x)
        return policy, value_ext, value_int


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

#RND MODEL
class RNDModel(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()
        
        
        self.input_size = input_size
        self.output_size = output_size
        
        
        feature_output = 7 * 7 * 64
        
        
        #Prediction network
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels = 32,
                kernel_size = 8,
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
                in_channels = 64,
                out_channels = 64,
                kernel_size =3,
                stride=1,
            ),
            nn.ELU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512)
        )
        
        
        #Taregt network
        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels = 32,
                kernel_size = 8,
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
                in_channels = 64,
                out_channels = 64,
                kernel_size =3,
                stride=1,
            ),
            nn.ELU(),
            Flatten(),
            nn.Linear(feature_output, 512),
        )
        
        
        #Initalze the weights and biases
        for p in self.modules():
            
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
            
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
                
        
        #Set that target netowrk is not trainable
        for param in self.target.parameters():
            param.requres_grad = False
    
    
    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        
        return predict_feature, target_feature


#Predict Reward Model
class PredictRewardModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PredictRewardModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size


        #Prediction network
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels = 32,
                kernel_size = 8,
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
                in_channels = 64,
                out_channels = 64,
                kernel_size =3,
                stride=1,
            ),
            nn.ELU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512)
        )

        #Taregt network
        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels = 32,
                kernel_size = 8,
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
                in_channels = 64,
                out_channels = 64,
                kernel_size =3,
                stride=1,
            ),
            nn.ELU(),
            Flatten(),
            nn.Linear(feature_output, 512),
        )

        self.actor = nn.Sequential(
            linear(448, 448),
            nn.ELU(),
            linear(448, output_size)
        )
        
        #Initalze the weights and biases
        for p in self.modules():
            
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
            
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
                
        
        #Set that target netowrk is not trainable
        for param in self.target.parameters():
            param.requres_grad = False
    
    
    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        
        return predict_feature, target_feature

    