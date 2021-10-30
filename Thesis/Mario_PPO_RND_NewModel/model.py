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
        
        #print("Target features Shape: ", target_feature.shape)
        #print("Predict features Shape: ", predict_feature.shape)
        #print("Next_obs Shape: ", next_obs.shape)

        return predict_feature, target_feature

#Using Gans for predicitng features
class Discriminator(nn.Module):
    '''
    Discriminator Model for Gans
    '''

    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        feature_output = 7 * 7 * 64
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        #Initalze the weights and biases
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()


    def forward(self, x):
        output = self.discriminator(x)
        return output

class Generator(nn.Module):
    '''
    Generator Model for Gans
    '''

    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        feature_output = 7 * 7 * 64

        self.generator = nn.Sequential(
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

     #Initalze the weights and biases
        for p in self.modules():
            
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
            
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, input):
        generator_features = self.generator(input)
        return generator_features


#Using a AutoEncoder-------------------------------------------------------------------------------
class ConvAutoEncoder(nn.Module):
    def __init__(self, output_size):
        super(ConvAutoEncoder, self).__init__()

        feature_output = 7 * 7 * 64

        #Encoder
        self.encoder = nn.Sequential(
            
            #Encoding
            nn.Conv2d(in_channels=1, out_channels = 32, kernel_size = 8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels = 64, kernel_size = 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels = 64, kernel_size = 3, stride=1, padding=1),
        )
        #Decoder
        self.decoder = nn.Sequential(

            #Decoding
            nn.ConvTranspose2d(64, 64, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x