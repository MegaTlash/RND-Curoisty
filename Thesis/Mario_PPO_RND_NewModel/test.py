from model import *
from envs import *

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