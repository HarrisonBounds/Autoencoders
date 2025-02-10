import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_width, input_height, num_channels):
        super().__init__()
        
        #Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_width*input_height*num_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        
        #Decoder
        self.decoder = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, input_width*input_height*num_channels),
        nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
        
        
        

        
        