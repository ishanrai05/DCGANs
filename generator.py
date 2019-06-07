import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, num_z, num_feat_gen, num_channels):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_z, out_channels=num_feat_gen*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_feat_gen*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=num_feat_gen*8, out_channels=num_feat_gen*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat_gen*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=num_feat_gen*4, out_channels=num_feat_gen*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat_gen*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=num_feat_gen*2, out_channels=num_feat_gen, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat_gen),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=num_feat_gen, out_channels=num_channels, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.generator(input)
