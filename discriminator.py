import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, num_channels, num_feat_disc):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_feat_disc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=num_feat_disc, out_channels=num_feat_disc*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_feat_disc*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=num_feat_disc*2, out_channels=num_feat_disc*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_feat_disc*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=num_feat_disc*4, out_channels=num_feat_disc*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_feat_disc*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=num_feat_disc*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.discriminator(input)
