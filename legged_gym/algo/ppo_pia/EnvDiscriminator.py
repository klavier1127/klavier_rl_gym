import torch.nn as nn

class EnvDiscriminator(nn.Module):
    def __init__(self, input_num):
        super(EnvDiscriminator, self).__init__()
        self.input_num = input_num

        self.env_discriminator = nn.Sequential(
            nn.Linear(self.input_num, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )

    def forward(self, env_info):
        return self.env_discriminator(env_info)