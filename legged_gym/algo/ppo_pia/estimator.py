import torch.nn as nn
import torch
from legged_gym.algo.utils.torch_utils import get_activation, check_cnnoutput
from torch.nn import functional as F


class EstimatorNet(nn.Module):
    def __init__(self,num_obs_history):
        super(EstimatorNet, self).__init__()
        activation_fn = get_activation("elu")
        self.encoder = nn.Sequential(
            nn.Linear(num_obs_history, 512),
            activation_fn,
            nn.Linear(512, 256),
            activation_fn,
            nn.Linear(256, 6),
        )

    def forward(self, obs_history):
        return self.encoder(obs_history)











class TCNHistoryEncoder(nn.Module):
    def __init__(self,
                 num_obs_history,
                 num_latent,
                 activation='elu', ):
        super(TCNHistoryEncoder, self).__init__()
        self.num_latent = num_latent

        activation_fn = get_activation(activation)
        self.tsteps = tsteps = 15
        input_size = 40
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            activation_fn,
            nn.Linear(128, 32),
        )
        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=4), nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1), nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1), nn.LeakyReLU(), nn.Flatten())
            last_dim = 32 * 3
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=2), nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1), nn.LeakyReLU(),
                nn.Flatten())
            last_dim = 32 * 3
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6, stride=2), nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=2), nn.LeakyReLU(),
                nn.Flatten())
            last_dim = 32 * 3
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=2), nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1), nn.LeakyReLU(),
                nn.Flatten())
            last_dim = check_cnnoutput(input_size=(32, self.tsteps), list_modules=[self.conv_layers])

        self.output_layer = nn.Sequential(
            nn.Linear(last_dim, self.num_latent),
            activation_fn,
        )

    def forward(self, obs_history):
        """
        obs_history.shape = (bz, T , obs_dim)
        """
        bs = obs_history.shape[0]
        T = self.tsteps
        projection = self.encoder(obs_history)  # (bz, T , 32) -> (bz, 32, T) bz, channel_dim, Temporal_dim
        output = self.conv_layers(projection.permute(0, 2, 1))  # (bz, last_dim)
        output = self.output_layer(output)
        return output


class MLPHistoryEncoder(nn.Module):
    def __init__(self,
                 num_obs_history,
                 num_latent,
                 activation='elu',
                 adaptation_module_branch_hidden_dims=[256, 128]):
        super(MLPHistoryEncoder, self).__init__()

        self.num_latent = num_latent
        input_size = num_obs_history
        output_size = num_latent
        activation = get_activation(activation)
        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(input_size, adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l], output_size))
            else:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.encoder = nn.Sequential(*adaptation_module_layers)

    def forward(self, obs_history):
        output = self.encoder(obs_history)
        return output

