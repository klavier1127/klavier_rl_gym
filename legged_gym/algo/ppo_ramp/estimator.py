import torch.nn as nn
import torch
from legged_gym.algo.utils.torch_utils import get_activation, check_cnnoutput


class VAE(nn.Module):
    def __init__(self,
                 num_obs_history,
                 num_latent,
                 activation='elu',
                 decoder_hidden_dims=[512, 256, 128], ):
        super(VAE, self).__init__()

        self.num_obs_history = num_obs_history
        self.num_latent = num_latent

        # Build Encoder
        self.encoder = MLPHistoryEncoder(
            num_obs_history=num_obs_history,
            num_latent=num_latent * 4,
            activation=activation,
            adaptation_module_branch_hidden_dims=[512, 256],
        )
        self.latent_mu = nn.Linear(num_latent * 4, num_latent)
        self.latent_var = nn.Linear(num_latent * 4, num_latent)

        # Build Decoder
        modules = []
        activation_fn = get_activation(activation)
        decoder_input_dim = num_latent
        modules.extend(
            [nn.Linear(decoder_input_dim, decoder_hidden_dims[0]),
             activation_fn]
        )
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                modules.append(nn.Linear(decoder_hidden_dims[l], num_latent))
            else:
                modules.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
                modules.append(activation_fn)
        self.decoder = nn.Sequential(*modules)

    def encode(self, obs_history):
        encoded = self.encoder(obs_history)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        return latent_mu, latent_var

    def decode(self, z):
        output = self.decoder(z)
        return output

    def forward(self, obs_history):
        latent_mu, latent_var = self.encode(obs_history)
        z = self.reparameterize(latent_mu, latent_var)
        return z, latent_mu, latent_var

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_fn(self, obs_history, next_obs, kld_weight=1.0):
        z, latent_mu, latent_var = self.forward(obs_history)
        # Reconstruction loss
        recons = self.decode(z)
        recons_loss = torch.nn.MSELoss()(recons, next_obs)
        # Supervised loss
        kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim=-1)
        total_loss = recons_loss + kld_weight * kld_loss
        return total_loss

    def sample(self, obs_history):
        z, _, _ = self.forward(obs_history)
        return z

    def inference(self, obs_history):
        _, latent_mu, latent_var = self.forward(obs_history)
        return latent_mu














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

