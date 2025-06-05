import torch.nn as nn
import torch
from legged_gym.algo.utils.torch_utils import  get_activation,check_cnnoutput




class PrivilegedEncoder(nn.Module):
    def __init__(self, priv_num, latent_num):
        super(PrivilegedEncoder, self).__init__()
        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(priv_num, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, latent_num),
        )

    def forward(self, obs_history):
        latent = self.encoder(obs_history)
        return latent




class MLPHistoryEncoder(nn.Module):
    def __init__(self, obs_history, latent_num):
        super(MLPHistoryEncoder, self).__init__()
        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_history, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, latent_num),
        )

    def forward(self, obs_history):
        latent = self.encoder(obs_history)
        return latent




class LSTMHistoryEncoder(nn.Module):
    def __init__(self, obs_history, latent_num, hidden_size, num_layers):
        super(LSTMHistoryEncoder, self).__init__()
        # Build Encoder
        self.lstm_encoder = nn.LSTM(
            input_size=obs_history,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc =  nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ELU(),
            nn.Linear(32, latent_num)
        )

    def forward(self, obs_history):
        lstm_out, _ = self.lstm_encoder(obs_history)
        latent = self.fc(lstm_out[:, -1, :])
        return latent






class TCNHistoryEncoder(nn.Module):
    def __init__(self,
                 num_obs,
                 num_history,
                 num_latent,
                 activation='elu', ):
        super(TCNHistoryEncoder, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent

        activation_fn = get_activation(activation)
        self.tsteps = tsteps = num_history
        input_size = num_obs
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





class VAE(nn.Module):
    def __init__(self, num_obs_history, latent_num, recons_num):
        super(VAE, self).__init__()
        self.num_obs_history = num_obs_history

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_obs_history, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
        )

        self.latent_mu = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, latent_num),
        )
        self.latent_var = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, latent_num),
        )

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_num, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, recons_num),
        )

    def encode(self, obs_history):
        encoded = self.encoder(obs_history)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        return [latent_mu, latent_var]

    def decode(self, latent):
        recons = self.decoder(latent)
        return recons

    def forward(self, obs_history):
        latent_mu, latent_var = self.encode(obs_history)
        latent = self.reparameterize(latent_mu, latent_var)
        return latent, latent_mu, latent_var

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_fn(self, obs_history, ref_latent, priv_obs, kld_weight=1.0):
        est_latent, latent_mu, latent_var = self.forward(obs_history)
        # supervised loss
        supervised_loss = torch.nn.MSELoss()(est_latent, ref_latent)
        # recons loss
        recons = self.decode(est_latent)
        recons_loss = torch.nn.MSELoss()(recons, priv_obs)
        # kl loss
        kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim=-1)
        total_loss = supervised_loss + recons_loss  # + kld_weight * kld_loss
        return total_loss

    # 输出采样值（student）
    def sample(self, obs_history):
        latent, latent_mu, latent_var = self.forward(obs_history)
        return latent

    # 输出均值（teacher）
    def inference(self, obs_history):
        latent, latent_mu, latent_var = self.forward(obs_history)
        return latent_mu




class Regulator(nn.Module):
    def __init__(self, input_num):
        super(Regulator, self).__init__()

        self.regulator = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.LayerNorm(input_num)
        )

    def forward(self, input_num):
        return self.regulator(input_num)

