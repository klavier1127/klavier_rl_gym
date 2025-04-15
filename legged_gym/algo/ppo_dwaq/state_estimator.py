import torch.nn as nn
import torch


class VAE(nn.Module):
    def __init__(self, 
                 num_obs_history,
                 num_single_obs,
                 num_latent,
                 activation = nn.ELU(),
                 decoder_hidden_dims = [512, 256, 128],):
        super(VAE, self).__init__()

        # Build Encoder
        self.encoder = MLPHistoryEncoder(
            num_obs_history = num_obs_history,
            num_latent=num_latent * 4,
            activation=activation,
            adaptation_module_branch_hidden_dims=[512, 256],
        )
        self.latent_mu = nn.Linear(num_latent * 4, num_latent)
        self.latent_var = nn.Linear(num_latent * 4, num_latent)

        self.vel_mu = nn.Linear(num_latent * 4, 3)
        self.vel_var = nn.Linear(num_latent * 4, 3)

        # Build Decoder
        modules = []
        decoder_input_dim = num_latent + 3
        modules.extend([nn.Linear(decoder_input_dim, decoder_hidden_dims[0]), activation])
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                modules.append(nn.Linear(decoder_hidden_dims[l], num_single_obs))
            else:
                modules.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
                modules.append(activation)
        self.decoder = nn.Sequential(*modules)
    
    def encode(self,obs_history):
        encoded = self.encoder(obs_history)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        vel_mu = self.vel_mu(encoded)
        vel_var = self.vel_var(encoded)
        return [latent_mu, latent_var, vel_mu, vel_var]

    def decode(self,z,v):
        input = torch.cat([z,v], dim = 1)
        output = self.decoder(input)
        return output

    def forward(self,obs_history):
        latent_mu, latent_var, vel_mu, vel_var = self.encode(obs_history)
        z = self.reparameterize(latent_mu, latent_var)
        vel = self.reparameterize(vel_mu, vel_var)
        return [z, vel],[latent_mu, latent_var, vel_mu, vel_var]
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def loss_fn(self, obs_history, next_obs, vel, kld_weight=0.002):
        estimation, latent_params = self.forward(obs_history)
        z, v = estimation
        latent_mu, latent_var, vel_mu, vel_var = latent_params 
        # Reconstruction loss
        recons = self.decode(z,vel)
        recons_loss =nn.MSELoss()(recons, next_obs)
        # Supervised loss
        vel_loss = nn.MSELoss()(v, vel)
        kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim=1).mean()
        total_loss = recons_loss + vel_loss# + kld_weight * kld_loss
        return total_loss

    def sample(self,obs_history):
        estimation, _ = self.forward(obs_history)
        return estimation

    def inference(self,obs_history):
        _, latent_params = self.forward(obs_history)
        latent_mu, latent_var, vel_mu, vel_var = latent_params
        return [latent_mu, vel_mu]



class MLPHistoryEncoder(nn.Module):
    def __init__(self,
                 num_obs_history,
                 num_latent,
                 activation = nn.ELU(),
                 adaptation_module_branch_hidden_dims = [256, 128],):
        super(MLPHistoryEncoder, self).__init__()
        self.num_latent = num_latent

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(num_obs_history, adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l], num_latent))
            else:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.encoder = nn.Sequential(*adaptation_module_layers)

    def forward(self, obs_history):
        output = self.encoder(obs_history)
        return output