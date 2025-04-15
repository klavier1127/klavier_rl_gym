import torch
import torch.nn as nn
from torch.distributions import Normal
from legged_gym.algo.ppo_dwaq.state_estimator import VAE


class ActorCritic(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_privileged_obs,
                        num_obs_history,
                        num_single_obs,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        init_noise_std=1.0,
                        activation = nn.ELU(),
                        num_latent = 32,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        # Expert Module
        self.vae = VAE(num_obs_history=num_obs_history, num_single_obs=num_single_obs, num_latent=num_latent)

        mlp_input_dim_a = num_actor_obs + num_latent + 3
        mlp_input_dim_c = num_critic_obs + num_latent + 3

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, obs_history, **kwargs):
        z, vel = self.vae.sample(obs_history)
        self.update_distribution(torch.cat((observations, z, vel), dim=-1))
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, obs_history):
        z_mu, vel_mu = self.vae.inference(obs_history)
        actions_mean = self.actor(torch.cat((observations, z_mu, vel_mu), dim=-1))
        return actions_mean

    def evaluate(self, critic_observations, obs_history, **kwargs):
        z_mu, vel_mu = self.vae.inference(obs_history)
        value = self.critic(torch.cat((critic_observations, z_mu, vel_mu), dim=-1))
        return value