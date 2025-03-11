import torch
import torch.nn as nn
from torch.distributions import Normal
from legged_gym.algo.utils import unpad_trajectories


class ActorCritic(nn.Module):
    is_recurrent = True
    def __init__(self, num_actor_obs,
                        num_critic_obs,
                        num_obs_history,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        rnn_type="lstm",
                        rnn_hidden_size=64,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        priv_num=6,
                        env_obs_num=62,
                        *args,
                        **kwargs):

        super().__init__(*args, **kwargs)
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        self.priv_num = priv_num
        self.env_obs_num = env_obs_num

        self.ae_p = Autoencoder(priv_num, 6)
        self.ae_l = Autoencoder(env_obs_num, 32)
        self.vae = VariationalAutoencode(num_obs_history, 6, 32)
        self.memory_a = Memory(num_actor_obs+6, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs+6, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.actor = Actor(rnn_hidden_size+32, num_actions, actor_hidden_dims)
        self.critic = Critic(rnn_hidden_size+32, critic_hidden_dims)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

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
        self.distribution = Normal(mean, mean * 0. + self.std)

    def get_priv(self, critic_obs):
        ref_priv = critic_obs[..., -self.priv_num:]
        latent_priv, decoded_priv = self.ae_p(ref_priv)
        return latent_priv, decoded_priv

    def get_latent(self, env_obs):
        latent, decoded_latent = self.ae_l(env_obs)
        return latent, decoded_latent

    def act(self, observations, critic_observations, obs_history, env_obs, masks=None, hidden_states=None):
        latent_priv, _ = self.get_priv(critic_observations)
        latent, _ = self.get_latent(env_obs)
        # latent_priv, env_value = self.vae.sample(obs_history)
        input_ma = torch.cat((observations, latent_priv), dim=-1)
        memory_output = self.memory_a(input_ma, masks, hidden_states)
        input_a = torch.cat((memory_output.squeeze(0), latent), dim=-1)
        self.update_distribution(input_a)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, critic_observations, obs_history):
        priv_mu, latent_mu = self.vae.inference(obs_history)
        input_ma = torch.cat((observations, priv_mu), dim=-1)
        memory_output = self.memory_a(input_ma)
        input_a = torch.cat((memory_output.squeeze(0), latent_mu), dim=-1)
        return self.actor(input_a)

    def evaluate(self, critic_observations, env_obs, masks=None, hidden_states=None):
        latent_priv, _ = self.get_priv(critic_observations)
        latent, _ = self.get_latent(env_obs)
        input_mc = torch.cat((critic_observations, latent_priv), dim=-1)
        memory_output = self.memory_c(input_mc, masks, hidden_states)
        input_c = torch.cat((memory_output.squeeze(0), latent), dim=-1)
        return self.critic(input_c)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states



class Actor(nn.Module):
    def __init__(self, num_actor_obs, num_actions, actor_hidden_dims = [512, 256, 128], activation = nn.ELU()):
        super().__init__()
        mlp_input_dim_a = num_actor_obs
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

    def forward(self, obs):
        actions = self.actor(obs)
        return actions



class Critic(nn.Module):
    def __init__(self,num_critic_obs, critic_hidden_dims=[512, 256, 128], activation = nn.ELU()):
        super().__init__()
        mlp_input_dim_c = num_critic_obs
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

    def forward(self, critic_obs):
        value = self.critic(critic_obs)
        return value



class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=64, decoder_hidden_dims=[128], activation = nn.ELU()):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0



class Autoencoder(nn.Module):
    def __init__(self, input_num, out_put_num):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_num, 64),
            nn.ELU(),
            nn.Linear(64, out_put_num),
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_put_num, 64),
            nn.ELU(),
            nn.Linear(64, input_num),
        )

    def forward(self, env_obs):
        encoded = self.encoder(env_obs)
        decoded = self.decoder(encoded)
        return encoded, decoded



class VariationalAutoencode(nn.Module):
    def __init__(self, num_obs_history, priv_num, latent_num):
        super(VariationalAutoencode, self).__init__()
        self.num_obs_history = num_obs_history

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_obs_history, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
        )

        self.priv_mu = nn.Linear(128, priv_num)
        self.priv_var = nn.Linear(128, priv_num)

        self.latent_mu = nn.Linear(128, latent_num)
        self.latent_var = nn.Linear(128, latent_num)

        # Build Decoder
        decoder_input = priv_num + latent_num
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 62),
        )

    def encode(self, obs_history):
        encoded = self.encoder(obs_history)
        priv_mu = self.priv_mu(encoded)
        priv_var = self.priv_var(encoded)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        return [priv_mu, priv_var, latent_mu, latent_var]

    def decode(self, priv, latent):
        input = torch.cat((priv, latent), dim=-1)
        recons = self.decoder(input)
        return recons

    def forward(self, obs_history):
        priv_mu, priv_var, latent_mu, latent_var = self.encode(obs_history)
        priv = self.reparameterize(priv_mu, priv_var)
        latent = self.reparameterize(latent_mu, latent_var)
        return [priv, latent], [priv_mu, priv_var, latent_mu, latent_var]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_fn(self, obs_history, ref_priv, ref_latent, ref_env, kld_weight=1.0):
        estimation, latent_params = self.forward(obs_history)
        est_priv, est_latent = estimation
        priv_mu, priv_var, latent_mu, latent_var = latent_params
        # supervised loss
        supervised_loss = torch.nn.MSELoss()(est_priv, ref_priv) + torch.nn.MSELoss()(est_latent, ref_latent)
        # recons loss
        recons = self.decode(est_priv, est_latent)
        recons_loss = torch.nn.MSELoss()(recons, ref_env)
        # kl loss
        kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim=-1)
        kld_loss = torch.mean(kld_loss)
        total_loss = supervised_loss + recons_loss# + kld_weight * kld_loss
        return total_loss

    # 输出采样值（student）
    def sample(self, obs_history):
        estimation, _ = self.forward(obs_history)
        return estimation

    # 输出均值（teacher）
    def inference(self, obs_history):
        _, latent_params = self.forward(obs_history)
        priv_mu, priv_var, latent_mu, latent_var = latent_params
        return [priv_mu, latent_mu]



class Adversary(nn.Module):
    def __init__(self, env_features, hidden_size=32, num_layers=2, epsilon=0.1):
        super(Adversary, self).__init__()
        self.epsilon = epsilon
        layers = []
        input_size = env_features
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LeakyReLU(0.2))
            input_size = hidden_size
        layers.append(nn.Linear(hidden_size, env_features))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        delta = self.net(x) * self.epsilon
        return delta



