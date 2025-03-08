import torch
import torch.nn as nn
from torch.distributions import Normal

from legged_gym.algo.ppo_pia.EnvDiscriminator import EnvDiscriminator
from legged_gym.algo.ppo_pia.adversary import AdversaryNet
from legged_gym.algo.utils import unpad_trajectories
from legged_gym.algo.ppo_pia.estimator import EstimatorNet


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
                        priv_num=8,
                        latent_num=6,
                        *args,
                        **kwargs):

        super().__init__(*args, **kwargs)
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        self.priv_num = priv_num
        self.latent_num = latent_num

        self.env_discriminator = EnvDiscriminator(50)
        self.autoencoder = Autoencoder(self.priv_num, self.latent_num)
        self.estimator = EstimatorNet(num_obs_history)
        self.adversary = AdversaryNet(self.latent_num + 1)
        memory_input_a = num_actor_obs + self.latent_num + 1
        memory_input_c = num_critic_obs + self.latent_num + 1
        self.memory_a = Memory(memory_input_a, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(memory_input_c, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.actor = Actor(rnn_hidden_size, num_actions, actor_hidden_dims)
        self.critic = Critic(rnn_hidden_size, critic_hidden_dims)

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

    def get_latent_priv(self, critic_obs):
        priv = critic_obs[..., -self.priv_num:]
        latent_priv, decoded_priv = self.autoencoder(priv)
        return latent_priv, decoded_priv

    def get_env(self, env_obs):
        return self.env_discriminator(env_obs)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, critic_observations, obs_history, env_obs, masks=None, hidden_states=None):
        # env_features = self.estimator(obs_history)
        latent_priv, _ = self.get_latent_priv(critic_observations)
        env = self.get_env(env_obs)
        observations = torch.cat((observations, latent_priv, env), dim=-1)
        input_a = self.memory_a(observations, masks, hidden_states)
        self.update_distribution(input_a.squeeze(0))
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, critic_observations, obs_history):
        latent_priv, latent_env = self.estimator(obs_history)
        observations = torch.cat((observations, latent_priv, latent_env), dim=-1)
        input_a = self.memory_a(observations)
        return self.actor(input_a.squeeze(0))

    def evaluate(self, critic_observations, env_obs, masks=None, hidden_states=None):
        latent_priv, _ = self.get_latent_priv(critic_observations)
        env = self.get_env(env_obs)
        critic_observations = torch.cat((critic_observations, latent_priv, env), dim=-1)
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return self.critic(input_c.squeeze(0))

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
    def __init__(self, input_num, output_num):
        super(Autoencoder, self).__init__()

        self.input_num = input_num
        self.output_num = output_num

        self.encoder = nn.Sequential(
            nn.Linear(self.input_num, 64),
            nn.ELU(),
            nn.Linear(64, self.output_num),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.output_num, 64),
            nn.ELU(),
            nn.Linear(64, self.input_num),
        )

    def forward(self, env_factor):
        encoded = self.encoder(env_factor)
        decoded = self.decoder(encoded)
        return encoded, decoded
