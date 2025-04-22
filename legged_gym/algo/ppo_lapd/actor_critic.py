import torch
import torch.nn as nn
from torch.distributions import Normal

from legged_gym.algo.ppo_lapd.estimator import AE, MLPHistoryEncoder, LSTMHistoryEncoder, PrivilegedEncoder
from legged_gym.algo.utils import unpad_trajectories


class ActorCritic(nn.Module):
    is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_privileged_obs,
                        num_obs_history,
                        num_actions,
                        actor_hidden_dims=[64],
                        critic_hidden_dims=[64],
                        rnn_type="lstm",
                        rnn_hidden_size=128,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        *args,
                        **kwargs):

        super().__init__(*args, **kwargs)
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        self.priv_encoder = PrivilegedEncoder(num_privileged_obs, 4)
        self.estimator = MLPHistoryEncoder(num_obs_history, 4)
        self.memory_a = Memory(num_actor_obs+4, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.actor = Actor(rnn_hidden_size, num_actions, actor_hidden_dims)
        self.critic = Critic(rnn_hidden_size, critic_hidden_dims)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.distribution_stu = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self):
        raise NotImplementedError

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    @property
    def action_stu_mean(self):
        return self.distribution_stu.mean

    @property
    def action_stu_std(self):
        return self.distribution_stu.stddev

    def update_distribution_stu(self, observations):
        mean = self.actor(observations)
        self.distribution_stu = Normal(mean, mean * 0. + self.std)

    def act(self, observations, privileged_obs, masks=None, hidden_states=None):
        latent = self.priv_encoder(privileged_obs)
        input_memory = torch.cat((observations, latent), dim=-1)
        input_a = self.memory_a(input_memory, masks, hidden_states)
        self.update_distribution(input_a.squeeze(0))
        return self.distribution.sample()

    def act_student(self, observations, obs_history, masks=None, hidden_states=None):
        latent = self.estimator(obs_history)
        input_memory = torch.cat((observations, latent), dim=-1)
        input_a = self.memory_a(input_memory, masks, hidden_states)
        self.update_distribution_stu(input_a.squeeze(0))
        return self.distribution_stu.sample()

    def act_inference(self, observations, obs_history):
        latent_mu = self.estimator(obs_history)
        input_memory = torch.cat((observations, latent_mu), dim=-1)
        input_a = self.memory_a(input_memory)
        return self.actor(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
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



