from torch import nn


class StateEstimator(nn.Module):
    def __init__(self, obs_history_num, num_obs_history, out_put_num):
        super(StateEstimator, self).__init__()

        self.num_obs_history = num_obs_history
        self.obs_history_num = obs_history_num
        self.out_put_num = out_put_num

        self.mlp_layers = nn.Sequential(
            nn.Linear(self.num_obs_history * self.obs_history_num, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, self.out_put_num),
        )

    def forward(self, obs_history):
        """
        obs_history.shape = (batch_size, num_history, num_obs)
        """
        obs_history = obs_history.reshape(-1, self.num_obs_history * self.obs_history_num)
        output = self.mlp_layers(obs_history)
        return output