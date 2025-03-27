import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd

from legged_gym.algo.utils import utils


class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, device):
        super(AMPDiscriminator, self).__init__()

        self.device = device
        self.input_dim = input_dim

        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ELU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self,expert_state,expert_next_state,lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)
            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            d = torch.sigmoid(d)
            amp_reward = -torch.log(torch.clamp(1 - d, min=1e-8))
            reward = task_reward.unsqueeze(-1) + amp_reward
            self.train()
        return reward.squeeze(), d

