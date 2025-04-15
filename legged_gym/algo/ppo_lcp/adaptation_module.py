import torch.nn as nn
import torch


class MLPExpertEncoder(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_latent,
                 activation = nn.ELU(),
                 adaptation_module_branch_hidden_dims = [256, 128],):
        super(MLPExpertEncoder, self).__init__()
        self.num_obs = num_obs
        self.num_latent = num_latent

        input_size = num_obs + num_privileged_obs
        output_size = num_latent
        expert_encoder_layers = []
        expert_encoder_layers.append(nn.Linear(input_size, adaptation_module_branch_hidden_dims[0]))
        expert_encoder_layers.append(activation)
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                expert_encoder_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],output_size))
            else:
                expert_encoder_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                expert_encoder_layers.append(activation)
        self.encoder = nn.Sequential(*expert_encoder_layers)

    def forward(self,obs, privileged_obs):
        input = torch.cat([obs, privileged_obs], dim=-1)
        output = self.encoder(input)
        return output




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

