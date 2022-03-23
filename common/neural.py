import torch
from tianshou.utils.net.common import MLP
from torch import nn
from torch.nn.functional import one_hot


class MarioNet(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_dim, output_dim, device='cpu'):
        super().__init__()
        self.device = device
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        self.output_dim = output_dim
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, input, state=None):
        if self.device is not None:
            input = torch.as_tensor(
                input,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
        return self.online(input), None


class LatentTransition(nn.Module):
    def __init__(self, input_dim, action_dim, output_dim, hidden_size, device='cpu'):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.action_mlp = MLP(action_dim, hidden_size, [hidden_size])
        self.mlp = MLP(input_dim + hidden_size, output_dim, [hidden_size, hidden_size])

    def forward(self, latent, action, state=None):
        action = torch.as_tensor(
            action,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        )
        action = one_hot(action.to(int), self.action_dim).float()
        action = self.action_mlp(action)
        x = torch.cat((latent, action), -1)
        return self.mlp(x)