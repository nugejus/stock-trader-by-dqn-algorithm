import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.action = nn.Linear(64, action_dim)
        self.value = nn.Linear(64,1)

    def forward(self, state):
        x = self.fc(state)
        action = self.action(x)
        value = self.value(x)

        return action, value
