import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size, device, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(device)
        self.fc3 = nn.Linear(hidden_size, action_size).to(device)
                
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x