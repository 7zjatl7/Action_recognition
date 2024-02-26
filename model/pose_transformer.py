import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Action_LSTM(nn.Module):
    def __init__(self):
        super(Action_LSTM, self).__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 32)
        self.fc9 = nn.Linear(32, 3)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)
        self.dp4 = nn.Dropout(0.1)
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dp2(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.dp3(x)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.dp4(x)
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        
        return x
    