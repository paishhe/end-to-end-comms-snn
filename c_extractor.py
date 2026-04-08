import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelFeatureExtractor(nn.Module):
    def __init__(self):
        super(ChannelFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 2, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(2 * 128, 100)
        self.fc2 = nn.Linear(100, 40)

    def forward(self, x):
        # x: (batch, 1, 135)

        x = F.relu(self.conv1(x))  # (batch, 256, 135)
        x = F.relu(self.conv2(x))  # (batch, 128, 135)
        x = F.relu(self.conv3(x))  # (batch, 64, 135)
        x = self.conv4(x)          # (batch, 2, 135)

        x = x.view(x.size(0), -1)  # flatten → (batch, 270)

        x = F.relu(self.fc1(x))    # (batch, 100)
        x = torch.sigmoid(self.fc2(x))            # (batch, 40)

        return x