import torch
import torch.nn as nn
import torch.nn.functional as F

class Receiver(nn.Module):
    def __init__(self, Nt = 1):
        super(Receiver, self).__init__()

        

        self.conv1 = nn.Conv1d(in_channels=40, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(64, 64, kernel_size=5, padding=2)

       
        self.conv8 = nn.Conv1d(64, Nt, kernel_size=3, padding=1) 

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(256, 128)
        

    def forward(self, x):
      

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))  
        x = F.relu(self.conv7(x))  
        x = self.conv8(x)
        x = self.fc_out(x)  # (batch, 128, signal_length)

          # output in range [0, 1] for binary classification

        # x = torch.sigmoid(self.fc_out(x))  

        

        return x