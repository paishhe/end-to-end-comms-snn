import torch
import torch.nn as nn
import torch.nn.functional as F

class Receiver(nn.Module):
    def __init__(self, Nt = 1):
        super(Receiver, self).__init__()

        

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(64, 64, kernel_size=5, padding=2)

       
        self.conv8 = nn.Conv1d(64, Nt, kernel_size=8, padding=0) # i am using 8 instead of 3 in the paper to get the size to be 128
        # i don't know how they've done it

    def forward(self, x):
      

        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))  
        x = F.relu(self.conv3(x))  
        x = F.relu(self.conv4(x))  
        x = F.relu(self.conv5(x))  
        x = F.relu(self.conv6(x))  
        x = F.relu(self.conv7(x))  

        x = torch.sigmoid(self.conv8(x))  

        return x