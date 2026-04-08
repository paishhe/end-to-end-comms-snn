import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Transmitter(nn.Module):
    def __init__(self):
        super(Transmitter, self).__init__()

        
        self.conv1 = nn.Conv1d(1, 256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)

        
        self.conv4 = nn.Conv1d(64, 2, kernel_size=3, padding=1)

    def forward(self, x):

        x = F.relu(self.conv1(x))   
        x = F.relu(self.conv2(x))   
        x = F.relu(self.conv3(x))   
        x = self.conv4(x)           

        #  Power normalization
        
        power = torch.mean(x**2, dim=(1,2), keepdim=True) 
        # why that dim(1,2)? the shape of x is (batch-size, no-of-channels = 2, length of signal) so the power is taken by averaging the last two dims
        # therefore, one power value per signal in a sense, batch-size no. of powers in a batch
        # 2 channels mean real + im -> R*cos + I*sin is sent from the antenna
        x = x / torch.sqrt(power + 1e-8) # 1e-8 is to avoid divide by zero error
        # now x will have roughly power = 1

        return x
    

# x = np.random.randn(500)
# x = torch.tensor(x, dtype=torch.float32)
model = Transmitter()
# x = x.unsqueeze(0).unsqueeze(0) # the cnn expects input in the form (batch-size, no-of-channels, signal length) -> (1,1,128)
# output = model(x)
# print(output.shape) # we got shape (1,2,signal-length)  so the signal became two channels (complex) of the same length



# # same length output as input? seems counter intuitive -> size is a fn of the padding -> which i put so that the size is the same
# # the paper seems to keep embedded vector size = input vector size


# 'SET INPUT TO FIXED SIZE !'
# N = 128 # input size


