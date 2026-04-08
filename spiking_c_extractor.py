import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class ChannelFeatureSNN(nn.Module):
    def __init__(self, beta=0.9, T=16):
       
        super(ChannelFeatureSNN, self).__init__()
        self.T = T

        spike_grad = surrogate.fast_sigmoid()  

       
        self.conv1 = nn.Conv1d(2,   256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128,  64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64,    2, kernel_size=3, padding=1)

        
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        
        self.fc1  = nn.Linear(2 * 128, 100)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2  = nn.Linear(100, 40)
        

    def forward(self, x):
       
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        
        spike_accumulator = torch.zeros(x.size(0), 40, device=x.device)

        for t in range(self.T):
            

            cur = self.conv1(x)              
            spk1, mem1 = self.lif1(cur, mem1)

            cur = self.conv2(spk1)          
            spk2, mem2 = self.lif2(cur, mem2)

            cur = self.conv3(spk2)          
            spk3, mem3 = self.lif3(cur, mem3)

            cur = self.conv4(spk3)          
            spk4, mem4 = self.lif4(cur, mem4)

            flat = spk4.view(spk4.size(0), -1)  

            cur = self.fc1(flat)                  
            spk5, mem5 = self.lif5(cur, mem5)

            cur = self.fc2(spk5)                  
            spike_accumulator += cur            

        
        out = spike_accumulator / self.T          
        return out                                # ready for outer product
