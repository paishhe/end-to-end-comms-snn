import torch
import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F
import torch.nn as nn


class SNNChannelFeatureExtractor(nn.Module):
    def __init__(self):
        super(SNNChannelFeatureExtractor, self).__init__()
        self.T = 4
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 2, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=surrogate.fast_sigmoid())
        self.lif2 = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=surrogate.fast_sigmoid())
        self.lif3 = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=surrogate.fast_sigmoid())
        self.lif4 = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=surrogate.fast_sigmoid())
        self.proj = nn.Linear(2, 40)  # projects channel dim 2 → 40

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        spike_accumulator = torch.zeros(x.size(0), 2, x.size(2), device=x.device)

        for T in range(self.T):
            cur = self.conv1(x)
            spk1, mem1 = self.lif1(cur, mem1)
            cur = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)
            cur = self.conv3(spk2)
            spk3, mem3 = self.lif3(cur, mem3)
            cur = self.conv4(spk3)
            spk4, mem4 = self.lif4(cur, mem4)
            spike_accumulator += spk4

        out = spike_accumulator / self.T          # (B, 2, 128)
        out = out.permute(0, 2, 1)                # (B, 128, 2)
        out = self.proj(out)                      # (B, 128, 40)
        out = out.permute(0, 2, 1)                # (B, 40, 128)
        return out                                # ← ready for Receiver conv1(in_channels=40)