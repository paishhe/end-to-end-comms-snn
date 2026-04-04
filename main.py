import torch
import torch.nn as nn

from tx import Transmitter
from channel import AWGNChannel
from rx import Receiver
from c_extractor import ChannelFeatureExtractor
from synthetic_bit_data import test_loader, train_loader


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader




class EndToEndSystem(nn.Module):

    def __init__(self):
        super().__init__()

        self.tx = Transmitter()
        self.rx = Receiver()
        self.c_ext = ChannelFeatureExtractor()
        

    def forward(self, input_signal, SNR):

        transmitted_signal = self.tx(input_signal)

        post_channel_signal = AWGNChannel(transmitted_signal, SNR)

        features_ext = self.c_ext(post_channel_signal)

        rx_signal = self.rx(post_channel_signal + features_ext) # THIS OPERATION SHOULD NOT BE ADDITION
        # LOOK INTO "BILINEAR OPERATION" THAT DOESN'T NEED SAME VECTOR LENGTHS TO OPERATE 

        return rx_signal
    

model = EndToEndSystem()

seq = next(iter(train_loader))
with torch.no_grad():
    out = model(seq, SNR = 5)

print("output shape", out.shape)


