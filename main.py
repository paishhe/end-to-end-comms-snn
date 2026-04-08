from numpy import rint
import torch
import torch.nn as nn

from tx import Transmitter
from channel import AWGNChannel
from rx import Receiver
from c_extractor import ChannelFeatureExtractor
from synthetic_bit_data import test_loader, train_loader
from spiking_c_extractor import ChannelFeatureSNN


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EndToEndSystem(nn.Module):

    def __init__(self):
        super().__init__()

        self.tx = Transmitter()
        self.rx = Receiver()
        self.c_ext = ChannelFeatureExtractor()
        self.c_ext_snn = ChannelFeatureSNN()

        

    def forward(self, input_signal, SNR):

        transmitted_signal = self.tx(input_signal)

        post_channel_signal = AWGNChannel(transmitted_signal, SNR)
        

        features_ext = self.c_ext_snn(post_channel_signal)

        post_channel_signal_flat = post_channel_signal.reshape(post_channel_signal.size(0), -1)  

        bilinear_output = torch.einsum(
    'bi,bj->bij',
    post_channel_signal_flat,
    features_ext
)  # (batch, 270, 40)

        bilinear_permuted = bilinear_output.permute(0, 2, 1)

        rx_signal = self.rx(bilinear_permuted) 

        

        return rx_signal
    

model = EndToEndSystem().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

num_epochs = 50
SNR = 12

print(f"Using device: {device}")

for epoch in range(num_epochs):
    # training
    model.train()
    train_loss = 0
    for seq in train_loader:
        seq = seq.to(device)
        optimizer.zero_grad()
        out = model(seq, SNR=SNR) 
             # (8, 1, 128)
        loss = loss_fn(out, seq)         # seq is also (8, 1, 128)
        loss.backward()
        
        optimizer.step()
        train_loss += loss.item()

        # Add after first forward pass, before training loop
        # model.eval()
        # with torch.no_grad():
        #     sample = next(iter(train_loader)).to(device)
        #     print("Input shape:         ", sample.shape)
        #     tx_out = model.tx(sample)
        #     print("After TX:            ", tx_out.shape)
        #     ch_out = AWGNChannel(tx_out, SNR)
        #     print("After channel:       ", ch_out.shape)
        #     feat = model.c_ext(ch_out)
        #     print("Channel features:    ", feat.shape)
        #     flat = ch_out.reshape(ch_out.size(0), -1)
        #     print("Signal flat:         ", flat.shape)
        #     bi = torch.einsum('bi,bj->bij', flat, feat)
        #     print("Bilinear output:     ", bi.shape)
        #     bi_p = bi.permute(0, 2, 1)
        #     print("Bilinear permuted:   ", bi_p.shape)
        #     rx_out = model.rx(bi_p)
        #     print("RX output:           ", rx_out.shape)
        #     print("Target shape:        ", sample.shape)
        #     print("Output range: min={:.4f} max={:.4f} mean={:.4f}".format(
        #         rx_out.min(), rx_out.max(), rx_out.mean()))


    # validation
    model.eval()
    correct_bits = 0
    total_bits = 0
    with torch.no_grad():
        for seq in test_loader:
            seq = seq.to(device)
            out = model(seq, SNR=SNR)
            predicted = (torch.sigmoid(out) > 0.5).float()
            correct_bits += (predicted == seq).sum().item()
            total_bits += seq.numel()

    ber = 1 - correct_bits / total_bits
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | BER: {ber:.4f}")

    








