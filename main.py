import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tx import Transmitter
from channel import AWGNChannel
from rx import Receiver
from c_extractor import ChannelFeatureExtractor
from spiking_c_extractor import ChannelFeatureSNN
from synthetic_bit_data import test_loader, train_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EndToEndSystem(nn.Module):

    def __init__(self):
        super().__init__()
        self.tx = Transmitter()
        self.rx = Receiver()
        self.c_ext = ChannelFeatureExtractor()
        self.snn_c_ext = ChannelFeatureSNN()

    def forward(self, input_signal, SNR):
        
        transmitted_signal = self.tx(input_signal)
        post_channel_signal = AWGNChannel(transmitted_signal, SNR)
        features_ext = self.snn_c_ext(post_channel_signal)
        post_channel_signal_flat = post_channel_signal.reshape(post_channel_signal.size(0), -1)
        features_ext_flat = features_ext.reshape(features_ext.size(0), -1)
        bilinear_output = torch.einsum('bi,bj->bij', post_channel_signal_flat, features_ext_flat)
        bilinear_permuted = bilinear_output.permute(0, 2, 1)
        rx_signal = self.rx(bilinear_permuted)
        
        return rx_signal


model = EndToEndSystem().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 30
SNR_train = 11.5
SNR_val = 11.5

for epoch in range(num_epochs):

    # =======================
    # Training
    # =======================
    model.train()
    train_loss = 0.0

    for bits in train_loader:
        bits = bits.to(device)
        optimizer.zero_grad()
        outputs = model(bits, SNR_train)
        loss = criterion(outputs, bits)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # =======================
    # Validation
    # =======================
    model.eval()
    val_loss = 0.0
    total_bits = 0
    total_errors = 0

    with torch.no_grad():
        for bits in test_loader:
            bits = bits.to(device)
            outputs = model(bits, SNR_val)
            loss = criterion(outputs, bits)
            val_loss += loss.item()

            predicted_bits = (torch.sigmoid(outputs) > 0.5).float()
            total_errors += (predicted_bits != bits).sum().item()
            total_bits += bits.numel()

    val_loss /= len(test_loader)
    ber = total_errors / total_bits

    # =======================
    # Print metrics
    # =======================
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.6f} "
          f"Val Loss: {val_loss:.6f} "
          f"BER: {ber:.6e}")