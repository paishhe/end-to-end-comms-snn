import torch
import numpy as np

def AWGNChannel(tx_message, SNR):

    snr_linear = 10 ** (SNR / 10)

    signal_power = torch.mean(tx_message**2)
    noise_power = signal_power/snr_linear

    noise_vector = torch.randn_like(tx_message)*torch.sqrt(noise_power)

    out = tx_message + noise_vector

    return out



