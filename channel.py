import torch
import numpy as np

def AWGNChannel(tx_message, SNR):

    signal_power = torch.mean(torch.abs(tx_message**2))
    noise_power = signal_power/SNR

    noise_vector = torch.randn_like(tx_message) *torch.sqrt(noise_power)

    out = tx_message + noise_vector

    return out



