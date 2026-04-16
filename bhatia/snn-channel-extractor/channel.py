



import torch
import numpy as np

def RayleighFadingChannel(tx_message, SNR):

    snr_linear = 10 ** (SNR / 10)
    signal_power = torch.mean(tx_message**2)
    noise_power = signal_power / snr_linear

    
    h_real = torch.randn(1, device = 'cuda') / (2**0.5)
    h_imag = torch.randn(1, device = 'cuda') / (2**0.5)
    
    h_magnitude = torch.sqrt(h_real**2 + h_imag**2)  

    # Apply fading
    faded_signal = h_magnitude * tx_message

    # Add AWGN noise
    noise_vector = torch.randn_like(tx_message) * torch.sqrt(noise_power)
    out = faded_signal + noise_vector

    return out