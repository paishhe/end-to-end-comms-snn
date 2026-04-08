import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticBitsDataset(Dataset):
    def __init__(self, seq_len, dataset_size):
        self.seq_len = seq_len
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        bits = torch.randint(0, 2, (self.seq_len,)).float()
        bits = bits.unsqueeze(0) # need output to be (channels, length) which is (1, length)
        return bits

train_dataset = SyntheticBitsDataset(seq_len=128, dataset_size=10000)
test_dataset = SyntheticBitsDataset(seq_len=128, dataset_size=5000)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle = True)

sample = next(iter(train_loader))
print("seq unique values:", sample.unique())
