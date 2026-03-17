import torch
from torch.utils.data import Dataset, DataLoader

class OptionsDataset(Dataset):
    """
    Asynchronous PyTorch Dataset for Option Pricing Tensors.
    Features: (S, v, t) - spot price, implied vol from surface, time-to-maturity.
    Label: Mid-price (target V).
    """
    def __init__(self, S: torch.Tensor, v: torch.Tensor, t: torch.Tensor, V_true: torch.Tensor):
        self.features = torch.stack([S, v, t], dim=1)
        self.labels = V_true.reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_dataloader(df: pd.DataFrame, batch_size: int = 256, shuffle: bool = True) -> DataLoader:
    """
    Prepares an asynchronous DataLoader.
    Expects DataFrame with columns: ['spot', 'iv', 'T', 'price']
    """
    S = torch.tensor(df['spot'].values, dtype=torch.float32)
    v = torch.tensor(df['iv'].values, dtype=torch.float32)
    t = torch.tensor(df['T'].values, dtype=torch.float32)
    V_true = torch.tensor(df['price'].values, dtype=torch.float32)
    
    dataset = OptionsDataset(S, v, t, V_true)
    
    # Asynchronous loading as per GEMINI.md instructions: 
    # pin_memory=True for faster CPU-to-GPU transfer.
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        pin_memory=True, 
        num_workers=4
    )
