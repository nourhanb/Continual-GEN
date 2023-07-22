import torch
from torch.utils.data import Dataset
from PIL import Image

class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

        # Filter out rows with None image paths
        self.df = self.df.dropna(subset=['path'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        try:
            X = Image.open(self.df['path'].iloc[index])
        except (IOError, OSError) as e:
            print(f"Error opening image: {self.df['path'].iloc[index]} - {e}")
            return None, None

        y = torch.tensor(int(self.df['cell_type_idx'].iloc[index]))

        if self.transform:
            X = self.transform(X)

        return X, y
