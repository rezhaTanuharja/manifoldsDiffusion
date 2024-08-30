import torch
from torch.utils.data import DataLoader, TensorDataset

# Example dataset: 100 samples of 3x224x224 images and corresponding labels
data = torch.randn(100, 3, 224, 224)
labels = torch.randint(0, 10, (100,))

# Create a TensorDataset and DataLoader with shuffling enabled
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over the DataLoader
for batch_data, batch_labels in dataloader:
    print(batch_data.is_contiguous())  # Check if the tensor is contiguous
    break  # Just to demonstrate the check
