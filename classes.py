import numpy as np
import scipy
import os
import random
from matplotlib import pyplot as plt

data_dir = "../Assignment1-Dataset/"
random.seed(100)


""" Custom DataSet and DataLoader classes"""
class DataSet:
    def __init__(self, data_file : str, label_file : str):
        self.data = np.load(data_file)
        self.labels = np.load(label_file)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __shape__(self) -> tuple:
        return self.data.shape[0], self.data.shape[1], self.labels.shape[1]
        
    def __getitem__(self, idx) -> tuple:
        return self.data[idx], self.labels[idx]


class DataLoader:
    def __init__(self, dataset : np.ndarray, batch_size : int = 32, shuffle : bool = True):
        self.dataset = dataset
        self.batch_size = batch_size if batch_size <= len(dataset) else len(dataset)
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_idx = 0
        return self
    
    def __next__(self) -> list:
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        batch_end_idx = self.current_idx + self.batch_size
        if batch_end_idx >= len(self.dataset):
            batch_end_idx = len(self.dataset) - 1

        batch_indices = self.indices[self.current_idx : batch_end_idx]
        batch = [self.dataset[i] for i in batch_indices]
        self.current_idx = batch_end_idx

        return batch


""" Activation Function Classes"""
class RELU:
    def __init__(self):
        self.input = -1

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_x):
        return grad_x if self.input > 0 else 0
    

class GELU:
    def __init__(self):
        self.input = None

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        self.input = x
        return x * 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))

    def backward(self, grad_x):
        x = self.input
        phi = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.pow(x, 2))
        Phi = 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))
        return grad_x * (x * phi + Phi)

    
train_dataset = DataSet(data_dir + "train_data.npy", data_dir + "train_label.npy")
test_dataset = DataSet(data_dir + "test_data.npy", data_dir + "test_label.npy")

print(train_dataset.__shape__(), test_dataset.__shape__())
# print(train_dataset.__getitem__(11))
# print(test_dataset.__getitem__(90))

train_dataloader = DataLoader(train_dataset)
test_dataloader = DataLoader(test_dataset, 1, False)

# for idx, batch in enumerate(test_dataloader):
#     print(batch)
#     break
        
relu = RELU()
gelu = GELU()

print(relu(1.13))
print(relu(-92.85))
print(relu(-0.35))

print(gelu(1.13))
print(gelu(-93.85))
print(gelu(-0.35))