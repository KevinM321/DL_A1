import numpy as np
import os
import random
from matplotlib import pyplot as plt

data_dir = "../Assignment1-Dataset/"


class DataSet:
    def __init__(self, data_file : str, label_file : str):
        self.data = np.load(data_file)
        self.labels = np.load(label_file)
    
    def __len__(self):
        return len(self.data)
    
    def __shape__(self):
        return (self.data.shape[0], self.data.shape[1], self.labels.shape[1])
        
    def __getitem__(self, idx):
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
    
    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        batch_end_idx = self.current_idx + self.batch_size
        if batch_end_idx >= len(self.dataset):
            batch_end_idx = len(self.dataset) - 1

        batch_indices = self.indices[self.current_idx : batch_end_idx]
        batch = [self.dataset[i] for i in batch_indices]
        self.current_idx = batch_end_idx

        return batch

        


train_dataset = DataSet(data_dir + "train_data.npy", data_dir + "train_label.npy")
test_dataset = DataSet(data_dir + "test_data.npy", data_dir + "test_label.npy")

print(train_dataset.__shape__(), test_dataset.__shape__())
# print(train_dataset.__getitem__(11))
# print(test_dataset.__getitem__(90))

        