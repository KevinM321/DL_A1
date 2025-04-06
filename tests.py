import classes as C
import random
import numpy as np


data_dir = "../Assignment1-Dataset/"

seed = 100
np.random.seed(seed)
random.seed(seed)

train_dataset = C.DataSet(data_dir + "train_data.npy", data_dir + "train_label.npy")
test_dataset = C.DataSet(data_dir + "test_data.npy", data_dir + "test_label.npy")

print(train_dataset._shape(), test_dataset._shape())
# print(train_dataset.__getitem__(11))
# print(test_dataset.__getitem__(90))

train_dataloader = C.DataLoader(train_dataset)
test_dataloader = C.DataLoader(test_dataset, 1, False)

# for idx, batch in enumerate(test_dataloader):
#     print(batch)
#     break
        
relu = C.RELU()
gelu = C.GELU()

test_vals = [1.13, -92.85, -0.35]

for val in test_vals:
    print("Val:", val, "RELU:", relu(val), "RELU Deriv:", relu.backward(relu(val)), "GELU:", gelu(val), "GELU Deriv:", gelu.backward(gelu(val)))


dropout = C.Dropout()
t1 = np.random.rand(10)
print(t1)
print(dropout(t1))