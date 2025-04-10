from classes import *
import random
import numpy as np


data_dir = "../Assignment1-Dataset/"

seed = 100
np.random.seed(seed)
random.seed(seed)

train_dataset = DataSet(data_dir + "train_data.npy", data_dir + "train_label.npy")
test_dataset = DataSet(data_dir + "test_data.npy", data_dir + "test_label.npy")

print(train_dataset._shape(), test_dataset._shape())

train_dataloader = DataLoader(train_dataset)
test_dataloader = DataLoader(test_dataset, 1, False)

relu = RELU()
gelu = GELU()

test_vals = [1.13, -92.85, -0.35]

for val in test_vals:
    print("Val:", val, "RELU:", relu(val), "RELU Deriv:", relu.backward(relu(val)), "GELU:", gelu(val), "GELU Deriv:", gelu.backward(gelu(val)))


dropout = Dropout()
t1 = np.random.rand(10)
print(t1)
print(dropout(t1))

print("=== softmax and CEL === ")
softmax = Softmax()
loss_fn = CrossEntropyLoss()

# logits = np.array([[2.0, 1.0, 0.1],
#                    [0.5, 2.5, 0.3]])
# labels = np.array([0, 1])

# t1 = softmax(logits)
# print(t1)
# print(loss_f(t1, labels))


model = Model([
    Linear(128, 16),
    RELU(),
    # GELU(),
    Dropout(0.3),
    Linear(16, 10),
    Softmax()
])

print(np.dot(np.array([[1, 2, 3, 4]]).T, np.array([[4, 3, 2, 1]])))
lr = 0.01
optim = SGD(model.get_params())
print("===================== FORWARD ===================")
loss = loss_fn(
    np.expand_dims(model.forward(train_dataset[10][0]), axis=0), 
    train_dataset[10][1])

print("===================== BACKWARD ===================")
model.backward(loss_fn.backward())
print(optim.params)
print("===================== OPTIMISE ===================")
optim.step()

# for epoch in range(5):
#     total_loss = 0
#     for batch in train_dataloader:
#         inputs, labels = zip(*batch)
#         inputs = np.array(inputs)
#         labels = np.array(labels)

#         # Forward pass
#         x = inputs
#         for layer in model:
#             x = layer.forward(x)

#         loss = loss_fn.forward(x, labels)
#         total_loss += loss

#         # Backward pass
#         grad = loss_fn.backward()
#         for layer in reversed(model):
#             grad = layer.backward(grad)

#         # Gradient descent step
#         for layer in model:
#             if isinstance(layer, Linear):
#                 layer.W -= lr * layer.grad_W
#                 layer.b -= lr * layer.grad_b

#     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")