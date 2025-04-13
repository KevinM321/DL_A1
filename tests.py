from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from classes import *


# X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=1.0, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train_dataset = DataSet(X_train, y_train)
# test_dataset = DataSet(X_test, y_test)

# train_loader = DataLoader(train_dataset, 8)
# test_loader = DataLoader(test_dataset, 8)

# # Visualize
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10')
# plt.title("Simple 2D Multi-Class Dataset (3 classes)")
# plt.show()

data_dir = "../Assignment1-Dataset/"
train_dataset = DataSet(data_dir + "train_data.npy", data_dir + "train_label.npy")
test_dataset = DataSet(data_dir + "test_data.npy", data_dir + "test_label.npy")

train_mean, train_std = train_dataset.get_mean_std()
train_dataset.normalise(train_mean, train_std)
test_dataset.normalise(train_mean, train_std)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size)
test_loader = DataLoader(test_dataset, batch_size)


# model = Model([
#     Linear(128, 256),
#     BatchNorm(256),
#     GELU(),
#     Dropout(0.1),

#     Linear(256, 128),
#     BatchNorm(128),
#     GELU(),
#     Dropout(0.1),

#     Linear(128, 64),
#     BatchNorm(64),
#     GELU(),
#     # Dropout(0.1),

#     Linear(64, 32),
#     GELU(),

#     Linear(32, 10),
#     Softmax()
# ])

def accuracy_fn(y_true, y_pred):
    """
    y_true: shape (batch_size,) — ground truth labels (integers)
    y_pred: shape (batch_size, num_classes) — model output (logits or probabilities)
    """
    y_pred_labels = np.argmax(y_pred, axis=1)

    correct = np.sum(y_pred_labels == y_true)

    acc = (correct / len(y_true)) * 100
    return acc

def training_loop(model, loss_fn, optim):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        total_samples = 0

        model.set_train()
        for X_batch, y_batch in train_loader:
            logits = model.forward(X_batch)

            loss = loss_fn(logits, y_batch)
            total_loss += loss * len(y_batch)

            acc = accuracy_fn(y_true=y_batch, y_pred=logits)
            total_acc += acc * len(y_batch)
            total_samples += len(y_batch)

            grad = loss_fn.backward()
            model.backward(grad)

            optim.step()
            optim.zero_grad()

        test_X = np.array([d[0] for d in test_dataset])
        test_y = np.array([d[1] for d in test_dataset])

        model.set_test()

        test_logits = model.forward(test_X)
        test_loss = loss_fn.forward(test_logits, test_y)
        test_acc = accuracy_fn(test_y, test_logits)

        if epoch % 5 == 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_acc / total_samples
            print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        avg_loss = total_loss / total_samples
        train_losses.append(avg_loss)
        test_losses.append(test_loss)
        train_accs.append(avg_acc)
        test_accs.append(test_acc)

    cm = confusion_matrix(test_y, np.argmax(test_logits, axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(10)])
    disp.plot(cmap="Blues", xticks_rotation=90, colorbar=False)

    return {'train_l': train_losses, 'test_l': test_losses, 'train_a': train_accs, 'test_a': test_acc}


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

results = []

def test_model_res(seed):
    set_seed(seed)
    model = Model([
        Linear(128, 256),
        BatchNorm(256),
        GELU2(),

        Linear(256, 128),
        BatchNorm(128),
        GELU2(),
    
        Linear(128, 10),
        Softmax()
    ])
    lr = 0.0001
    optim = Adam(model.get_params(), lr)
    loss_fn = CrossEntropyLoss()

    return training_loop(model, loss_fn, optim)

def best_model_res(seed):
    set_seed(seed)
    model = Model([
        Linear(128, 128),
        BatchNorm(128),
        GELU2(),

        Linear(128, 64),
        BatchNorm(64),
        GELU2(),
        Dropout(0.15),

        Linear(64, 10),
        Softmax()
    ])
    lr = 0.0001
    optim = Adam(model.get_params(), lr)
    loss_fn = CrossEntropyLoss()

    return training_loop(model, loss_fn, optim)

seed = 23
results.append(best_model_res(seed))
# results.append(test_model_res(seed))

plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label="Train Loss")
# plt.plot(test_losses, label="Test Loss")

i = 1
for run in results:
    plt.plot(run['test_l'], label='Test Loss' + str(i))
    i += 1

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss")
plt.legend()
# plt.grid(True)
plt.show()
