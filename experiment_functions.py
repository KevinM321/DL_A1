import csv
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from classes import *

figure_save_path = './results/figures/'
data_save_path = './results/data/'

# set seed for RNG functions
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# load train and test datasets
def prepare_data(seed, batch_size=32):
    set_seed(seed)
    data_dir = "./Assignment1-Dataset/"
    train_dataset = DataSet(data_dir + "train_data.npy", data_dir + "train_label.npy")
    test_dataset = DataSet(data_dir + "test_data.npy", data_dir + "test_label.npy")

    # normalise train and test datasets using mean and standard deviation of train set
    train_mean, train_std = train_dataset.get_mean_std()
    train_dataset.normalise(train_mean, train_std)
    test_dataset.normalise(train_mean, train_std)

    return (
        test_dataset,
        DataLoader(train_dataset, batch_size)
    )

# calculate accuracy as percentage of total correct predicted label vs true label
def accuracy_fn(y_true, y_pred):
    y_pred_labels = np.argmax(y_pred, axis=1)

    correct = np.sum(y_pred_labels == y_true)

    acc = (correct / len(y_true)) * 100
    return acc

# training  and test loop for a given model with given loss function,
# optimizer, train dataloader, test dataset, and number of epochs to train for
def model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, optim, epochs=100):
    set_seed(seed)
    optim.set_params(model.get_params())

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        total_samples = 0

        # set model to training mode
        model.set_train()

        # mini-batch training
        for X_batch, y_batch in train_loader:

            # get output from one forward pass
            logits = model.forward(X_batch)

            # calculate train output loss
            loss = loss_fn(logits, y_batch)
            total_loss += loss * len(y_batch)

            # calculate train output accuracy
            acc = accuracy_fn(y_true=y_batch, y_pred=logits)
            total_acc += acc * len(y_batch)
            total_samples += len(y_batch)

            # backpropagate gradient
            grad = loss_fn.backward()
            model.backward(grad)

            # optimise trainable parameters of model
            optim.step()
            optim.zero_grad()

        test_X = np.array([d[0] for d in test_dataset])
        test_y = np.array([d[1] for d in test_dataset])

        # set model to inference mode
        model.set_test()

        # run inference and calculate test loss and accuracy
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

    # create confusion matrix for given model
    # cm = confusion_matrix(test_y, np.argmax(test_logits, axis=1))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(10)])
    # disp.plot(cmap="Blues", xticks_rotation=90, colorbar=False)
    final_res_data = (train_losses[-1], test_losses[-1], train_accs[-1], test_accs[-1])

    return final_res_data, {'Train': {'Loss': train_losses, 'Accuracy': train_accs}, 'Test': {'Loss': test_losses, 'Accuracy': test_accs}}


def plot_helper(results, category: str, y_label: str, title: str, filename: str, train: bool, test: bool, show_legend: bool):
    plt.figure(figsize=(10, 6))

    for i in range(len(results)):
        model_name, model_res = results[i]
        if train:
            plt.plot(model_res['Train'][category], label=model_name)
        if test:
            plt.plot(model_res['Test'][category], label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    if show_legend:
        plt.legend()
    plt.savefig(figure_save_path + filename + '.png', dpi=300, bbox_inches='tight')
    plt.close()

def result_store_helper(final_datas, filename):
    with open(data_save_path + filename + '.csv', 'w', newline="") as f:
        f_writer = csv.writer(f)
        f_writer.writerow(['Model', 'Train Loss', 'Test Loss', 'Train Accuracy (%)', 'Test Accuracy (%)'])
        for model_name, final_data in final_datas:
            f_writer.writerow([model_name, *final_data])