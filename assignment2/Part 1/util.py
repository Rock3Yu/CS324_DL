import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import float32
from sklearn import datasets
from sklearn.model_selection import train_test_split

USE_SEED_DEFAULT = True
SEED_DEFAULT = 27

DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500  # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10

USE_BATCH_DEFAULT = False
BATCH_SIZE_DEFAULT = 5
DATAS_DEFAULT = ['moons', 'circles']


def make_data(is_numpy, seed=np.random.randint(4294967293), is_moons=True):
    dataset, labels = datasets.make_moons(n_samples=(500, 500), shuffle=True, noise=0.2, random_state=seed) \
        if is_moons else datasets.make_circles(n_samples=(500, 500), shuffle=True, noise=5e-2, random_state=seed)
    dataset_train, dataset_test, labels_train, labels_test = \
        train_test_split(dataset, labels, test_size=0.2, random_state=seed)
    if is_numpy:
        labels_train_oh = np.array([[1, 0] if i == 0 else [0, 1] for i in labels_train])
        labels_test_oh = np.array([[1, 0] if i == 0 else [0, 1] for i in labels_test])
    else:
        dataset_train, dataset_test = \
            torch.tensor(dataset_train, dtype=float32), torch.tensor(dataset_test, dtype=float32)
        labels_train_oh = torch.tensor([[1, 0] if i == 0 else [0, 1] for i in labels_train], dtype=float32)
        labels_test_oh = torch.tensor([[1, 0] if i == 0 else [0, 1] for i in labels_test], dtype=float32)

    return dataset, labels, dataset_train, dataset_test, labels_train, labels_test, labels_train_oh, labels_test_oh


def plots(dataset_train, labels_train, pred_oh_train, dataset_test, labels_test, pred_oh_test,
          acc_train, acc_test, loss_train, loss_test):
    _plot_point(dataset_train, labels_train, pred_oh_train, "Train")
    _plot_point(dataset_test, labels_test, pred_oh_test, "Test")
    _plot_acc(acc_train, acc_test)
    _plot_loss(loss_train, loss_test)


def _plot_point(dataset, labels, pred_oh, type):
    # plot 1, point map [Using ChatGPT]
    pred = np.argmax(pred_oh, axis=-1)
    class_0_true = dataset[np.logical_and(labels == 0, labels == pred)]
    class_0_false = dataset[np.logical_and(labels == 0, labels != pred)]
    class_1_true = dataset[np.logical_and(labels == 1, labels == pred)]
    class_1_false = dataset[np.logical_and(labels == 1, labels != pred)]
    plt.figure(figsize=(8, 6))
    plt.scatter(class_0_true[:, 0], class_0_true[:, 1], c='blue', label='Class 0 - right')
    plt.scatter(class_0_false[:, 0], class_0_false[:, 1], c='blue', label='Class 0 - wrong', marker='x')
    plt.scatter(class_1_true[:, 0], class_1_true[:, 1], c='red', label='Class 1 - right')
    plt.scatter(class_1_false[:, 0], class_1_false[:, 1], c='red', label='Class 1 - wrong', marker='x')
    plt.title('Generated Data with Labels - ' + type)
    plt.xlabel('x-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()


def _plot_acc(acc_train, acc_test):
    # plot 2, accuracy curve (train + test) [Using ChatGPT]
    x_train = list(range(len(acc_train)))
    x_test = [i * EVAL_FREQ_DEFAULT for i in range(len(acc_test))]
    plt.figure(figsize=(8, 6))
    plt.plot(x_train, acc_train, label='Train Accuracy', color='blue')
    plt.plot(x_test, acc_test, label='Test Accuracy', color='red')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def _plot_loss(loss_train, loss_test):
    # plot 3, loss curve (train + test) [Using ChatGPT]
    x_train = list(range(len(loss_train)))
    x_test = [i * EVAL_FREQ_DEFAULT for i in range(len(loss_test))]
    plt.figure(figsize=(8, 6))
    plt.plot(x_train, loss_train, label='Train Loss', color='blue')
    plt.plot(x_test, loss_test, label='Test Loss', color='red')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
