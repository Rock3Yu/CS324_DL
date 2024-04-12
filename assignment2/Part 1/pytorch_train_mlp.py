from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch
from torch import float32
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pytorch_mlp import MLP

USE_SEED_DEFAULT = False
SEED_DEFAULT = 27

DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

DRAW_PLOTS_DEFAULT = True
USE_PYTORCH_DEFAULT = True


def accuracy(predictions, targets):
    targets = torch.argmax(targets, dim=-1)
    predictions = torch.argmax(predictions, dim=-1)
    return accuracy_score(targets, predictions) * 100


def counter(predictions, targets):
    targets = torch.argmax(targets, dim=-1)
    predictions = torch.argmax(predictions, dim=-1)
    return torch.sum(targets == predictions)


def plots(dataset, labels, acc_train, acc_test, loss_train, loss_test):
    # plot 1, point map [Using ChatGPT]
    class_0 = dataset[labels == 0]
    class_1 = dataset[labels == 1]
    plt.figure(figsize=(8, 6))
    plt.scatter(class_0[:, 0], class_0[:, 1], c='blue', label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], c='red', label='Class 1')
    plt.title('Generated Data with Labels')
    plt.xlabel('x-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

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


def train(dnn_hidden_units: str, learning_rate: float, max_steps: int, eval_freq: int):
    print("mini-batch training, with batch = 5")
    seed = SEED_DEFAULT if USE_SEED_DEFAULT else np.random.randint(4294967293)
    dataset, labels = datasets.make_moons(n_samples=(500, 500), shuffle=True, noise=0.2, random_state=seed)
    dataset_train, dataset_test, labels_train, labels_test = (
        train_test_split(dataset, labels, test_size=0.2, random_state=seed))
    dataset_train, dataset_test = (
        torch.tensor(dataset_train, dtype=float32), torch.tensor(dataset_test, dtype=float32))
    labels_train_oh = torch.tensor([[1, 0] if i == 0 else [0, 1] for i in labels_train], dtype=float32)
    labels_test_oh = torch.tensor([[1, 0] if i == 0 else [0, 1] for i in labels_test], dtype=float32)

    hidden_layers = [int(i) for i in dnn_hidden_units.split(",")]
    mlp = MLP(2, hidden_layers, 2)
    loss_fn = mlp.loss_fn
    optimizer = torch.optim.Adam(mlp.parameters())

    loss_train, loss_test = [], []
    acc_train, acc_test = [], []

    for step in range(max_steps):
        mlp.train()
        loss_sum = 0
        count_right = 0
        indices = np.random.permutation(len(dataset_train))  # shuffle in the same order
        xs = dataset_train[indices]
        ys = labels_train_oh[indices]
        for i in range(0, len(xs), 5):
            x = xs[i:i + 5]
            y = ys[i:i + 5]
            pred_oh = mlp(x)
            optimizer.zero_grad()
            loss = loss_fn(pred_oh, y)
            loss_sum += loss
            right = counter(pred_oh, y)
            count_right += right
            if right < 5:
                loss.backward()
                optimizer.step()
            loss_train.append(loss_sum)
            acc_train.append(count_right / len(xs) * 100)

        if step % eval_freq == 0 or step == max_steps - 1:
            mlp.eval()
            pred_oh = mlp(dataset_test)
            loss_test.append(loss_fn(pred_oh, labels_test_oh) * 4)
            acc_test.append(accuracy(pred_oh, labels_test_oh))
            print(f"Step: {step}, Loss: {loss_test[-1]}, Accuracy: {acc_test[-1]}")

    print("Training complete!")

    exit()
    plots(dataset, labels, acc_train, acc_test, loss_train, loss_test)
    print("Plots complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')

    flags, unparsed = parser.parse_known_args()
    train(flags.dnn_hidden_units, flags.learning_rate, flags.max_steps, flags.eval_freq)


if __name__ == '__main__':
    print("MLP utilized in PyTorch")
    main()
