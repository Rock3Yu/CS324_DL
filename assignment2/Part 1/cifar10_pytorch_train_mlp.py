from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
from sklearn.metrics import accuracy_score

from pytorch_mlp import MLP
from util import *


def accuracy(predictions, targets):
    targets = torch.argmax(targets, dim=-1)
    predictions = torch.argmax(predictions, dim=-1)
    return accuracy_score(targets, predictions) * 100


def counter(predictions, targets):
    targets = torch.argmax(targets, dim=-1)
    predictions = torch.argmax(predictions, dim=-1)
    return torch.sum(targets == predictions)


def train(dnn_hidden_units: str, learning_rate: float, max_steps: int, eval_freq: int, data: str):
    print("mini-batch training, with batch = ", BATCH_SIZE_DEFAULT)
    seed = SEED_DEFAULT if USE_SEED_DEFAULT else np.random.randint(4294967293)
    dataset, labels, dataset_train, dataset_test, labels_train, labels_test, \
        labels_train_oh, labels_test_oh = make_data(False, seed, data == DATAS_DEFAULT[0])

    hidden_layers = [int(i) for i in dnn_hidden_units.split(",")]
    mlp = MLP(2, hidden_layers, 2)
    loss_fn = mlp.loss_fn
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    loss_train, loss_test = [], []
    acc_train, acc_test = [], []

    for step in range(max_steps):
        mlp.train()
        loss_sum = 0
        count_right = 0
        indices = np.random.permutation(len(dataset_train))  # shuffle in the same order
        xs = dataset_train[indices]
        ys = labels_train_oh[indices]

        for i in range(0, len(xs), BATCH_SIZE_DEFAULT):
            x = xs[i:i + BATCH_SIZE_DEFAULT]
            y = ys[i:i + BATCH_SIZE_DEFAULT]
            pred_oh = mlp(x)
            optimizer.zero_grad()
            loss = loss_fn(pred_oh, y)
            loss_sum += loss
            right = counter(pred_oh, y)
            count_right += right
            if right < BATCH_SIZE_DEFAULT:
                loss.backward()
                optimizer.step()

        loss_train.append(loss_sum)
        acc_train.append(count_right / len(xs) * 100)

        if step % eval_freq == 0 or step == max_steps - 1:
            mlp.eval()
            pred_oh = mlp(dataset_test)
            loss_test.append(loss_fn(pred_oh, labels_test_oh) * 4 * 200 / BATCH_SIZE_DEFAULT)
            acc_test.append(accuracy(pred_oh, labels_test_oh))
            print(f"Step: {step}, Loss: {loss_test[-1]}, Accuracy: {acc_test[-1]}")

    print("Training complete!")

    loss_train, loss_test = [i.detach().numpy() for i in loss_train], [i.detach().numpy() for i in loss_test]
    plots(np.array(dataset_train), np.array(labels_train), mlp(dataset_train).detach().numpy(),
          np.array(dataset_test), np.array(labels_test), mlp(dataset_test).detach().numpy(),
          acc_train, acc_test,
          loss_train, loss_test)
    print("Plots complete!")

    print(len(acc_train), len(acc_test))


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
    parser.add_argument('--data', type=str, default=DATAS_DEFAULT[0], choices=DATAS_DEFAULT,
                        help='moons/circles')

    flags, unparsed = parser.parse_known_args()
    train(flags.dnn_hidden_units, flags.learning_rate, flags.max_steps, flags.eval_freq, flags.data)


if __name__ == '__main__':
    print("MLP utilized in PyTorch, CIFAR10")
    main()
