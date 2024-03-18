import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from mlp_numpy import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
# DNN_HIDDEN_UNITS_DEFAULT = '16,32,16'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500  # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10
SEED_DEFAULT = 27


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # done: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    targets = np.argmax(targets, axis=-1)
    predictions = np.argmax(predictions, axis=-1)
    return accuracy_score(targets, predictions) * 100


def plots():
    pass


def train(dnn_hidden_units: str, learning_rate: float, max_steps: int, eval_freq: int, draw_plots=True):
    """
    Performs training and evaluation of MLP model.
    NOTE: Add necessary arguments such as the data, your model...
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        draw_plots: Draw analysis plots
    """
    # done: Load your data here, use make_moons to create a dataset of 1000 points
    dataset, labels = datasets.make_moons(n_samples=(400, 600), shuffle=True, noise=0.3, random_state=SEED_DEFAULT)
    dataset_train, dataset_test, labels_train, labels_test = (
        train_test_split(dataset, labels, test_size=0.2, random_state=SEED_DEFAULT))
    labels_train_oh = np.array([[1, 0] if i == 0 else [0, 1] for i in labels_train])
    labels_test_oh = np.array([[1, 0] if i == 0 else [0, 1] for i in labels_test])

    # points plot
    plot = False
    if plot:
        class_0 = dataset[labels == 0]
        class_1 = dataset[labels == 1]
        plt.figure(figsize=(8, 6))
        plt.scatter(class_0[:, 0], class_0[:, 1], c='blue', label='Class 0')
        plt.scatter(class_1[:, 0], class_1[:, 1], c='red', label='Class 1')
        plt.title('Generated Data with Labels')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    # done: Initialize your MLP model and loss function (CrossEntropy) here
    hidden_layers = [int(i) for i in dnn_hidden_units.split(",")]
    mlp = MLP(2, hidden_layers, 2, learning_rate)
    loss_fn = mlp.loss_fn

    loss_train, loss_test = [], []
    acc_train, acc_test = [], []

    for step in range(max_steps):
        # done: The training loop
        batch = True
        if batch:
            # 1 Forward pass
            pred_oh = mlp(dataset_train)
            # 2 Compute loss
            loss_train.append(loss_fn(pred_oh, labels_train_oh))
            acc_train.append(accuracy(pred_oh, labels_train_oh))
            # 3&4 Backward pass (compute gradients); Update weights
            dout = loss_fn.backward(pred_oh, labels_train_oh)
            mlp.backward(dout)
            mlp.update()

        else:  # stochastic
            loss = 0
            count_right = 0
            for eg, y in zip(dataset_train, labels_train_oh):
                eg = np.reshape(eg, newshape=(1, 2))
                pred_oh = mlp(eg)
                loss += loss_fn(pred_oh, y)
                count_right += (accuracy(pred_oh, [y]) == 100)
                dout = loss_fn.backward(pred_oh, y)
                mlp.backward(dout)
                mlp.update()
            loss_train.append(loss)
            acc_train.append(count_right / len(dataset_train) * 100)

        # others
        print(f"Step: {step}, Loss: {loss_train[-1]}, Accuracy: {acc_train[-1]}")

        if step % eval_freq == 0 or step == max_steps - 1:
            # done: Evaluate the model on the test set
            # 1. Forward pass on the test set
            # 2. Compute loss and accuracy
            pred_oh = mlp(dataset_test)
            loss_test.append(loss_fn(pred_oh, labels_test_oh))
            acc_test.append(accuracy(pred_oh, labels_test_oh))
            print(f"Step: {step}, Loss: {loss_test[-1]}, Accuracy: {acc_test[-1]}")

    print("Training complete!")
    # if draw_plots:
    #     plots()
    #     print("Plots complete!")

    # count = 0
    # for layer in mlp.layers:
    #     print(count)
    #     count += 1
    #     print(type(layer))
    #     if 'Linear' in str(type(layer)):
    #         print(layer.params['weight'].shape)
    #         print(layer.params)
    #         print(layer.grads)
    #         print(layer.x)


def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    flags = parser.parse_known_args()[0]

    # np.random.seed(SEED_DEFAULT)
    train(flags.dnn_hidden_units, flags.learning_rate, flags.max_steps, flags.eval_freq)


if __name__ == '__main__':
    main()
