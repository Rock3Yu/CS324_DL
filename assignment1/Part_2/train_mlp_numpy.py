import argparse
import numpy as np
from sklearn import datasets

from mlp_numpy import MLP
from modules import CrossEntropy

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
# DNN_HIDDEN_UNITS_DEFAULT = '16,32,16'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500  # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10


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
    return (np.sum(predictions == targets)) / len(predictions) * 100


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
    dataset, labels = datasets.make_moons(n_samples=(500, 500), shuffle=True, noise=0.1)
    dataset_train, dataset_test = dataset[:800], dataset[800:]
    labels_train, labels_test = labels[:800] + 1, labels[800:] + 1
    labels_train_one_hot = [[0, 1] if i == 0 else [1, 0] for i in labels[:800]]
    labels_test_one_hot = [[0, 1] if i == 0 else [1, 0] for i in labels[800:]]

    # done: Initialize your MLP model and loss function (CrossEntropy) here
    hidden_layers = [int(i) for i in dnn_hidden_units.split(",")]
    mlp = MLP(n_inputs=2, n_hidden=hidden_layers, n_classes=2, learning_rate=learning_rate)
    loss_fc = mlp.loss_fc

    # others
    loss_train, loss_test = [], []
    acc_train, acc_test = [], []

    for step in range(max_steps):
        # done: The training loop
        # 1 Forward pass
        pred_one_hot = mlp(dataset_train)
        pred = np.argmax(pred_one_hot, axis=1) + 1
        # 2 Compute loss
        # loss_train.append(loss_fc.forward(pred, labels_train))
        # 3&4 Backward pass (compute gradients); Update weights
        mlp.backward(loss_fc.backward(pred_one_hot, labels_train_one_hot))
        # others
        acc_train.append(accuracy(pred, labels_train))

        if step % eval_freq == 0 or step == max_steps - 1:
            # done: Evaluate the model on the test set
            # 1. Forward pass on the test set
            # 2. Compute loss and accuracy
            pred = mlp(dataset_test, True)
            pred = np.argmax(pred, axis=1)
            # loss_test.append(loss_fc.forward(pred, labels_test))
            acc_test.append(accuracy(pred, labels_test))
            # print(f"Step: {step}, Loss: {loss_test[-1]}, Accuracy: {acc_test[-1]}")
            print(f"Step: {step}, Loss: ..., Accuracy: {acc_test[-1]}")

    print("Training complete!")
    # if draw_plots:
    #     plots()
    #     print("Plots complete!")


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

    train(flags.dnn_hidden_units, flags.learning_rate, flags.max_steps, flags.eval_freq)


if __name__ == '__main__':
    main()
