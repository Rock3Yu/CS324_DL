import argparse

from sklearn.metrics import accuracy_score

from numpy_mlp import MLP
from util import *


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


def counter(predictions, targets):
    targets = np.argmax(targets, axis=-1)
    predictions = np.argmax(predictions, axis=-1)
    return np.sum(predictions == targets)


def train(dnn_hidden_units: str, learning_rate: float, max_steps: int, eval_freq: int,
          use_batch: bool, stochastic_size: int, data: str):
    """
    Performs training and evaluation of MLP model.
    NOTE: Add necessary arguments such as the data, your model...
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        draw_plots: Draw analysis plots
        use_batch: Use batch or stochastic
        stochastic_size: The size of batch, when using stochastic
    """

    print(f"use_batch={use_batch}, stochastic_size={stochastic_size}")

    seed = SEED_DEFAULT if USE_SEED_DEFAULT else np.random.randint(4294967294)
    dataset, labels, dataset_train, dataset_test, labels_train, labels_test, \
        labels_train_oh, labels_test_oh = make_data(True, seed, data == DATAS_DEFAULT[0])

    hidden_layers = [int(i) for i in dnn_hidden_units.split(",")]
    mlp = MLP(2, hidden_layers, 2, learning_rate)
    loss_fn = mlp.loss_fn

    loss_train, loss_test = [], []
    acc_train, acc_test = [], []

    for step in range(max_steps):
        # batch
        if use_batch:
            pred_oh = mlp(dataset_train)
            loss_train.append(loss_fn(pred_oh, labels_train_oh))
            acc_train.append(accuracy(pred_oh, labels_train_oh))
            dout = loss_fn.backward(pred_oh, labels_train_oh)
            mlp.backward(dout)
            mlp.update()
        # stochastic
        else:
            loss = 0
            count_right = 0
            indices = np.random.permutation(len(dataset_train))  # shuffle in the same order
            xs = dataset_train[indices]
            ys = labels_train_oh[indices]
            if stochastic_size == 1:
                for eg, y in zip(xs, ys):
                    eg = np.reshape(eg, newshape=(1, 2))
                    pred_oh = mlp(eg)
                    loss += loss_fn(pred_oh, y)
                    if accuracy(pred_oh, [y]) == 100:
                        count_right += 1
                    else:
                        dout = loss_fn.backward(pred_oh, y)
                        mlp.backward(dout)
                        mlp.update()
            else:
                for i in range(0, len(xs), stochastic_size):
                    x = xs[i:i + stochastic_size]
                    y = ys[i:i + stochastic_size]
                    pred_oh = mlp(x)
                    loss += loss_fn(pred_oh, y)
                    right = counter(pred_oh, y)
                    count_right += right
                    if right < stochastic_size:
                        dout = loss_fn.backward(pred_oh, y)
                        mlp.backward(dout)
                        mlp.update()

            loss_train.append(loss)
            acc_train.append(count_right / len(xs) * 100)

        # print(f"Step: {step}, Loss: {loss_train[-1]}, Accuracy: {acc_train[-1]}")

        if step % eval_freq == 0 or step == max_steps - 1:
            pred_oh = mlp(dataset_test)
            loss_test.append(loss_fn(pred_oh, labels_test_oh) * 4)
            acc_test.append(accuracy(pred_oh, labels_test_oh))
            print(f"Step: {step}, Loss: {loss_test[-1]}, Accuracy: {acc_test[-1]}")

    print("Training complete!")

    plots(dataset_train, labels_train, mlp(dataset_train),
          dataset_test, labels_test, mlp(dataset_test),
          acc_train, acc_test,
          loss_train, loss_test)
    print("Plots complete!")


def main():
    """
    Main function.
    """
    # np.random.seed(SEED_DEFAULT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')

    parser.add_argument('--use_batch', type=str, default=str(USE_BATCH_DEFAULT),
                        help='Use batch when training, otherwise use stochastic way')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='The size of batch, when training by using stochastic')
    parser.add_argument('--data', type=str, default=DATAS_DEFAULT[0], choices=DATAS_DEFAULT,
                        help='moons/circles')
    flags = parser.parse_known_args()[0]

    use_batch = flags.use_batch.lower() == 'true'
    train(flags.dnn_hidden_units, flags.learning_rate, flags.max_steps, flags.eval_freq,
          use_batch, flags.batch_size, flags.data)


if __name__ == '__main__':
    main()
