import numpy as np
import matplotlib.pyplot as plt


def plots_for_part2(acc_train, acc_test, loss_train, loss_test, interval):
    _plot_acc(acc_train, acc_test, interval)
    _plot_loss(loss_train, loss_test, interval)


def _plot_acc(acc_train, acc_test, interval):
    assert len(acc_train) == len(acc_test)
    x = [i * interval for i in range(len(acc_train))]
    plt.figure(figsize=(8, 6))
    plt.plot(x, acc_train, label='Train Accuracy', color='blue')
    plt.plot(x, acc_test, label='Test Accuracy', color='red')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('# Training Images (or epoches)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def _plot_loss(loss_train, loss_test, interval):
    x = [i * interval for i in range(len(loss_test))]
    plt.figure(figsize=(8, 6))
    plt.plot(x, loss_train, label='Train Loss', color='blue')
    plt.plot(x, loss_test, label='Test Loss', color='red')
    plt.title('Training and Testing Loss')
    plt.xlabel('# Training Images (or epoches)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
