import torch
import numpy as np
import matplotlib.pyplot as plt


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


@torch.no_grad()
def accuracy(output, target):
    equals = np.argmax(output.cpu().detach().numpy(), axis=1) == target.cpu().detach().numpy()
    acc = np.sum(equals) / output.shape[0]
    return acc


def plots(train_loss, train_acc, eval_loss, eval_acc, interval=1):
    _plot_loss(train_loss, eval_loss, interval)
    _plot_acc(train_acc, eval_acc, interval)


def _plot_acc(acc_train, acc_eval, interval):
    len_rate = len(acc_eval) / len(acc_train)
    x = [i * interval for i in range(len(acc_train))]
    x_eval = [i * interval * len_rate for i in range(len(acc_eval))]
    plt.figure(figsize=(8, 6))
    plt.plot(x, acc_train, label='Train Accuracy', color='blue')
    plt.plot(x_eval, acc_eval, label='Evaluation Accuracy', color='red')
    plt.title('Train and Evaluation Accuracy')
    plt.xlabel('# Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def _plot_loss(loss_train, loss_eval, interval):
    len_rate = len(loss_train) / len(loss_eval)
    x = [i * interval for i in range(len(loss_train))]
    x_eval = [i * interval * len_rate for i in range(len(loss_eval))]
    plt.figure(figsize=(8, 6))
    plt.plot(x, loss_train, label='Train Loss', color='blue')
    plt.plot(x_eval, loss_eval, label='Evaluation Loss', color='red')
    plt.title('Training and Evaluation Loss')
    plt.xlabel('# Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
