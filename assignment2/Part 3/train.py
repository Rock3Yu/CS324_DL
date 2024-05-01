from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop, lr_scheduler

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from utils import AverageMeter, accuracy, plots


def train(model, data_loader, optimizer, scheduler, criterion, device, config):
    model.train()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    loss_return, acc_return = [], []
    # for step, (batch_inputs, batch_targets) in enumerate(data_loader):
    (batch_inputs, batch_targets) = next(iter(data_loader))
    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)
    optimizer.zero_grad()
    output = model.forward(batch_inputs)
    loss = criterion(output, batch_targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)  # handle exploding gradients
    optimizer.step()
    losses.update(loss.item(), config.batch_size)
    accuracies.update(accuracy(output, batch_targets))
    if np.random.random() > 0.9:
        scheduler.step()
    # if step % 1000 == 0:
    #     print(f'Train:     [{step}/{len(data_loader)}]', losses, accuracies)
    # if step % 10 == 0:
    loss_return.append(losses.avg)
    acc_return.append(accuracies.avg)
    return loss_return, acc_return


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    model.eval()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    loss_return, acc_return = [], []
    # for step, (batch_inputs, batch_targets) in enumerate(data_loader):
    (batch_inputs, batch_targets) = next(iter(data_loader))
    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)
    output = model.forward(batch_inputs)
    loss = criterion(output, batch_targets)
    losses.update(loss.item(), config.batch_size)
    accuracies.update(accuracy(output, batch_targets))
    # if step % 1000 == 0:
    #     print(f'Evaluate:  [{step}/{len(data_loader)}]', losses, accuracies)
    # if step % 10 == 0:
    loss_return.append(losses.avg)
    acc_return.append(accuracies.avg)
    return loss_return, acc_return


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes,
                       device, config.batch_size)
    model.to(device)

    dataset = PalindromeDataset(config.input_length, config.data_size)
    train_val_num = [int(config.data_size * config.portion_train),
                     config.data_size - int(config.data_size * config.portion_train)]
    train_dataset, val_dataset = data.random_split(dataset, train_val_num)
    train_loader = DataLoader(train_dataset, config.batch_size, num_workers=2)
    val_loader = DataLoader(val_dataset, config.batch_size, num_workers=2)

    criterion = CrossEntropyLoss()
    optimizer = RMSprop(model.parameters(), config.learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    for epoch in range(config.max_epoch):
        train_loss_1, train_acc_1 = train(model, train_loader, optimizer, scheduler, criterion, device, config)
        val_loss_1, val_acc_1 = evaluate(model, val_loader, criterion, device, config)
        train_loss += train_loss_1
        train_acc += train_acc_1
        val_loss += val_loss_1
        val_acc += val_acc_1
        print(f'Epoch:  [{epoch}/{config.max_epoch}],'
              f' train loss: {np.average(train_loss_1)}, train acc: {np.average(train_acc_1)}')
        print(f'Epoch:  [{epoch}/{config.max_epoch}],'
              f' val loss: {np.average(val_loss_1)}, val acc: {np.average(val_acc_1)}')

    print('Done training.')

    plots(train_loss, train_acc, val_loss, val_acc, interval=config.batch_size)
    print('Done plots.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int, default=15, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int, default=1000000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8, help='Rate of train to total')
    configuration = parser.parse_args()

    main(configuration)
