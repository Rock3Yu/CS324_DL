# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop, lr_scheduler

from dataset import PalindromeDataset
# from vanilla_rnn import VanillaRNN
from lstm import LSTM
from utils import AverageMeter, accuracy, plots


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes)
    # device, config.batch_size)
    model.to(device)

    dataset = PalindromeDataset(config.input_length, config.data_size)
    train_eval_num = [int(len(dataset) * config.portion_train), 0]
    train_eval_num[1] = len(dataset) - train_eval_num[0]
    train_dataset, eval_dataset = data.random_split(dataset, train_eval_num)
    train_loader = DataLoader(train_dataset, config.batch_size, num_workers=2)
    eval_loader = DataLoader(eval_dataset, config.batch_size, num_workers=2)

    criterion = CrossEntropyLoss()
    optimizer = RMSprop(model.parameters(), config.learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    train_loss, train_acc, eval_loss, eval_acc = [], [], [], []
    train_iter = iter(train_loader)  # attention: this process is very slow
    eval_iter = iter(eval_loader)
    for epoch in range(config.max_epoch):
        losses_meter_train = AverageMeter("Train Loss")
        accuracies_meter_train = AverageMeter("Train Accuracy")
        losses_meter_eval = AverageMeter("Evaluation Loss")
        accuracies_meter_eval = AverageMeter("Evaluation Accuracy")

        # train
        model.train()
        try:
            batch_inputs, batch_targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_inputs, batch_targets = next(train_iter)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        output = model.forward(batch_inputs)
        loss = criterion(output, batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)  # handle exploding gradients
        optimizer.step()
        losses_meter_train.update(loss.item(), config.batch_size)
        accuracies_meter_train.update(accuracy(output, batch_targets))
        if epoch % 10 == 0:
            scheduler.step()
        if epoch % 50 == 0:
            print(f'Train: [{epoch}/{len(train_loader)}]', losses_meter_train, accuracies_meter_train)
        train_loss.append(losses_meter_train.avg)
        train_acc.append(accuracies_meter_train.avg)

        # evaluation
        if epoch % 10 == 0:
            model.eval()
            try:
                batch_inputs, batch_targets = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_loader)
                batch_inputs, batch_targets = next(eval_iter)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            output = model.forward(batch_inputs)
            loss = criterion(output, batch_targets)
            losses_meter_eval.update(loss.item(), config.batch_size)
            accuracies_meter_eval.update(accuracy(output, batch_targets))
            eval_loss.append(losses_meter_eval.avg)
            eval_acc.append(accuracies_meter_eval.avg)

    print('Done training.')

    plots(train_loss, train_acc, eval_loss, eval_acc, interval=config.batch_size)
    print('Done plots.')


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_length', type=int, default=19, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--max_epoch', type=int, default=2000, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int, default=100000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8, help='Rate of train to total')
    configuration = parser.parse_args()

    main(configuration)
