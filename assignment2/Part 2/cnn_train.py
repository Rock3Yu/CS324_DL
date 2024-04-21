# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import numpy as np
import torch
from torch.utils.data import Subset
import torchvision
from torchvision import transforms
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

from cnn_model import CNN
from util import plots_for_part2

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = '../Part 1/CIFAR10'


def accuracy(predictions, targets):
    targets = torch.argmax(targets, dim=-1)
    predictions = torch.argmax(predictions, dim=-1)
    return accuracy_score(targets, predictions) * 100


def counter(predictions, targets):
    targets = torch.argmax(targets, dim=-1)
    predictions = torch.argmax(predictions, dim=-1)
    return torch.sum(targets == predictions)


def train(lr: int, max_steps: int, batch_size: int, eval_freq: int, data_dir: str, pure_test: bool):
    print('mini-batch training, with batch =', batch_size)
    MODEL_PATH = './cifar_cnn_' + (datetime.now() + timedelta(hours=0)).strftime('%Y%m%d_%H%M%S') + '.pth'

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    dataset_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # 使用子集，选择总数10%的样本
    subset_indices_train = torch.randperm(len(dataset_train))[:max_steps]
    # subset_indices_test = torch.randperm(len(dataset_test))[:1000]
    dataset_train = Subset(dataset_train, subset_indices_train)
    # dataset_test = Subset(dataset_test, subset_indices_test)

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    cnn = CNN(3 * 32 * 32, 10)
    loss_fn = cnn.loss_fn
    optimizer = torch.optim.Adam(cnn.parameters(), lr)

    print('CNN model structure:\n', cnn)

    # rate of correct
    def test(print_result=False, partial=False):
        cnn.eval()
        correct = 0
        total = 0
        loss_ = 0
        random_num = 0.5 + np.random.rand(1)
        with torch.no_grad():
            for data in loader_test:
                random_num += np.random.rand(1) * 0.2
                if partial and random_num > 1:
                    random_num = 0
                    continue
                images, labels = data
                outputs = cnn(images)
                loss_ += loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        rate = 100 * correct // total
        if print_result:
            print(f'Accuracy of the network on the 10000 test images: {rate} %')
        loss_ = loss_ / total * eval_freq  # normalization
        return rate, loss_

    loss_train, loss_test = [], []
    acc_train, acc_test = [], []

    # train
    if not pure_test:
        for epoch in range(1):
            cnn.train()
            running_loss = 0
            cnt, cnt_right = 0., 0
            for i, data in enumerate(loader_train, 0):
                cnt += batch_size
                inputs, labels = data
                optimizer.zero_grad()
                outputs = cnn(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                cnt_right += counter(outputs, labels)
                real_i = i * batch_size

                if int(real_i / eval_freq) != int((real_i - batch_size) / eval_freq):
                    print(f'[{epoch + 1}, {real_i:5d}] loss: {running_loss / eval_freq:.3f}')
                    acc_train.append((cnt_right / cnt) * 100)
                    loss_train.append(running_loss)
                    running_loss = 0.0
                    cnt = 0.
                    # test while train
                    acc_test_1, loss_test_1 = test(partial=True)
                    acc_test.append(acc_test_1)
                    loss_test.append(loss_test_1)
                    cnn.train()

        print('Training complete!')
        torch.save(cnn.state_dict(), MODEL_PATH)
        print('Model save to:', MODEL_PATH)
        # plots
        plots_for_part2(acc_train, acc_test, loss_train, loss_test, eval_freq)

    # test
    else:
        MODEL_PATH = './cifar_cnn_demo.pth'
        cnn.load_state_dict(torch.load(MODEL_PATH))
    test(print_result=True)

    # rate of correct by classes
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        for data in loader_test:
            images, labels = data
            outputs = cnn(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--pure_test', type=str, default='False',
                        help='[True or False] ---> [Test or Train&Test]')
    flags, unparsed = parser.parse_known_args()

    pure_test = str(flags.pure_test).lower() == 'true'
    train(flags.learning_rate, flags.max_steps, flags.batch_size, flags.eval_freq, flags.data_dir, pure_test)


if __name__ == '__main__':
    print("CNN utilized by PyTorch. CIFAR10 Dataset")
    main()
