from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import torchvision
from torchvision import transforms
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

from cnn_model import CNN

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 1  # 5000
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
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    cnn = CNN(3 * 32 * 32, 10)
    loss_fn = cnn.loss_fn
    optimizer = torch.optim.Adam(cnn.parameters(), lr)
    # print('CNN model structure:\n', cnn)

    # train
    if not pure_test:
        for epoch in range(max_steps):
            cnn.train()
            running_loss = 0
            for i, data in enumerate(loader_train, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = cnn(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % eval_freq == eval_freq - 1:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / eval_freq:.3f}')
                    running_loss = 0.0
        print('Training complete!')
        torch.save(cnn.state_dict(), MODEL_PATH)
        print('Model save to:', MODEL_PATH)

    # test
    MODEL_PATH = './cifar_cnn_20240422_030956.pth'
    cnn.load_state_dict(torch.load(MODEL_PATH))

    # rate of correct
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader_test:
            images, labels = data
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

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
    parser.add_argument('--pure_test', type=bool, default=True,
                        help='[True or False] ---> [Test or Train&Test]')
    flags, unparsed = parser.parse_known_args()

    train(flags.learning_rate, flags.max_steps, flags.batch_size, flags.eval_freq, flags.data_dir, flags.pure_test)


if __name__ == '__main__':
    print("CNN utilized by PyTorch. CIFAR10 Dataset")
    main()
