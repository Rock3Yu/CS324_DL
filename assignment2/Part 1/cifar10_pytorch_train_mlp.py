from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torchvision
from torchvision import transforms
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


# def show_img(task2, labels, classes):
#     def imshow(img):
#         img = img / 2 + 0.5
#         npimg = img.numpy()
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))
#         plt.show()
#
#     # imshow(torchvision.utils.make_grid(task2))
#     print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE_DEFAULT)))


def train(dnn_hidden_units: str, learning_rate: float, pure_test: bool):
    print('mini-batch training, with batch =', BATCH_SIZE_DEFAULT)
    print('dnn_hidden_units =', dnn_hidden_units)
    seed = SEED_DEFAULT if USE_SEED_DEFAULT else np.random.randint(4294967293)
    MODEL_PATH = './cifar_mlp_' + dnn_hidden_units + '.pth'

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset_train = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
    dataset_test = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE_DEFAULT,
                                               shuffle=True, num_workers=2)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE_DEFAULT,
                                              shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # dataiter = iter(loader_train)
    # task2, labels = next(dataiter)
    # show_img(task2, labels, classes)

    # print(task2[0].shape)  # torch.Size([3, 32, 32])
    hidden_layers = [int(i) for i in dnn_hidden_units.split(',')]
    mlp = MLP(3 * 32 * 32, hidden_layers, 10)
    loss_fn = mlp.loss_fn
    # optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)  # very bad here
    optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate, momentum=0.8)  # good!

    # train
    if not pure_test:
        for epoch in range(2):
            mlp.train()
            running_loss = 0
            for i, data in enumerate(loader_train, 0):
                inputs, labels = data
                inputs = inputs.view(inputs.size(0), -1)
                optimizer.zero_grad()
                outputs = mlp(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        print('Training complete!')
        torch.save(mlp.state_dict(), MODEL_PATH)
        print('Model save to:', MODEL_PATH)

    # test
    mlp.load_state_dict(torch.load(MODEL_PATH))
    # dataiter = iter(loader_test)
    # task2, labels = next(dataiter)
    # task2 = task2.view(task2.size(0), -1)
    # show_img(task2, labels, classes)
    # outputs = mlp(task2)
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(BATCH_SIZE_DEFAULT)))

    # rate of correct
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader_test:
            images, labels = data
            images = images.view(images.size(0), -1)
            outputs = mlp(images)
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
            images = images.view(images.size(0), -1)
            outputs = mlp(images)
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
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--pure_test', type=bool, default=False,
                        help='[True or False] ---> [Test or Train&Test]')

    flags, unparsed = parser.parse_known_args()
    flags.learning_rate = 5e-3
    # flags.dnn_hidden_units = '1024,256,64,32'
    # flags.dnn_hidden_units = '1024,256,32'
    # flags.dnn_hidden_units = '256,32'
    flags.dnn_hidden_units = '256'  # best
    # flags.dnn_hidden_units = '64'
    train(flags.dnn_hidden_units, flags.learning_rate, flags.pure_test)


if __name__ == '__main__':
    print('MLP utilized in PyTorch. CIFAR10')
    main()
