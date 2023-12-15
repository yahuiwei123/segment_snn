import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../../..')
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import *
import time
from braincog.utils import setup_seed
from dataset import SegmentationDataset
from braincog.base.node.node import *
from model import SegmentModel

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '/data/datasets'


def train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs, losstype='mse'):
    best = 0
    net = net.to(device)
    print("training on ", device)
    if losstype == 'mse':
        loss = torch.nn.MSELoss()
    else:
        loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    losses = []

    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        losss = []
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            optimizer.zero_grad()
            # X = X.to(device)
            # y = y.to(device)
            X = torch.ones(6, 8, 2, 128, 128).to(device)
            y = torch.ones(8, 13, 128, 128).to(device)
            y_hat = net(X)
            label = y
            l = loss(y_hat, label)
            losss.append(l.cpu().item())
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            # train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        scheduler.step()
        test_acc = evaluate_accuracy(test_iter, net)
        losses.append(np.mean(losss))
        print('epoch %d, lr %.6f, loss %.6f, train acc %.6f, test acc %.6f, time %.1f sec'
              % (epoch + 1, learning_rate, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

        if test_acc > best:
            best = test_acc
            torch.save(net.state_dict(), './checkpoints/CIFAR10_VGG16.pth')


def evaluate_accuracy(data_iter, net, device=None, only_onebatch=False):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]

            if only_onebatch: break
    return acc_sum / n


if __name__ == '__main__':
    setup_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batch_size = 8
    step = 8

    # get coco dataset
    train_dataset = SegmentationDataset()
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = SegmentationDataset()
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print('dataloader finished')

    lr, num_epochs = 0.01, 300
    net = SegmentModel(output_size=(128, 128), out_cls=train_dataset.num_class, node=BiasLIFNode, step=step)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs,
          losstype='crossentropy')  # 'crossentropy')

    # net.load_state_dict(torch.load("./CIFAR10_VGG16.pth", map_location=device))
    net = net.to(device)
    acc = evaluate_accuracy(test_iter, net, device)
    print(acc)
