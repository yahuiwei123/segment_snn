import sys
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

sys.path.append('../../..')
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import *
import time
from braincog.utils import setup_seed
from dataset import *
from braincog.base.node.node import *
from model import SegmentModel
from tqdm import tqdm
from torchmetrics import Dice
import os


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '/data/datasets'


def train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs, save_path='./checkpoints', losstype='mse'):
    best = 0
    net = net.to(device)
    print("training on ", device)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if losstype == 'mse':
        loss = torch.nn.MSELoss()
    else:
        loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    losses = []

    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        losss = []
        train_acc = []
        dice = Dice(average='micro').to(device)
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        tbar = tqdm(train_iter)
        for X, y in tbar:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            # X = torch.ones(6, 8, 2, 128, 128).to(device)
            # y = torch.ones(8, 13, 128, 128).to(device)
            y_hat = net(X)
            y_hat_argmax = torch.argmax(y_hat, dim=1)
            label = y
            label_resized = F.interpolate(label.unsqueeze(1).float(), size=(128, 128), mode='nearest').squeeze(1).long()
            l = loss(y_hat, label_resized)
            losss.append(l.cpu().item())
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            with torch.no_grad():
                acc = dice(y_hat_argmax.detach(), label_resized.detach())
                train_acc.append(acc.cpu().item())
            n += y.shape[0]
            batch_count += 1
            tbar.set_description(f'Epoch {epoch}: Loss = {l.cpu().item(): .4f}, dice = {acc.item():.4f}')
            scheduler.step()
        test_acc = evaluate_accuracy(test_iter, net)
        losses.append(np.mean(losss))
        print('epoch %d, lr %.6f, loss %.6f, train acc %.6f, test acc %.6f, time %.1f sec'
              % (epoch + 1, learning_rate, train_l_sum / batch_count, sum(train_acc) / len(train_acc), test_acc, time.time() - start))

        if test_acc > best:
            best = test_acc
            torch.save(net.state_dict(), os.path.join(save_path, '/SegmentModel.pth'))

def estimate_dice(gt_msk, prt_msk):
    intersection = gt_msk * prt_msk
    dice = 2 * float(intersection.sum()) / float(gt_msk.sum() + prt_msk.sum())
    return dice

def evaluate_accuracy(data_iter, net, device=None, only_onebatch=False):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    dice = Dice(average='micro').to(device)
    net.eval()
    with torch.no_grad():
        tbar = tqdm(data_iter)
        for X, y in tbar:
            logits = net(X.to(device))
            y = y.to(device)
#             softmax = nn.Softmax(dim=1)
#             logits = softmax(logits)
            acc = dice(logits, y.detach())
            acc_sum += acc.cpu().item()
            n += y.shape[0]
            tbar.set_description(f'Validation acc = {acc: .4f}')

            if only_onebatch: 
                break
    net.train()
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
