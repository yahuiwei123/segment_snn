import argparse

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import test_dataset
from model import SegmentModel
from braincog.base.node.node import *
from torchvision import transforms
from braincog.utils import setup_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img_path", type=int, default=8, help="input image path")
    parser.add_argument("-s", "--step", type=int, default=8, help="time slice")
    parser.add_argument("-os", "--output_size", type=tuple, default=(128, 128), help="size of images(H, W)")
    parser.add_argument("-od", "--output_dir", type=str, default='./test/mask.png', help="output directory")
    args = parser.parse_args()

    setup_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    img_path = args.img_path
    step = args.step
    output_size = args.output_size
    output_dir = args.output_dir
    device = 'cuda:0'
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    # get image
    test_data = test_dataset(root=img_path)
    test_iter = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)


    net = SegmentModel(output_size=output_size, out_cls=21, node=BiasLIFNode, step=step)
    # load model
    net.load_state_dict(torch.load('./checkpoints/Segment_SNN.pth'))
    net = net.to(device)
    with torch.no_grad():
        for (idx, img) in enumerate(test_iter):
            logits = net(img.to(device))
            mask = torch.argmax(logits, dim=1)
            for m in mask:
                cv2.imwrite(f"{output_dir}/mask_{idx}.png", m.cpu().numpy())


