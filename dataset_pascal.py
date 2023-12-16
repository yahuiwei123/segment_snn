"""Pascal VOC Semantic Segmentation Dataset."""
import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import torch
import random
from torchvision import transforms
import torch.utils.data as data

class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


class VOCSegmentation(SegmentationDataset):
    """Pascal VOC Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/VOCdevkit'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    # >>> from torchvision import transforms
    # >>> import torch.utils.data as data
    # >>> # Transforms for Normalization
    # >>> input_transform = transforms.Compose([
    # >>>     transforms.ToTensor(),
    # >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    # >>> ])
    # >>> # Create Dataset
    # >>> trainset = VOCSegmentation(split='train', transform=input_transform)
    # >>> # Create Training Loader
    # >>> train_data = data.DataLoader(
    # >>>     trainset, 4, shuffle=True,
    # >>>     num_workers=4)
    """
    BASE_DIR = 'VOC2012'
    NUM_CLASS = 21

    def __init__(self, root='../../data_pascal_voc', split='train', mode=None, transform=None, **kwargs):
        super(VOCSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        _voc_root = os.path.join(root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if split != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if split != 'test':
            assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(len(self.images), _voc_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')

        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        # transform = transforms.ToTensor()
        # img = transform(img)
        dvs = self.generate_dynamic_translation(img)
        dvs = torch.as_tensor(dvs)
        frames = torch.diff(dvs, dim=3)
        p_img = torch.zeros_like(frames)
        n_img = torch.zeros_like(frames)
        p_img[frames > 0] = frames[frames > 0]
        n_img[frames < 0] = frames[frames < 0]
        output = torch.cat([p_img, n_img], dim=1)

        return output, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    # def tensor_PIL(self,img):
    #     img = Image.open(dataset.images[0]).convert("RGB")
    #     # 定义转换
    #     transform = transforms.ToTensor()
    #     img = transform(img)
    #     img = img.permute(1, 2, 0)
    #     return img

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')

    def generate_dynamic_translation(self,image):
        STRIDE = 1
        tracex = STRIDE * 2 * np.array([0, 2, 1, 0, 2, 1, 1, 2, 1])
        tracey = STRIDE * 2 * np.array([0, 1, 2, 1, 0, 2, 1, 1, 2])

        num_frames = len(tracex)
        height = image.shape[0]
        width = image.shape[1]
        channel = image.shape[2]
        frames = np.zeros((height, width, channel, num_frames))
        for i in range(num_frames):
            anchor_x = tracex[i]
            anchor_y = tracey[i]
            frames[anchor_y // 2: height - anchor_y // 2, anchor_x // 2: width - anchor_x // 2, :, i] = image[anchor_y:,anchor_x:, :]
        return frames




if __name__ == '__main__':
    input_transform = transforms.Compose([
             transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456,  0.406), (0.229, 0.224, 0.225)),
    ])
    # Create Dataset
    trainset = VOCSegmentation( split='val', transform=input_transform)
    # Create Training Loader
    train_data = data.DataLoader(trainset, 4, shuffle=True,)
