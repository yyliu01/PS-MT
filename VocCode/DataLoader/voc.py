from Base.base_dataset import BaseDataSet
from Base.base_dataloader import BaseDataLoader

import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset


def get_voc_pallete(num_classes):
    n = num_classes
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


class VOCDataset(BaseDataSet):
    def __init__(self, ddp_training, dgx, **kwargs):
        self.num_classes = 21
        self.ddp_training = ddp_training
        self.dgx = dgx
        self.palette = get_voc_pallete(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')
        prefix = "VocCode"
#if self.dgx else ""
        if self.split == "val":
            file_list = os.path.join(prefix, "DataLoader/voc_splits", f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join(prefix, "DataLoader/voc_splits", f"{self.n_labeled_examples}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id + ".png")
        else:
            label_path = os.path.join(self.root, self.labels[index][1:])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image, label, image_id


class VOC(BaseDataLoader):
    def __init__(self, kwargs, ddp_training=False, dgx=False):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')
        self.dataset = VOCDataset(ddp_training, **kwargs, dgx=dgx)
        if ddp_training:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            train_sampler = None
        super(VOC, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,
                                  sampler=train_sampler)

