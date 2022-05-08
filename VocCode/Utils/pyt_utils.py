# encoding: utf-8
import time
import torch
import random
import torchvision
from collections import OrderedDict

# from engine.logger import get_logger

# logger = logging.getLogger()

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def load_model(model, model_file, is_restore=False):
    t_start = time.time()

    if model_file is None:
        return model

    if isinstance(model_file, str):
        state_dict = torch.load(model_file)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    del state_dict
    t_end = time.time()
    print(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model


class SquarePad:
    def __call__(self, image, output_size):

        if len(image.shape) == 4:
            w, h = image.shape[3], image.shape[2]
        else:
            w, h = image.shape[2], image.shape[1]

        o_w, o_h = output_size[0], output_size[1]
        hp = int((o_w - w) / 2)
        vp = int((o_h - h) / 2)
        padding = (hp, vp, hp, vp)
        return torchvision.transforms.functional.pad(image, padding,
                                                     fill=0, padding_mode='constant')


class PostAug(torch.nn.Module):
    def __init__(self, width_size, height_size):
        super(PostAug, self).__init__()
        self.width_size = width_size
        self.height_size = height_size
        self.zoom_in_area = [0.1, 0.3]
        self.zoom_out_area = [0.3, 0.5]
        self.image_resize = torchvision.transforms.Resize(size=(self.width_size, self.height_size),
                                                          interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

        self.mask_resize = torchvision.transforms.Resize(size=(self.width_size, self.height_size),
                                                         interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.square_pad = SquarePad()

    def zoom_in_operation(self, x, y, y_hat):
        i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(x, scale=self.zoom_in_area,
                                                                         ratio=[0.995, 1.333])
        x = torchvision.transforms.functional.crop(x, i, j, h, w)
        y = torchvision.transforms.functional.crop(y, i, j, h, w)

        y_hat = torchvision.transforms.functional.crop(y_hat, i, j, h, w) if y_hat is not None else None
        return self.image_resize(x), self.mask_resize(y), self.mask_resize(y_hat) if y_hat is not None else None

    def zoom_out_operation(self, x, y, y_hat):
        current_scale = random.uniform(self.zoom_out_area[0], self.zoom_out_area[1])
        x = torchvision.transforms.functional.resize(x, [int(self.height_size*current_scale),
                                                         int(self.width_size*current_scale)])
        x = self.square_pad(image=x, output_size=[self.width_size, self.height_size])

        y = torchvision.transforms.functional.resize(y, [int(self.height_size*current_scale),
                                                         int(self.width_size*current_scale)])
        y = self.square_pad(image=y, output_size=[self.width_size, self.height_size])

        if y_hat is not None:
            y_hat = torchvision.transforms.functional.resize(y_hat, [int(self.height_size*current_scale),
                                                                     int(self.width_size*current_scale)])

            y_hat = self.square_pad(image=y_hat, output_size=[self.width_size, self.height_size])

        return self.image_resize(x), self.mask_resize(y), self.mask_resize(y_hat) if y_hat is not None else None

    def forward(self, x, y, y_hat):
        dice_throw = random.uniform(.0, 1.)
        if dice_throw < .4:
            return self.zoom_in_operation(x, y, y_hat)
        elif .4 <= dice_throw < .6:
            return self.zoom_out_operation(x, y, y_hat)
        else:
            return x, y, y_hat

