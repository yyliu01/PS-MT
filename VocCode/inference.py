import argparse

from DataLoader.voc import VOC
import json
from Utils.helpers import colorize_mask
import torch
import torchvision
import torch.nn.functional as F
import os
from tqdm import tqdm
from math import ceil
from Model.Deeplabv3_plus.EntireModel import EntireModel as model_deep
import matplotlib.pyplot as plt
import seaborn as sns

voc_class_name = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                  'cat', 'chair', 'cow', 'dining_table', 'dog', 'horse', 'motorbike', 'person',
                  'potted_plant', 'sheep', 'sofa', 'train', 'tv_monitor']


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


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


palette = get_voc_pallete(21)
restore_transform = torchvision.transforms.Compose([
    DeNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage()])


def running_inference(loader, model, folder_name, save_img=False):
    tbar = tqdm(loader, ncols=100)
    statistic_conf_bar = torch.zeros([21, ], dtype=torch.float).cuda(non_blocking=True)
    num_record = torch.zeros([21, ], dtype=torch.float).cuda(non_blocking=True)
    for batch_idx, (data, target) in enumerate(tbar):
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        H, W = target.size(1), target.size(2)
        up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
        data = torch.nn.functional.interpolate(data, size=(up_sizes[0], up_sizes[1]),
                                               mode='bilinear', align_corners=True)
        with torch.no_grad():
            prediction = model.module.decoder1(model.module.encoder1(data),
                                               data_shape=[data.shape[-2], data.shape[-1]],
                                               req_feature=True)

        prediction = torch.nn.functional.interpolate(prediction, size=(H, W),
                                                     mode='bilinear', align_corners=True)

        conf_result = torch.softmax(prediction.squeeze(), dim=0).max(0)[0]
        hard_result = torch.argmax(prediction.squeeze().squeeze(), dim=0)
        false_mask = hard_result != target.squeeze()

        for i in range(0, 21):
            idx_mask = i == target.squeeze()
            mask = idx_mask & false_mask
            num_record[i] += torch.sum(mask)
            statistic_conf_bar[i] += torch.sum(conf_result[mask])

        if save_img:
            prediction = prediction.squeeze().detach()
            _ = sns.heatmap(torch.softmax(prediction, dim=0).max(0)[0].cpu().numpy(),
                            vmin=.0, vmax=1., yticklabels=False, xticklabels=False, cbar=True)
            plt.savefig(os.path.join(folder_name, str(batch_idx) + "_conf" + ".png"))
            plt.clf()

            prediction = torch.argmax(prediction, dim=0).cpu().numpy()
            prediction_im = colorize_mask(prediction, palette)
            prediction_im.save(os.path.join(folder_name, str(batch_idx) + "_pred" + ".png"))

            gt = colorize_mask(target.detach().squeeze().cpu().numpy(), palette)
            gt.save(os.path.join(folder_name, str(batch_idx) + "_gt" + ".png"))
            data = restore_transform(data.squeeze())
            data.save(os.path.join(folder_name, str(batch_idx) + "_img" + ".png"))

    plt.xlim(-1, 22)
    plt.xticks(ticks=range(0, 21), labels=voc_class_name, rotation='vertical')
    plt.bar(range(0, 21), (statistic_conf_bar/num_record).detach().cpu().numpy(), width=0.5)
    plt.savefig(os.path.join(folder_name, "static_overconfident.png"))


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='your path to configs', type=str,
                        help='Path to the config file')
    parser.add_argument('--model', default="your path to mode", type=str,
                        help='Path to the trained .pth model')
    parser.add_argument('--save', action='store_true', help='Save images')
    parser.add_argument('--backbone', default=int, help='backbone in use')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    config = json.load(open(args.config))
    # DATA
    loader = VOC(config['val_loader'])
    config['model']['resnet'] = args.backbone
    config['model']['data_h_w'] = [0, 0]

    # MODEL
    model = model_deep(num_classes=21, config=config['model'],
                       sup_loss=None, cons_w_unsup=None,
                       ignore_index=255)

    checkpoint = torch.load(args.model)
    model = torch.nn.DataParallel(model)
    try:
        model.module.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.module.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    model.cuda()

    folder_name = os.path.join("saved", args.model.split('/')[-1].split('.')[0])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    running_inference(loader=loader, model=model, folder_name=folder_name, save_img=args.save)


if __name__ == '__main__':
    main()
