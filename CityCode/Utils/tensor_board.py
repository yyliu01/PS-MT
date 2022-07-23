import os
import PIL
import wandb
import numpy
import torch
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.helpers import DeNormalize


class Tensorboard:
    def __init__(self, config, online=False):
        os.environ['WANDB_API_KEY'] = ""
        os.system("wandb login")
        os.system("wandb {}".format("online" if online else "offline"))
        self.tensor_board = wandb.init(project=config['name'], name=config['experim_name'],
                                       config=config)

        self.restore_transform = torchvision.transforms.Compose([
            DeNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage()])

        self.palette = self.get_voc_pallete(19)
        self.current_step = 0
        self.root_dir = os.path.join(config['trainer']['save_dir'],
                                     config['experim_name'])

    def step_forward(self, global_step):
        self.current_step = global_step

    def upload_single_info(self, info):
        key, value = info.popitem()
        self.tensor_board.log({key: value,
                               "global_step": self.current_step})
        return

    def upload_wandb_info(self, info_dict):
        for i, info in enumerate(info_dict):
            self.tensor_board.log({info: info_dict[info],
                                   "global_step": self.current_step})
        return

    @staticmethod
    def get_class_colors(*args):
        return [[128, 64, 128], [244, 35, 232], [70, 70, 70],
                [102, 102, 156], [190, 153, 153], [153, 153, 153],
                [250, 170, 30], [220, 220, 0], [107, 142, 35],
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
                [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                [0, 0, 230], [119, 11, 32]]

    @staticmethod
    def set_img_color(colors, background, img, pred):
        for i in range(0, len(colors)):
            img[numpy.where(pred == i)] = colors[i]

        if len(pred[pred > len(colors)]) > 0:
            # original color
            if numpy.any(pred == 255):
                img[numpy.where(pred == 255)] = [0, 0, 0]
            # change to pure white; class == 19
            else:
                img[numpy.where(pred > len(colors))] = [255, 255, 255]
        return img / 255.0

    def update_wandb_city_image(self, images,
                                teacher_prediction,
                                ground_truth, img_number=4):

        img_number = min(ground_truth.shape[0], img_number)
        predict_mask_soft = torch.softmax(teacher_prediction, dim=1)
        predict_mask_hard = torch.argmax(predict_mask_soft, dim=1)
        predict_mask_soft = predict_mask_soft.max(1)[0]
        upload_weight = []
        upload_prediction = []
        upload_ground_truth = []

        for i in range(0, img_number):
            clean_pad = numpy.zeros_like(images[0].permute(1, 2, 0).cpu().numpy())
            upload_prediction.append(self.set_img_color(self.get_class_colors(), -1, clean_pad,
                                                        predict_mask_hard[i].detach().cpu().numpy()))

        for i in range(0, img_number):
            clean_pad = numpy.zeros_like(images[0].permute(1, 2, 0).cpu().numpy())
            upload_ground_truth.append(self.set_img_color(self.get_class_colors(), -1, clean_pad,
                                                          ground_truth[i].detach().cpu().numpy()))
        predict_mask_soft = predict_mask_soft.cpu()
        predict_mask_soft = predict_mask_soft.numpy()

        confident_heat_map_path = os.path.join(self.root_dir, 'confident_heat_map')
        teacher_prediction_path = os.path.join(self.root_dir, 'teacher_prediction')
        if not os.path.exists(confident_heat_map_path):
            os.mkdir(path=confident_heat_map_path)

        if not os.path.exists(teacher_prediction_path):
            os.mkdir(path=teacher_prediction_path)

        if not os.path.exists(os.path.join(confident_heat_map_path, 'step_{}'.format(str(self.current_step)))):
            os.mkdir(path=os.path.join(confident_heat_map_path, 'step_{}'.format(str(self.current_step))))

        if not os.path.exists(os.path.join(teacher_prediction_path, 'step_{}'.format(str(self.current_step)))):
            os.mkdir(path=os.path.join(teacher_prediction_path, 'step_{}'.format(str(self.current_step))))

        for i in range(0, img_number):
            _ = sns.heatmap(predict_mask_soft[i], vmin=.0, vmax=1., yticklabels=False, xticklabels=False,
                            cbar=True if i == img_number-1 else False)
            plt.savefig(os.path.join(confident_heat_map_path,
                        'step_{}/heatmap_{}'.format(str(self.current_step), str(i)) + '.png'))

            plt.clf()
            upload_weight.append(plt.imread(os.path.join(confident_heat_map_path,
                                            'step_{}/heatmap_{}.png'.format(str(self.current_step), str(i)))))

        upload_weight = numpy.asarray(upload_weight)
        upload_prediction = numpy.asarray(upload_prediction)
        wandb.log({"confident_weight": [wandb.Image(j, caption="id {}".format(str(i)))
                                        for i, j in enumerate(upload_weight)], "my_step": self.current_step})

        wandb.log({"teacher_prediction": [wandb.Image(j, caption="id {}".format(str(i)))
                                          for i, j in enumerate(upload_prediction)]})

        wandb.log({"ground_truth": [wandb.Image(j, caption="id {}".format(str(i)))
                                    for i, j in enumerate(upload_ground_truth)]})
        wandb.log({"images": [wandb.Image(j, caption="id {}".format(str(i)))
                              for i, j in enumerate(images[:img_number])], "my_step": self.current_step})

    def update_table(self, table_info, axis_name, title=""):
        x_name, y_name = axis_name['x'], axis_name['y']
        table = wandb.Table(data=table_info,
                            columns=[x_name, y_name])
        wandb.log({title: wandb.plot.bar(table, x_name, y_name, title=title)})

    def de_normalize(self, image):
        return [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                else self.colorize_mask(i.numpy(), self.palette)
                for i in image.detach().cpu()]

    def colorize_mask(self, mask, palette):
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        palette[-3:] = [255, 255, 255]
        new_mask = PIL.Image.fromarray(mask.astype(numpy.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask

    @staticmethod
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

    @staticmethod
    def finish():
        wandb.finish()
