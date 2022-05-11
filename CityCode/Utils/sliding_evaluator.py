import torch
import numpy
import cv2
import collections


class SlidingEval(torch.nn.Module):
    def __init__(self, model, crop_size, stride_rate, device, class_number=19, val_id=1):
        super(SlidingEval, self).__init__()
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.device = device
        self.class_number = class_number
        self.model = model
        self.val_id = val_id

    def forward(self, img):
        img = img.squeeze().permute(1, 2, 0).cpu().numpy()
        ori_rows, ori_cols, c = img.shape
        processed_pred = numpy.zeros((ori_rows, ori_cols, self.class_number))
        processed_pred += self.scale_process(img, (ori_rows, ori_cols), self.device)
        return processed_pred.transpose(2, 0, 1)

    # The img is normalized;
    def process_image(self, img, crop_size=None):
        p_img = img
        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = numpy.concatenate((im_b, im_g, im_r), axis=2)

        if crop_size is not None:
            p_img, margin = self.pad_image_to_shape(p_img, crop_size,
                                                    cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)
            return p_img, margin

        p_img = p_img.transpose(2, 0, 1)
        return p_img

    def get_2dshape(self, shape, *, zero=True):
        if not isinstance(shape, collections.Iterable):
            shape = int(shape)
            shape = (shape, shape)
        else:
            h, w = map(int, shape)
            shape = (h, w)
        if zero:
            minv = 0
        else:
            minv = 1

        assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
        return shape

    def pad_image_to_shape(self, img, shape, border_mode, value):
        margin = numpy.zeros(4, numpy.uint32)
        shape = self.get_2dshape(shape)
        pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
        pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

        margin[0] = pad_height // 2
        margin[1] = pad_height // 2 + pad_height % 2
        margin[2] = pad_width // 2
        margin[3] = pad_width // 2 + pad_width % 2

        img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                                 border_mode, value=value)

        return img, margin

    def scale_process(self, img, ori_shape, device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows
        if isinstance(self.crop_size, int):
            self.crop_size = (self.crop_size, self.crop_size)

        if long_size <= min(self.crop_size[0], self.crop_size[1]):
            input_data, margin = self.process_image(img, self.crop_size)  # pad image
            with torch.no_grad():
                input_data = torch.tensor(input_data, dtype=torch.float).cuda().unsqueeze(0)
                if self.val_id == 1:
                    score = self.model.module(input_data, id=1)
                else:
                    score = (self.model.module(input_data, id=1) + self.model.module(input_data, id=2))/2
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride_0 = int(numpy.ceil(self.crop_size[0] * self.stride_rate))
            stride_1 = int(numpy.ceil(self.crop_size[1] * self.stride_rate))
            img_pad, margin = self.pad_image_to_shape(img, self.crop_size,
                                                      cv2.BORDER_CONSTANT, value=0)
            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(numpy.ceil((pad_rows - self.crop_size[0]) / stride_0)) + 1
            c_grid = int(numpy.ceil((pad_cols - self.crop_size[1]) / stride_1)) + 1
            data_scale = torch.zeros(self.class_number, pad_rows, pad_cols).cuda(
                device)
            count_scale = torch.zeros(self.class_number, pad_rows, pad_cols).cuda(
                device)
            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride_1
                    s_y = grid_yidx * stride_0
                    e_x = min(s_x + self.crop_size[1], pad_cols)
                    e_y = min(s_y + self.crop_size[0], pad_rows)
                    s_x = e_x - self.crop_size[1]
                    s_y = e_y - self.crop_size[0]
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1
                    input_data, tmargin = self.process_image(img_sub, self.crop_size)
                    input_data = torch.tensor(input_data, dtype=torch.float).cuda().unsqueeze(0)
                    temp_score = self.model.module.decoder1(self.model.module.encoder1(input_data),
                                                            data_shape=[input_data.shape[-2],
                                                                        input_data.shape[-1]])
                    temp_score = temp_score.squeeze()
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score

            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                          margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output
