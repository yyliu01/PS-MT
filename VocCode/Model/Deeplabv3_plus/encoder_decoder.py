import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
# from config import config
from Model.Deeplabv3_plus.Backbones.resnet import resnet50, resnet101
bn_eps = 1e-5
bn_momentum = 0.1


class EncoderNetwork(nn.Module):
    def __init__(self, num_classes, norm_layer=nn.BatchNorm2d, back_bone=None, pretrained_model=None):
        super(EncoderNetwork, self).__init__()
        if back_bone is 101:
            self.backbone = resnet101(pretrained_model, norm_layer=norm_layer,
                                      bn_eps=bn_eps,
                                      bn_momentum=bn_momentum,
                                      deep_stem=True, stem_width=64)
        else:
            self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                     bn_eps=bn_eps,
                                     bn_momentum=bn_momentum,
                                     deep_stem=True, stem_width=64)

        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.head = Head(num_classes, norm_layer, bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.head)

    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, data):
        blocks = self.backbone(data)
        f = self.head(blocks)
        return f


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv_t(x, decoder1, decoder2, it=1, xi=1e-1, eps=10.0):
    # stop bn
    decoder1.eval()
    decoder2.eval()
    
    x_detached = x.detach()
    with torch.no_grad():
        # get the ensemble results from teacher
        pred = F.softmax((decoder1(x_detached) + decoder2(x_detached))/2, dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    # assist students to find the effective va-noise
    for _ in range(it):
        d.requires_grad_()
        pred_hat = (decoder1(x_detached + xi * d) + decoder2(x_detached + xi * d))/2
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder1.zero_grad()
        decoder2.zero_grad()

    r_adv = d * eps

    # reopen bn, but freeze other params.
    # https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/16
    decoder1.train()
    decoder2.train()
    return r_adv


class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, data_shape,
                 norm_act=nn.BatchNorm2d):
        super(upsample, self).__init__()
        self.data_shape = data_shape
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU())

    def forward(self, x, data_shape=None):
        f = self.last_conv(x)
        pred = self.classifier(f)
        if self.training:
            h, w = self.data_shape[0], self.data_shape[1]
        else:
            if data_shape is not None:
                h, w = data_shape[0], data_shape[1]
            else:
                h, w = self.data_shape[0], self.data_shape[1]

        return F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)


class DecoderNetwork(nn.Module):
    def __init__(self, num_classes,
                 data_shape,
                 conv_in_ch=256):

        super(DecoderNetwork, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, norm_act=torch.nn.BatchNorm2d,
                                 data_shape=data_shape)
        self.business_layer = []
        self.business_layer.append(self.upsample.last_conv)
        self.business_layer.append(self.upsample.classifier)

    def forward(self, f, data_shape=None):
        pred = self.upsample(f, data_shape)
        return pred


class VATDecoderNetwork(nn.Module):
    def __init__(self, num_classes,
                 data_shape,
                 conv_in_ch=256):

        super(VATDecoderNetwork, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, norm_act=torch.nn.BatchNorm2d,
                                 data_shape=data_shape)
        self.business_layer = []
        self.business_layer.append(self.upsample.last_conv)
        self.business_layer.append(self.upsample.classifier)

    def forward(self, f, data_shape=None, t_model=None):
        if t_model is not None:
            r_adv = get_r_adv_t(f, t_model[0], t_model[1], it=1, xi=1e-6, eps=2.0)
            f += r_adv

        pred = self.upsample(f, data_shape)
        return pred


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.leak_relu(pool)  # add activation layer
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            raise NotImplementedError
        return pool


class Head(nn.Module):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )

    def forward(self, f_list):
        f = f_list[-1]
        f = self.aspp(f)
        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)
        f = F.interpolate(f, size=(low_h, low_w),
                          mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        return f


