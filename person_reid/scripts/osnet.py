#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OSNet (Omni-Scale Network) 的纯 PyTorch 实现。

参考:
  Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV 2019.
  Zhou et al. Learning Generalisable Omni-Scale Representations. TPAMI 2021.

为什么要自己写一遍而不用 torchreid:
  1. torchreid 的预训练权重托管在 Google Drive, 国内无法访问;
     我们从 hf-mirror 下载权重后, 只需要网络结构本身。
  2. torchreid 0.2.5 依赖 gdown / 老版本 Cython 扩展, 在 Python 3.8 + torch 2.4 上
     安装容易失败。这里只用 torch 原生算子, 零额外依赖。
  3. 本文件与 torchreid 的层名逐一对齐, 可以直接 load 官方 state_dict (strict=True)。

输入约定: (B, 3, 256, 128)  BGR/RGB 均可, 但要与训练时一致(RGB)且做 ImageNet 归一化。
输出约定: eval 模式下 forward 返回 (B, 512) 的 L2 未归一化特征向量。
"""

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["OSNet", "osnet_x1_0", "osnet_x0_75", "osnet_x0_5", "osnet_x0_25",
           "load_pretrained"]


# --------------------------------------------------------------------------
# 基础层
# --------------------------------------------------------------------------
class ConvLayer(nn.Module):
    """conv + bn + relu"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False,
                              groups=groups)
        # IN=True 时用 InstanceNorm 替代 BatchNorm(OSNet-AIN 用), x1_0 用不到
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if IN \
            else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 conv + bn + relu"""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                              padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 conv + bn (无激活)"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                              padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """轻量 3x3: 1x1 线性 + 3x3 深度可分离"""

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1,
                               padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


# --------------------------------------------------------------------------
# 全尺度特征学习模块
# --------------------------------------------------------------------------
class ChannelGate(nn.Module):
    """通道注意力门控: 全局池化 -> fc1 -> relu -> fc2 -> sigmoid, 再逐通道加权。

    reduction=16: fc1 把 mid_channels 压到 mid_channels//16 再升回来。
    num_gates 默认等于 in_channels, 输出与输入同通道, 用于残差式加权。
    """

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction,
                             kernel_size=1, bias=True, padding=0)
        self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1)) \
            if layer_norm else None
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates,
                             kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        identity = x
        g = self.global_avgpool(x)
        g = self.fc1(g)
        if self.norm1 is not None:
            g = self.norm1(g)
        g = self.relu(g)
        g = self.fc2(g)
        if self.gate_activation is not None:
            g = self.gate_activation(g)
        if self.return_gates:
            return g
        return identity * g


class OSBlock(nn.Module):
    """Omni-Scale 残差块。

    4 条不同感受野的分支(1/2/3/4 个 LightConv3x3)各自过门控后相加,
    再经 1x1 升维, 与 identity 相加。
    """

    def __init__(self, in_channels, out_channels, IN=False,
                 bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if IN else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        # 每条分支单独过门控(与官方实现一致, 不能先加后门控)
        x2 = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1))
              + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


# --------------------------------------------------------------------------
# 网络
# --------------------------------------------------------------------------
class OSNet(nn.Module):
    def __init__(self, num_classes, blocks, layers, channels,
                 feature_dim=512, loss='softmax', IN=False, **kwargs):
        super(OSNet, self).__init__()
        assert len(blocks) == len(layers) == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0],
                                      channels[1], reduce_spatial_size=True, IN=IN)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1],
                                      channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2],
                                      channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(feature_dim, channels[3], dropout_p=None)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels,
                    reduce_spatial_size, IN=False):
        layers = [block(in_channels, out_channels, IN=IN)]
        for _ in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))
        if reduce_spatial_size:
            # 过渡层: 1x1 升/降维 + 2x2 平均池化(下采样)
            layers.append(nn.Sequential(
                Conv1x1(out_channels, out_channels),
                nn.AvgPool2d(2, stride=2)
            ))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        raise KeyError("Unsupported loss: {}".format(self.loss))


# --------------------------------------------------------------------------
# 构造器
# --------------------------------------------------------------------------
def _osnet(blocks, layers, channels, num_classes=1000, **kwargs):
    return OSNet(num_classes=num_classes, blocks=blocks, layers=layers,
                 channels=channels, **kwargs)


def osnet_x1_0(num_classes=1000, **kwargs):
    return _osnet([OSBlock] * 3, [2, 2, 2], [64, 256, 384, 512],
                  num_classes=num_classes, **kwargs)


def osnet_x0_75(num_classes=1000, **kwargs):
    return _osnet([OSBlock] * 3, [2, 2, 2], [48, 192, 384, 512],
                  num_classes=num_classes, **kwargs)


def osnet_x0_5(num_classes=1000, **kwargs):
    return _osnet([OSBlock] * 3, [2, 2, 2], [32, 128, 256, 512],
                  num_classes=num_classes, **kwargs)


def osnet_x0_25(num_classes=1000, **kwargs):
    return _osnet([OSBlock] * 3, [2, 2, 2], [16, 64, 128, 512],
                  num_classes=num_classes, **kwargs)


# 各结构的 checkpoint 里的分类头维度(用于自动推断 num_classes)
_ARCH = {
    "osnet_x1_0": (osnet_x1_0, [2, 2, 2], [64, 256, 384, 512]),
    "osnet_x0_75": (osnet_x0_75, [2, 2, 2], [48, 192, 384, 512]),
    "osnet_x0_5": (osnet_x0_5, [2, 2, 2], [32, 128, 256, 512]),
    "osnet_x0_25": (osnet_x0_25, [2, 2, 2], [16, 64, 128, 512]),
}


def load_pretrained(path, arch="osnet_x1_0", num_classes=None, device="cpu"):
    """加载 torchreid 格式的 OSNet 权重。

    num_classes 留空时从 checkpoint 的 classifier.weight 第一维自动推断,
    这样无论权重是在 Market-1501(751类) 还是 MSMT17(4101类) 上训练的都能直接load。
    """
    if arch not in _ARCH:
        raise ValueError("未知结构 %r, 可选: %s" % (arch, list(_ARCH)))
    builder = _ARCH[arch][0]

    sd = torch.load(path, map_location="cpu")
    # 兼容被包在 {'state_dict': ...} 里的 checkpoint
    if isinstance(sd, dict) and "state_dict" in sd and \
            not any(k.startswith("conv1.") for k in sd):
        sd = sd["state_dict"]
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
          for k, v in sd.items()}

    if num_classes is None:
        w = sd.get("classifier.weight")
        num_classes = int(w.shape[0]) if w is not None else 1000

    model = builder(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # classifier/fc 之外的关键层一个都不能少, 否则说明结构与权重不匹配
    critical = [k for k in missing if not k.startswith("classifier.")]
    if critical:
        raise RuntimeError(
            "权重与结构不匹配, 缺少 %d 个关键参数, 例如: %s"
            % (len(critical), critical[:5]))
    model.to(device).eval()
    return model, {"num_classes": num_classes,
                   "missing": list(missing),
                   "unexpected": list(unexpected)}
