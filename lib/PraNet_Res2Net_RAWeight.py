import torch
import torch.nn as nn
import torch.nn.functional as F

from .Res2Net_v1b import res2net50_v1b_26w_4s

# PyTorch 1.10 兼容：显式指定 align_corners，避免不同版本插值行为差异
ALIGN_CORNERS = True


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, used after MSF
    def __init__(self, channel):
        super().__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(
            self.upsample(x2)
        ) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x


class RAGate(nn.Module):
    """样本自适应 gating：根据(本层feature, 上层crop)预测 w 和 beta。

    输入：
      - feat: (B, C, H, W)
      - crop: (B, 1, H, W)  (已经对齐到本层分辨率)
    输出：
      - w:    (B, 1, 1, 1) in (0,1)
      - beta: (B, 1, 1, 1) in (0,1)

    设计要点：
      1) 用 GAP(feat) 表示“本层语义/纹理状态”
      2) 用 GAP(sigmoid(crop)) 表示“上层预测置信度”
      3) 两者拼接 -> 小 MLP -> 输出两个 logit
      4) 初始化让 w,beta 一开始接近 1（尽量等价原版），训练更稳
    """

    def __init__(self, in_channels: int, hidden: int = 128, init_logit: float = 5.0):
        super().__init__()
        self.fc1 = nn.Linear(in_channels + 1, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, 2)

        # 初始化：让输出一开始接近 init_logit（sigmoid≈1）
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, init_logit)

    def forward(self, feat: torch.Tensor, crop: torch.Tensor):
        b, c, _, _ = feat.shape
        feat_gap = F.adaptive_avg_pool2d(feat, 1).view(b, c)
        crop_conf = torch.sigmoid(crop)
        crop_gap = F.adaptive_avg_pool2d(crop_conf, 1).view(b, 1)
        z = torch.cat([feat_gap, crop_gap], dim=1)
        h = self.act(self.fc1(z))
        logits = self.fc2(h)
        w = torch.sigmoid(logits[:, 0]).view(b, 1, 1, 1)
        beta = torch.sigmoid(logits[:, 1]).view(b, 1, 1, 1)
        return w, beta


class PraNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, use_dynamic_gate: bool = False, gate_hidden: int = 128):
        super().__init__()
        self.use_dynamic_gate = use_dynamic_gate

        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)

        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)

        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)

        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        # ---------- 静态 gating（你现有的方案，但改为 logit 参数 + 更合理初始化） ----------
        # 原来你写的是 Parameter(1.0) 然后 sigmoid -> 0.731
        # 这里用 logit=5.0，sigmoid(5)=0.993，起步更接近“原版强 RA”
        init_logit = 5.0
        self.ra_logit_w4 = nn.Parameter(torch.tensor(init_logit))
        self.ra_logit_w3 = nn.Parameter(torch.tensor(init_logit))
        self.ra_logit_w2 = nn.Parameter(torch.tensor(init_logit))
        self.ra_logit_b4 = nn.Parameter(torch.tensor(init_logit))
        self.ra_logit_b3 = nn.Parameter(torch.tensor(init_logit))
        self.ra_logit_b2 = nn.Parameter(torch.tensor(init_logit))

        # ---------- 动态 gating（按样本自适应：推荐做法） ----------
        # 注意：这是“样本级标量 gating”，工作量小、可解释强、一般比较稳。
        self.gate4 = RAGate(2048, hidden=gate_hidden, init_logit=init_logit)
        self.gate3 = RAGate(1024, hidden=gate_hidden, init_logit=init_logit)
        self.gate2 = RAGate(512, hidden=gate_hidden, init_logit=init_logit)

        # 供训练脚本读取（用于 gate_reg 正则/可视化）
        self._last_gates = {}

    def get_gate_values(self):
        """返回最近一次 forward 中的 w/beta（带梯度），用于额外正则或 logging。"""
        return self._last_gates

    def _get_w_beta(self, stage: int, feat: torch.Tensor, crop: torch.Tensor):
        """统一获得某一层的 (w, beta) ，支持静态/动态两种模式。"""
        if self.use_dynamic_gate:
            if stage == 4:
                w, beta = self.gate4(feat, crop)
            elif stage == 3:
                w, beta = self.gate3(feat, crop)
            elif stage == 2:
                w, beta = self.gate2(feat, crop)
            else:
                raise ValueError(f"Unsupported stage: {stage}")
        else:
            if stage == 4:
                w = torch.sigmoid(self.ra_logit_w4).view(1, 1, 1, 1)
                beta = torch.sigmoid(self.ra_logit_b4).view(1, 1, 1, 1)
            elif stage == 3:
                w = torch.sigmoid(self.ra_logit_w3).view(1, 1, 1, 1)
                beta = torch.sigmoid(self.ra_logit_b3).view(1, 1, 1, 1)
            elif stage == 2:
                w = torch.sigmoid(self.ra_logit_w2).view(1, 1, 1, 1)
                beta = torch.sigmoid(self.ra_logit_b2).view(1, 1, 1, 1)
            else:
                raise ValueError(f"Unsupported stage: {stage}")

            # 静态时扩到 batch
            b = feat.size(0)
            w = w.expand(b, 1, 1, 1)
            beta = beta.expand(b, 1, 1, 1)

        return w, beta

    def forward(self, x):
        in_size = x.shape[2:]

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88

        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)  # (bs, 1, 44, 44)
        lateral_map_5 = F.interpolate(ra5_feat, size=in_size, mode='bilinear', align_corners=ALIGN_CORNERS)

        # ---------------- RA stage 4 ----------------
        crop_4 = F.interpolate(ra5_feat, size=x4.shape[2:], mode='bilinear', align_corners=ALIGN_CORNERS)
        w4, b4 = self._get_w_beta(stage=4, feat=x4, crop=crop_4)

        mask4 = 1 - w4 * torch.sigmoid(crop_4)
        x = mask4.expand(-1, 2048, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)

        x = b4 * ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, size=in_size, mode='bilinear', align_corners=ALIGN_CORNERS)

        # ---------------- RA stage 3 ----------------
        crop_3 = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=ALIGN_CORNERS)
        w3, b3 = self._get_w_beta(stage=3, feat=x3, crop=crop_3)

        mask3 = 1 - w3 * torch.sigmoid(crop_3)
        x = mask3.expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)

        x = b3 * ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, size=in_size, mode='bilinear', align_corners=ALIGN_CORNERS)

        # ---------------- RA stage 2 ----------------
        crop_2 = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=ALIGN_CORNERS)
        w2, b2 = self._get_w_beta(stage=2, feat=x2, crop=crop_2)

        mask2 = 1 - w2 * torch.sigmoid(crop_2)
        x = mask2.expand(-1, 512, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)

        x = b2 * ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, size=in_size, mode='bilinear', align_corners=ALIGN_CORNERS)

        # 记录（用于 train 脚本的 gate_reg 正则或你想画曲线）
        self._last_gates = {
            'w4': w4,
            'w3': w3,
            'w2': w2,
            'b4': b4,
            'b3': b3,
            'b2': b2,
        }

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2


if __name__ == '__main__':
    net = PraNet(use_dynamic_gate=True).cuda()
    x = torch.randn(2, 3, 352, 352).cuda()
    y = net(x)
    print([t.shape for t in y])
    print({k: v.mean().item() for k, v in net.get_gate_values().items()})
