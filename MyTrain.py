import os
import csv
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.autograd import Variable

# 使用带“可学习 RA 权重”的模型版本（包含 ra_w*/ra_beta*）
from lib.PraNet_Res2Net_RAWeight import PraNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter


ALIGN_CORNERS = True  # 与模型/测试保持一致，减少插值差异


def log_ra_to_csv(model: torch.nn.Module, csv_path: str, epoch: int, step: int, tag: str = "epoch_end") -> None:
    """把 ra_w*/ra_beta* 的“有效值”(sigmoid 后)记录到 CSV。

    - 目的：证明可学习参数确实在训练中发生变化；
    - 用途：后续可以直接用这个 CSV 画曲线，写消融/可解释性。
    """
    if not hasattr(model, "get_ra_strength"):
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = {"epoch": epoch, "step": step, "tag": tag, **model.get_ra_strength()}
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # PyTorch 1.10: 使用 reduction= 而不是 reduce=
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def set_ra_mode(model: torch.nn.Module, ra_mode: str) -> None:
    """严格消融开关。

    - learnable: ra_w*/ra_beta* 参与训练（你的“可学习权重”实验）
    - fixed: 固定 ra_w*/ra_beta*，等价于“原版强度”

    注意：模型里实际用的是 sigmoid(raw_param) 作为有效值。
    要让有效值接近 1，我们把 raw_param 设成一个较大的正数（如 10.0），使 sigmoid(10)≈1。
    """
    if ra_mode not in {"learnable", "fixed"}:
        raise ValueError(f"Unknown ra_mode: {ra_mode}")

    target_names = {"ra_w4", "ra_w3", "ra_w2", "ra_beta4", "ra_beta3", "ra_beta2"}

    if ra_mode == "fixed":
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in target_names:
                    p.fill_(10.0)

        for n, p in model.named_parameters():
            if n in target_names:
                p.requires_grad = False
    else:
        for n, p in model.named_parameters():
            if n in target_names:
                p.requires_grad = True


def train(train_loader, model, optimizer, epoch, log_every: int = 0):
    model.train()
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=ALIGN_CORNERS)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=ALIGN_CORNERS)

            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss = loss2 + loss3 + loss4 + loss5

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            msg = ('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                   '[lateral-2: {:.4f}, lateral-3: {:.4f}, lateral-4: {:.4f}, lateral-5: {:.4f}]').format(
                datetime.now(), epoch, opt.epoch, i, total_step,
                loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()
            )
            print(msg)

        # 可选：训练中途也记录 ra 强度（默认关闭，避免 I/O 太频繁）
        if log_every and (i % log_every == 0):
            csv_path = os.path.join('snapshots', opt.train_save, 'ra_strength.csv')
            log_ra_to_csv(model, csv_path, epoch=epoch, step=(epoch - 1) * total_step + i, tag='step')

    # epoch 结束：记录一次 ra 强度 + 打印
    csv_path = os.path.join('snapshots', opt.train_save, 'ra_strength.csv')
    log_ra_to_csv(model, csv_path, epoch=epoch, step=epoch * total_step, tag='epoch_end')
    if hasattr(model, 'get_ra_strength'):
        print('[RA Strength]', model.get_ra_strength())

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'PraNet-%d.pth' % epoch))
        print('[Saving Snapshot:]', os.path.join(save_path, 'PraNet-%d.pth' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str, default='PraNet_Res2Net_RAWeight', help='snapshot folder name')

    # 第六点(2)：严格消融开关
    parser.add_argument('--ra_mode', type=str, default='learnable', choices=['learnable', 'fixed'],
                        help='learnable: ra_w*/ra_beta* 可学习；fixed: 固定为原版强度')

    # 第六点(1)：可选 step 级记录频率（0 表示只记录每个 epoch 结束）
    parser.add_argument('--ra_log_every', type=int, default=0,
                        help='log ra strength every N steps (0 = only at epoch end)')

    opt = parser.parse_args()

    model = PraNet().cuda()
    set_ra_mode(model, opt.ra_mode)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print('#' * 20, 'Start Training', '#' * 20)
    print('[RA Mode]', opt.ra_mode)

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, log_every=opt.ra_log_every)
