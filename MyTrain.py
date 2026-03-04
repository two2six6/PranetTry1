import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.autograd import Variable

# 你的模型文件（保持 import 路径不变）
from lib.PraNet_Res2Net_RAWeight import PraNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter


def structure_loss(pred, mask):
    """PraNet 原论文常用的结构损失：加权 BCE + 加权 IoU。"""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


@torch.no_grad()
def evaluate(loader, model, trainsize):
    """用同一个 loss 在验证集上评估，用于 early stopping。"""
    model.eval()
    loss_meter = AvgMeter()
    for images, gts in loader:
        images = images.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)
        # 保持和训练一致的输入分辨率
        if images.shape[-1] != trainsize:
            images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

        l5, l4, l3, l2 = model(images)
        loss = (structure_loss(l5, gts) + structure_loss(l4, gts) +
                structure_loss(l3, gts) + structure_loss(l2, gts))
        loss_meter.update(loss.item(), images.size(0))
    return loss_meter.show()


def build_optimizer(model, base_lr, gate_lr_mult=10.0, weight_decay=0.0):
    """把 gating 参数单独分组，提高学习率（默认 ×10）。

    规则：名字里包含 'gate' 或 'ra_logit' 的参数认为是 gating 相关。
    """
    gate_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ('gate' in name) or ('ra_logit' in name):
            gate_params.append(p)
        else:
            other_params.append(p)

    param_groups = [
        {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay},
        {'params': gate_params, 'lr': base_lr * gate_lr_mult, 'weight_decay': weight_decay},
    ]
    optimizer = torch.optim.Adam(param_groups)
    return optimizer


def save_checkpoint(path, model, optimizer, epoch, best_loss, bad_epochs, opt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss,
        'bad_epochs': bad_epochs,
        'opt': vars(opt),
        'time': str(datetime.now())
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = int(ckpt.get('epoch', 0))
    best_loss = float(ckpt.get('best_loss', float('inf')))
    bad_epochs = int(ckpt.get('bad_epochs', 0))
    return epoch, best_loss, bad_epochs


def train_one_epoch(train_loader, model, optimizer, opt, epoch):
    model.train()
    size_rates = [0.75, 1, 1.25] if opt.ms_train else [1]

    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad(set_to_none=True)

            images, gts = pack
            images = Variable(images).cuda(non_blocking=True)
            gts = Variable(gts).cuda(non_blocking=True)

            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            l5, l4, l3, l2 = model(images)

            loss5 = structure_loss(l5, gts)
            loss4 = structure_loss(l4, gts)
            loss3 = structure_loss(l3, gts)
            loss2 = structure_loss(l2, gts)
            loss = loss2 + loss3 + loss4 + loss5

            # 可选：让 gating 参数“先学会原版”，再逐渐偏离（稳定性更好）
            # reg = λ * Σ_k [(1-w_k)^2 + (1-β_k)^2]
            if opt.gate_reg > 0:
                gates = model.get_gate_values()  # dict of tensors
                reg = 0.0
                for k in ('w4', 'w3', 'w2', 'b4', 'b3', 'b2'):
                    if k in gates:
                        reg = reg + (1.0 - gates[k]).pow(2).mean()
                loss = loss + opt.gate_reg * reg

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                bs = images.size(0)
                loss_record2.update(loss2.item(), bs)
                loss_record3.update(loss3.item(), bs)
                loss_record4.update(loss4.item(), bs)
                loss_record5.update(loss5.item(), bs)

        if i % opt.print_freq == 0 or i == len(train_loader):
            print(
                '{} Epoch [{:03d}/{:03d}] Step [{:04d}/{:04d}] '
                '[l2:{:.4f} l3:{:.4f} l4:{:.4f} l5:{:.4f}]'.format(
                    datetime.now(), epoch, opt.epochs, i, len(train_loader),
                    loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()
                )
            )

    # 用训练集的“主尺度”loss 作为 train-loss（用于没 val 时的 early stop）
    train_loss = loss_record2.show() + loss_record3.show() + loss_record4.show() + loss_record5.show()
    return float(train_loss)


def main(opt):
    torch.backends.cudnn.benchmark = True

    # ---- build model ----
    model = PraNet(channel=opt.channel, use_dynamic_gate=opt.dynamic_gate, gate_hidden=opt.gate_hidden).cuda()

    optimizer = build_optimizer(
        model,
        base_lr=opt.lr,
        gate_lr_mult=opt.gate_lr_mult,
        weight_decay=opt.weight_decay,
    )

    # ---- data ----
    train_image_root = os.path.join(opt.train_path, 'images')
    train_gt_root = os.path.join(opt.train_path, 'masks')
    train_loader = get_loader(train_image_root + '/', train_gt_root + '/', batchsize=opt.batchsize, trainsize=opt.trainsize)

    val_loader = None
    if opt.val_path:
        val_image_root = os.path.join(opt.val_path, 'images')
        val_gt_root = os.path.join(opt.val_path, 'masks')
        # 验证集一般 batch=1 更稳妥
        val_loader = get_loader(val_image_root + '/', val_gt_root + '/', batchsize=1, trainsize=opt.trainsize)

    # ---- resume ----
    start_epoch = 1
    best_loss = float('inf')
    bad_epochs = 0

    ckpt_dir = os.path.join('snapshots', opt.train_save, 'checkpoints')
    last_ckpt = os.path.join(ckpt_dir, 'last.pth')
    best_ckpt = os.path.join(ckpt_dir, 'best.pth')

    if opt.resume:
        print(f"[Resume] Loading checkpoint: {opt.resume}")
        last_epoch, best_loss, bad_epochs = load_checkpoint(opt.resume, model, optimizer)
        start_epoch = last_epoch + 1
        print(f"[Resume] start_epoch={start_epoch}, best_loss={best_loss:.6f}, bad_epochs={bad_epochs}")

    print('#' * 20, 'Start Training', '#' * 20)

    for epoch in range(start_epoch, opt.epochs + 1):
        # 仍保留你原本的 adjust_lr（如果你想换 scheduler，也行）
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

        train_loss = train_one_epoch(train_loader, model, optimizer, opt, epoch)

        # early stopping 监控：优先 val_loss
        if val_loader is not None:
            val_loss = evaluate(val_loader, model, opt.trainsize)
            monitor = val_loss
            print(f"[Eval] Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        else:
            monitor = train_loss
            print(f"[Eval] Epoch {epoch}: train_loss={train_loss:.6f} (no val)")

        # 保存 last
        save_checkpoint(last_ckpt, model, optimizer, epoch, best_loss, bad_epochs, opt)

        # 保存 best + 更新 early stop
        improved = monitor < (best_loss - opt.min_delta)
        if improved:
            best_loss = monitor
            bad_epochs = 0
            save_checkpoint(best_ckpt, model, optimizer, epoch, best_loss, bad_epochs, opt)
            print(f"[Checkpoint] New best: {best_loss:.6f} -> {best_ckpt}")
        else:
            bad_epochs += 1
            print(f"[EarlyStop] No improvement. bad_epochs={bad_epochs}/{opt.patience}")

        # 可选：每 N epoch 也保存一份 snapshot（兼容你原来的行为）
        if opt.save_every > 0 and (epoch % opt.save_every == 0):
            snap_path = os.path.join('snapshots', opt.train_save, f'PraNet-epoch{epoch}.pth')
            os.makedirs(os.path.dirname(snap_path), exist_ok=True)
            torch.save(model.state_dict(), snap_path)
            print('[Saving Snapshot:]', snap_path)

        if bad_epochs >= opt.patience:
            print(f"[EarlyStop] Stop at epoch {epoch}. Best loss = {best_loss:.6f}")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 训练周期
    parser.add_argument('--epochs', type=int, default=200, help='max epoch number (not limited to 20)')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='min improvement to reset patience')

    # 优化器/学习率
    parser.add_argument('--lr', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--gate_lr_mult', type=float, default=10.0, help='lr multiplier for gating params')
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # 数据
    parser.add_argument('--train_path', type=str, default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--val_path', type=str, default='', help='(optional) path to val dataset, same format as train')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training image size')

    # 模型
    parser.add_argument('--channel', type=int, default=32)
    parser.add_argument('--dynamic_gate', action='store_true', help='enable sample-adaptive RA gating')
    parser.add_argument('--gate_hidden', type=int, default=128, help='hidden dim for gating MLP')

    # 稳定训练
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--gate_reg', type=float, default=0.0, help='regularize gates toward 1 (e.g., 1e-3)')
    parser.add_argument('--ms_train', action='store_true', help='multi-scale training (0.75/1/1.25)')

    # 学习率衰减（保留你原来的策略）
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=50)

    # 保存/续训
    parser.add_argument('--train_save', type=str, default='PraNet_Res2Net_RAWeight')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint .pth to resume')
    parser.add_argument('--save_every', type=int, default=10, help='also save state_dict every N epochs (0 to disable)')

    # log
    parser.add_argument('--print_freq', type=int, default=20)

    opt = parser.parse_args()
    if opt.val_path == '':
        opt.val_path = ''

    main(opt)
