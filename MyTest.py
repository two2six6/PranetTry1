import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

# 使用带“可学习 RA 权重”的模型版本（与你训练保存的 ra_w*/ra_beta* checkpoint 对齐）
from lib.PraNet_Res2Net_RAWeight import PraNet
from utils.dataloader import test_dataset


ALIGN_CORNERS = True  # 与训练/模型文件保持一致，减少插值差异


def save_gray_png(path: str, arr01: np.ndarray) -> None:
    """保存单通道预测为 8-bit PNG。

    imageio/PIL 无法直接保存 float32 的 PNG（会报 cannot write mode F as PNG），
    所以这里统一转 uint8。
    """
    arr01 = np.clip(arr01, 0.0, 1.0)
    arr_u8 = (arr01 * 255.0).astype(np.uint8)
    imageio.imwrite(path, arr_u8)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """兼容 CPU/GPU、DataParallel('module.') 前缀的加载方式。"""
    state = torch.load(ckpt_path, map_location='cpu')
    # 兼容两类保存：
    # 1) torch.save(model.state_dict())
    # 2) torch.save({'model': model.state_dict(), ...})  (MyTrain.py 的 checkpoint)
    if isinstance(state, dict):
        if 'state_dict' in state:
            state = state['state_dict']
        elif 'model' in state:
            state = state['model']

    new_state = {}
    for k, v in state.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        new_state[k] = v

    # 这里 strict=True：因为你要做“可学习参数”的实验，模型结构必须匹配 checkpoint
    model.load_state_dict(new_state, strict=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument(
        '--pth_path',
        type=str,
        default='./snapshots/PraNet_Res2Net_RAWeight/checkpoints/best.pth',
        help='checkpoint path. If you train with MyTrain.py, best model is usually in snapshots/<train_save>/checkpoints/best.pth'
    )
    parser.add_argument('--dynamic_gate', action='store_true', help='enable sample-adaptive RA gating (must match training)')
    args = parser.parse_args()

    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        data_path = './data/TestDataset/{}/'.format(_data_name)
        save_path = './results/PraNet_RAWeight/{}/'.format(_data_name)
        os.makedirs(save_path, exist_ok=True)

        model = PraNet(use_dynamic_gate=args.dynamic_gate).cuda().eval()
        load_checkpoint(model, args.pth_path)

        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, args.testsize)

        with torch.no_grad():
            for i in range(test_loader.size):
                image, gt, name = test_loader.load_data()

                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)

                image = image.cuda()

                res5, res4, res3, res2 = model(image)
                res = res2

                # PyTorch 1.10: F.upsample 已弃用，使用 F.interpolate
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=ALIGN_CORNERS)

                res = torch.sigmoid(res).detach().cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                save_gray_png(os.path.join(save_path, name), res)
