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

    解释：imageio/PIL 不能直接把 float32(F mode) 写成 PNG，会报
    "cannot write mode F as PNG"；因此统一转 uint8。
    """
    arr01 = np.clip(arr01, 0.0, 1.0)
    arr_u8 = (arr01 * 255.0).astype(np.uint8)
    imageio.imwrite(path, arr_u8)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """兼容 CPU/GPU、DataParallel('module.') 前缀的加载方式。"""
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

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
    parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net_RAWeight/PraNet-19.pth')

    # 第六点(3)：为了做跨数据集泛化对比，建议把输出目录区分开，避免覆盖
    parser.add_argument('--exp_name', type=str, default='PraNet_RAWeight',
                        help='subfolder name under ./results/ to avoid overwriting')

    args = parser.parse_args()

    datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

    for _data_name in datasets:
        data_path = './data/TestDataset/{}/'.format(_data_name)
        save_path = './results/{}/{}/'.format(args.exp_name, _data_name)
        os.makedirs(save_path, exist_ok=True)

        model = PraNet().cuda().eval()
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
