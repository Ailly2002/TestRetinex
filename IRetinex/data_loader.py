import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random

random.seed(1143)


def populate_train_list(input_root, gt_root):
    """
    按「相对路径」匹配Input（低光）和GT（正常光照）图片对
    :param input_root: Input图片根目录（低光图）
    :param gt_root: GT图片根目录（正常光照图）
    :return: 成对的Input路径列表、GT路径列表
    """
    # 递归遍历Input根目录下所有jpg/png图片（含子目录）
    input_pattern = os.path.join(input_root, "**", "*.[jp][pn]g")  # 匹配jpg/png（兼容大小写可加: glob.IGNORECASE）
    input_list = glob.glob(input_pattern, recursive=True)

    train_input_list = []
    train_gt_list = []

    for input_abs_path in input_list:
        # 步骤1：计算Input图片相对于Input根目录的「相对路径」
        input_rel_path = os.path.relpath(input_abs_path, input_root)

        # 步骤2：拼接GT根目录 + Input相对路径，得到GT的绝对路径
        gt_abs_path = os.path.join(gt_root, input_rel_path)

        # 步骤3：检查GT文件是否存在（兼容扩展名大小写，如JPG/Png）
        if os.path.exists(gt_abs_path):
            train_input_list.append(input_abs_path)
            train_gt_list.append(gt_abs_path)
        else:
            # 可选：打印缺失的GT路径，便于排查数据集问题
            # print(f"Warning: GT文件缺失 - {gt_abs_path}（对应Input：{input_abs_path}）")
            continue

    # 打乱数据（保持Input和GT的对应关系）
    combined = list(zip(train_input_list, train_gt_list))
    random.shuffle(combined)
    train_input_list[:], train_gt_list[:] = zip(*combined) if combined else ([], [])

    return train_input_list, train_gt_list


class RetinexDataset(data.Dataset):
    def __init__(self, input_root, gt_root, size=512):
        """
        :param input_root: Input（低光图）根目录
        :param gt_root: GT（正常光照图）根目录
        :param size: 图片resize尺寸
        """
        self.input_list, self.gt_list = populate_train_list(input_root, gt_root)
        self.size = size
        print(f"Total valid training pairs: {len(self.input_list)}")
        if len(self.input_list) == 0:
            raise ValueError("No valid Input-GT pairs found! Check your dataset paths.")

    def __getitem__(self, index):
        # 加载低光图像（Input）
        input_path = self.input_list[index]
        input_img = Image.open(input_path).convert('RGB')
        input_img = input_img.resize((self.size, self.size), Image.ANTIALIAS)
        input_img = (np.asarray(input_img) / 255.0).astype(np.float32)
        input_img = torch.from_numpy(input_img).permute(2, 0, 1)  # (H,W,C) → (C,H,W)

        # 加载正常光照图像（GT）
        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = gt_img.resize((self.size, self.size), Image.ANTIALIAS)
        gt_img = (np.asarray(gt_img) / 255.0).astype(np.float32)
        gt_img = torch.from_numpy(gt_img).permute(2, 0, 1)

        return input_img, gt_img

    def __len__(self):
        return len(self.input_list)