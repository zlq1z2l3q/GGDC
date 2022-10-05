### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os

import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]


# 根据文件后缀名判断文件是否为图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def list_folder_images(dir, opt):
    images = []
    parsings = []
    landmarks = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in os.listdir(dir):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            # make sure there's a matching parsings for the image
            # parsing files must be png
            parsing_fname = fname[:-3] + 'png'
            landmarks_fname = fname[:-3] + 'png'

            images.append(path)
            # 获取图片和对应 mask 路径
            if os.path.isfile(os.path.join(dir, 'parsings', parsing_fname)):
                parsing_path = os.path.join(dir, 'parsings', parsing_fname)
                # images.append(path)
                parsings.append(parsing_path)
            if os.path.isfile(os.path.join(dir, 'landmarks', landmarks_fname)):
                landmark_path = os.path.join(dir, 'landmarks', landmarks_fname)
                landmarks.append(landmark_path)

    # sort according to identity in case of FGNET test
    if 'fgnet' in opt.dataroot.lower():
        images.sort(key=str.lower)
        parsings.sort(key=str.lower)

    return images, parsings, landmarks


#
def get_transform(opt, normalize=True):
    transform_list = []

    # 既 resize 又随机裁剪
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        # 此例中 loadSize 和 fineSize 恰好都为 256
        transform_list.append(transforms.Resize(osize, interpolation=Image.NEAREST))  # 按 loadSize resize 成统一尺寸（256*256）
        # （接上）插值为最近邻插值法，插值的目的是为了图像缩放后重新计算像素
        transform_list.append(transforms.RandomCrop(opt.fineSize))  # 按 fineSize 从 resize 后的图片中随机裁剪出新图片（256*256）
    # 只随机裁剪
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))  # 按 fineSize 从原图片中随机裁剪出新图片（256*256）

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())  # 依概率（默认0.5）对图片水平翻转

    transform_list += [transforms.ToTensor()]

    # 标准化处理-->转换为正太分布，使模型更容易收敛
    if normalize:
        mean = (0.5,)
        std = (0.5,)
        transform_list += [transforms.Normalize(mean, std)]

    return transforms.Compose(transform_list)
