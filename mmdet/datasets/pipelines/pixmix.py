###############
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance

from ..builder import PIPELINES
from torchvision import datasets
from torchvision import transforms


#########################################################
#################### AUGMENTATIONS ######################
#########################################################

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _, __):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _, __):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level, _):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, _):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level, _):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level, img_size):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level, img_size):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level, img_size):
  level = int_parameter(sample_level(level), img_size[0] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level, img_size):
  level = int_parameter(sample_level(level), img_size[1] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

#########################################################
######################## MIXINGS ########################
#########################################################

def get_ab(beta):
    if np.random.random() < 0.5:
        a = np.float32(np.random.beta(beta, 1))
        b = np.float32(np.random.beta(1, beta))
    else:
        a = 1 + np.float32(np.random.beta(1, beta))
        b = -np.float32(np.random.beta(1, beta))
    return a, b

def add(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    return (out + 1) / 2
    # try:
    #     a,b = get_ab(beta)
    #     img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    #     out = a * img1 + b * img2
    #     return (out + 1) / 2
    # except:
    #     print('=== add.except ===')
    #     print(f'beta={beta}')
    #     print(f'img1.shape={img1.shape}, img2.shape={img2.shape}')

def multiply(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2, img2 * 2
    # out = (img1 ** a) * (img2.clip(1e-37) ** b)
    out = (img1 ** a) * (img2.clamp(1e-37) ** b)
    return out / 2
    # print('mixing_op = multiply')
    # try:
    #     a,b = get_ab(beta)
    #     img1, img2 = img1 * 2, img2 * 2
    #     # out = (img1 ** a) * (img2.clip(1e-37) ** b)
    #     out = (img1 ** a) * (img2.clamp(1e-37) ** b)
    #     return out / 2
    # except:
    #     print('=== multiply.except ===')
    #     a, b = get_ab(beta)
    #     print(f'beta={beta}, a={a}, b={b}')
    #     print(f'img1.shape={img1.shape}, img2.shape={img2.shape}')


########################################
##### EXTRA MIXIMGS (EXPREIMENTAL) #####
########################################

def invert(img):
  return 1 - img

def screen(img1, img2, beta):
  img1, img2 = invert(img1), invert(img2)
  out = multiply(img1, img2, beta)
  return invert(out)

def overlay(img1, img2, beta):
  case1 = multiply(img1, img2, beta)
  case2 = screen(img1, img2, beta)
  if np.random.random() < 0.5:
    cond = img1 < 0.5
  else:
    cond = img1 > 0.5
  return torch.where(cond, case1, case2)

def darken_or_lighten(img1, img2, beta):
  if np.random.random() < 0.5:
    cond = img1 < img2
  else:
    cond = img1 > img2
  return torch.where(cond, img1, img2)

def swap_channel(img1, img2, beta):
  channel = np.random.randint(3)
  img1[channel] = img2[channel]
  return img1

class RandomImages300K(torch.utils.data.Dataset):
    def __init__(self, file, transform):
        self.dataset = np.load(file)
        self.transform = transform

    def __getitem__(self, index):
        img = self.dataset[index]
        return self.transform(img), 0

    def __len__(self):
        return len(self.dataset)

@PIPELINES.register_module()
class PixMix:
    def __init__(self, mixing_set, beta=3, k=4, aug_list='all_ops', aug_severity=3, use_300k=False):
        self.beta = beta
        self.k = k
        self.aug_list = aug_list
        self.aug_severity = aug_severity
        self.use_300k = use_300k

        to_tensor = transforms.ToTensor()
        self.mixing_set = mixing_set
        self.preprocess = {'tensorize': to_tensor}

    def __call__(self, results):
        img = results['img'].copy()

        img = Image.fromarray(img)
        mixing_set_transform = transforms.Compose(
            [transforms.Resize(max(img.height, img.width)),
             transforms.RandomCrop((img.height, img.width))])
        if self.use_300k:
            mixing_set = RandomImages300K(file='300K_random_images.npy', transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(img.height, img.width),# , padding=4),
                 transforms.RandomHorizontalFlip()]))
        else:
            mixing_set = datasets.ImageFolder(self.mixing_set, transform=mixing_set_transform)

        rnd_idx = np.random.choice(len(mixing_set))
        mixing_pic, _ = mixing_set[rnd_idx]

        pixmix1 = self.pixmix(img, mixing_pic, self.preprocess)
        results['img2'] = pixmix1
        pixmix2 = self.pixmix(img, mixing_pic, self.preprocess)
        results['img3'] = pixmix2

        ''' Save the result '''
        # img_orig = Image.fromarray(results['img'])
        # img_orig.save('/ws/external/data/0_orig.png')
        # mixing_pic.save('/ws/external/data/1_aug.png')
        # img_pixmix1 = Image.fromarray(results['img2'])
        # img_pixmix1.save('/ws/external/data/2_pixmix1.png')
        # img_pixmix2 = Image.fromarray(results['img3'])
        # img_pixmix2.save('/ws/external/data/3_pixmix2.png')


        if 'img_fields' in results:
            results['img_fields'].append('img2')
            results['img_fields'].append('img3')
        else:
            results['img_fields'] = ['img', 'img2', 'img3']
        return results

    def augment_input(self, image):
        img_size = image.size
        # print(f'img_size={img_size}')

        augmentations = [
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y
        ]
        augmentations_all = [
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y, color, contrast, brightness, sharpness
        ]
        aug_list = augmentations_all if self.aug_list == 'all_ops' else augmentations
        op = np.random.choice(aug_list)
        self.aug_severity = 1
        return op(image.copy(), self.aug_severity, img_size)

    def pixmix(self, orig, mixing_pic, preprocess):

        mixings = [add, multiply]

        tensorize = preprocess['tensorize']
        if np.random.random() < 0.5:
            mixed = tensorize(self.augment_input(orig))
        else:
            mixed = tensorize(orig)

        aug_image_copy = None
        for _ in range(np.random.randint(self.k + 1)):

            if np.random.random() < 0.5:
                aug_image_copy = tensorize(self.augment_input(orig))
            else:
                aug_image_copy = tensorize(mixing_pic)
            try:
                assert mixed.shape == aug_image_copy.shape
            except:
                print(f'mixed.shape={mixed.shape}, aug_image_copy.shape={aug_image_copy.shape}')

            mixed_op = np.random.choice(mixings)
            try:
                mixed = mixed_op(mixed, aug_image_copy, self.beta)
            except:
                print('=== === mixed_op.except')
                print(f'mixed_op={mixed_op}')
                print(f'orig.shape=({orig.height},{orig.width}), mixed.shape={mixed.shape}, aug_image_copy.shape={aug_image_copy.shape}')
                print('')
            mixed = torch.clamp(mixed, 0, 1)

        result = torch.tensor(mixed.clone().detach() * 255, dtype=torch.uint8).permute(1, 2, 0).numpy()
        return result
