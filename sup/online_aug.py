import numpy as np
from scipy import ndimage
import random
from scipy.ndimage.interpolation import zoom
import torch

def Standardize(images):
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    Mean and std parameter have to be provided explicitly.
    new: z-score is used but keep the background with zero!
    """
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)
    mask_location = images.sum(0) > 0
    for k in range(images.shape[0]):
        image = images[k,...]
        image = np.array(image, dtype = 'float32')
        mask_area = image[mask_location]
        image[mask_location] -= mask_area.mean()
        image[mask_location] /= mask_area.std()
        images[k,...] = image
    return images


def flip(image_array, flip_type):
    image_flip = np.flip(image_array, flip_type)
    return image_flip


def rotate(image_array, rotate_num):
    image_rotate = ndimage.rotate(image_array, rotate_num, axes=(1,2), reshape=False)
    return image_rotate


def random_crop(image_array, image_type):
    if image_type == 'bbox':
        crop_size = random.randint(100,128)
        x, y, z = image_array.shape[-3], image_array.shape[-2], image_array.shape[-1]
        image_ndim = image_array.ndim
        if x == 24:  # 24,128,128
            height_point = random.randint(0, 128 - crop_size)
            weight_point = random.randint(0, 128 - crop_size)
            if image_ndim == 4:
                crop_image = image_array[:, :, height_point:height_point + crop_size, weight_point:weight_point + crop_size]
            if image_ndim == 3:
                crop_image = image_array[:, height_point:height_point + crop_size, weight_point:weight_point + crop_size]
        if x == 128:  # 128,128,24
            height_point = random.randint(0, 128 - crop_size)
            weight_point = random.randint(0, 128 - crop_size)
            if image_ndim == 4:
                crop_image = image_array[:, height_point:height_point + crop_size, weight_point:weight_point + crop_size, :]
            if image_ndim == 3:
                crop_image = image_array[height_point:height_point + crop_size, weight_point:weight_point + crop_size, :]
        resize_factor = np.array(image_array.shape) / np.array(crop_image.shape)
        crop_image_resize = zoom(crop_image, resize_factor, mode='nearest')
    else:
        print('wrong image_type!')
    return crop_image_resize


def cutout_operation(img, length):
    """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

    h = img.shape[1]
    w = img.shape[2]
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)
    mask[y1: y2, x1: x2] = 0.
    img = img.copy()
    img = torch.from_numpy(img)
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img * mask
    img = np.array(img)
    return img


def cutout(image_array):

    image_cutout = []

    x, y, z = image_array.shape[-3], image_array.shape[-2], image_array.shape[-1]
    image_ndim = image_array.ndim

    if x == 24:  # 24,128,128
        length = int(int(image_array.shape[2]) / 5)
        if image_ndim == 4:
            for i in range(image_array.shape[0]):
                img = image_array[i]
                img_cutout = cutout_operation(img, length)
                if i == 0:
                    image_cutout = np.expand_dims(img_cutout, axis=0)
                else:
                    image_cutout = np.concatenate((image_cutout, np.expand_dims(img_cutout, axis=0)), axis=0)
        if image_ndim == 3:
            image_cutout = cutout_operation(image_array, length)
    if x == 128:  # 128,128,24
        length = int(int(image_array.shape[1]) / 10)
        if image_ndim == 4:
            image_array = image_array.transpose((0, 3, 1, 2))
            for i in range(image_array.shape[0]):
                img = image_array[i]
                img_cutout = cutout_operation(img, length)
                if i == 0:
                    image_cutout = np.expand_dims(img_cutout, axis=0)
                else:
                    image_cutout = np.concatenate((image_cutout, np.expand_dims(img_cutout, axis=0)), axis=0)
            image_cutout = image_cutout.transpose((0, 2, 3, 1))
        if image_ndim == 3:
            image_array = image_array.transpose((2, 0, 1))
            image_cutout = cutout_operation(image_array, length)
            image_cutout = image_cutout.transpose((1, 2, 0))
    return image_cutout



def gaussian_noise(image_array, divisor=2):
    '''
    ---add gussion noise to the image---
    input: img, mean, std,
    return: gaussian_out, noise
    '''
    new_image_list = []
    for i in range(image_array.shape[0]):
        img = image_array[i,:,:]
        noise = np.random.normal(np.mean(img), np.std(img), img.shape)
        gaussian_out = img + noise/divisor
        new_image_list.append(gaussian_out)
    new_image_array = np.array(new_image_list)
    return new_image_array


def online_aug(image_array, image_type):
    out_image = image_array

    ifrandom_crop = random.random()
    ifflip = random.random()
    ifnosie = random.random()

    if ifrandom_crop > 0.5:
        out_image = random_crop(out_image, image_type)
    if ifflip > 0.5:
        flip_type = random.randint(1, 2)
        out_image = flip(out_image, flip_type)
    if ifnosie > 0.5:
        out_image = gaussian_noise(out_image)

    if out_image.shape[0] == 1:
        new_out_img = Standardize(out_image)
        new_out_img = new_out_img.copy()
    else:
        for i in range(out_image.shape[0]):
            new_img = out_image[i, ...]
            new_img = Standardize(new_img)
            if i == 0:
                new_out_img = new_img
            else:
                new_out_img = np.concatenate((new_out_img, new_img), axis=0)
    return new_out_img

