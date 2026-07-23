import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
from PIL import ImageOps, ImageFilter, ImageEnhance

from src.datasets.manifest import load_records, sample_path


# region Augmentation
def img_aug_identity(img, scale=None):
    return img


def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)


def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def img_aug_contrast(img, scale=[0.5, 1.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Contrast(img).enhance(v)


def img_aug_brightness(img, scale=[0.5, 1.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Brightness(img).enhance(v)


def img_aug_sharpness(img, scale=[0.5, 2.0]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.posterize(img, v)


def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.solarize(img, v)


class RandomResizedCrop(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, image, mask=None):
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.8, 1.2), ratio=(1.0, 1.0))
        image = F.resized_crop(image, i, j, h, w, size=(self.img_size, self.img_size))
        if mask is None:
            return image
        mask = F.resized_crop(
            mask, i, j, h, w, size=(self.img_size, self.img_size), interpolation=T.InterpolationMode.NEAREST
        )
        return image, mask


class RandomAffine(object):
    def __init__(self, degrees=8.0, translate=0.08):
        self.degrees = degrees
        self.translate = translate

    def __call__(self, image, mask=None):
        angle = float(np.random.uniform(-self.degrees, self.degrees))
        max_dx = self.translate * image.size[0]
        max_dy = self.translate * image.size[1]
        translate = (
            int(np.random.uniform(-max_dx, max_dx)),
            int(np.random.uniform(-max_dy, max_dy)),
        )
        image = F.affine(
            image,
            angle=angle,
            translate=translate,
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=T.InterpolationMode.BILINEAR,
        )
        if mask is None:
            return image
        mask = F.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=T.InterpolationMode.NEAREST,
        )
        return image, mask


class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, image, mask=None):
        image = F.hflip(image)
        if mask is None:
            return image
        return image, F.hflip(mask)


class RandomVerticalFlip(object):
    def __init__(self):
        pass

    def __call__(self, image, mask=None):
        image = F.vflip(image)
        if mask is None:
            return image
        return image, F.vflip(mask)


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, image, mask=None):
        if mask is None:
            return image
        return image, mask


def get_strong_aug_list():
    op_list = [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_blur, [0.75, 1.25]),
        (img_aug_contrast, [0.75, 1.25]),
        (img_aug_brightness, [0.75, 1.25]),
        (img_aug_sharpness, [0.75, 1.25]),
    ]
    return op_list


class StrongAugmentation:
    def __init__(self):
        self.augment_list = get_strong_aug_list()
        self.total_ops = len(self.augment_list)

    def __call__(self, img):
        max_num = np.random.randint(0, high=self.total_ops + 1)
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            img = op(img, scales)
        return img


class WeakAugmentation(object):
    def __init__(self, args):
        self.augment_list = [
            RandomAffine(),
            RandomResizedCrop(args.img_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            Identity(),
        ]
        self.total_ops = len(self.augment_list)

    def __call__(self, image, mask):
        max_num = np.random.randint(0, high=self.total_ops + 1)
        ops = random.choices(self.augment_list, k=max_num)
        for op in ops:
            image, mask = op(image, mask)
        return image, mask


# endregion Augmentation


class USSegmentationDataset(Dataset):
    def __init__(self, args, records, dataloader_type):
        self.args = args
        self.records = records
        self.strong_aug = StrongAugmentation()
        self.weak_aug = WeakAugmentation(args)
        self.dataloader_type = dataloader_type
        self.transform = T.Compose([T.ToTensor(), T.ConvertImageDtype(torch.float32)])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = sample_path(self.args.data_root, record, "image_path")
        mask_path = sample_path(self.args.data_root, record, "mask_path")

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Apply augmentation
        if self.dataloader_type == "train":
            if self.args.strong_augs and self.args.weak_augs:
                if random.random() < 0.5:
                    image = self.strong_aug(image)
                    image, mask = self.weak_aug(image, mask)
            elif self.args.strong_augs:
                image = self.strong_aug(image)
            elif self.args.weak_augs:
                image, mask = self.weak_aug(image, mask)

        image = self.transform(image)
        mask = (self.transform(mask) > 0.5).int()

        if self.args.in_channels == 3:
            image = image.repeat(3, 1, 1)

        # INR: model outputs (B, N, C); batching stacks (N, C) -> (B, N, C). Avoid (B, C, N) from reshape(1,-1).
        c, _, _ = image.shape
        image = image.reshape(c, -1).transpose(0, 1).contiguous()
        mask = mask.reshape(-1)

        return image, mask, Path(record["image_path"]).name


class USSegmentationDataModule:
    def __init__(self, args):
        self.args = args

        manifest_kwargs = {
            "view_path": getattr(args, "data_view", None),
            "split_root": getattr(args, "split_root", None),
            "derivative_manifest": getattr(args, "derivative_manifest", None),
        }
        train_records = load_records(
            args.data_root, args.dataset, "segmentation", args.seed, ("train",), **manifest_kwargs
        )
        val_records = load_records(
            args.data_root, args.dataset, "segmentation", args.seed, ("val",), **manifest_kwargs
        )
        test_splits = ("train", "val", "test") if getattr(args, "test_all_splits", False) else ("test",)
        test_records = load_records(
            args.data_root, args.dataset, "segmentation", args.seed, test_splits, **manifest_kwargs
        )

        self.train_dataset = USSegmentationDataset(args, train_records, "train")
        self.val_dataset = USSegmentationDataset(args, val_records, "val")
        self.test_dataset = USSegmentationDataset(args, test_records, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.num_workers > 0,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.num_workers > 0,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.num_workers > 0,
            shuffle=False,
            drop_last=False,
        )
