""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
from typing import Optional

import torch
import torch.utils.data as data
from PIL import Image

from timm.data.readers import create_reader
from timm.data.dataset import ImageDataset, IterableImageDataset, AugMixDataset

import numpy as np
import cv2

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50

# CHECKPOINT: get canny image
def get_canny(img:Image):
    img=np.array(img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(gray,threshold1=75,threshold2=150)
    return Image.fromarray(edge)


# CHECKPOINT: define EdgeImageDataset to support edge_aux mode
class EdgeImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            input_img_mode='RGB',
            transform=None,
            target_transform=None,
            additional_features=None,
            **kwargs,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map,
                additional_features=additional_features,
                **kwargs,
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.input_img_mode = input_img_mode
        self.transform = transform
        self.target_transform = target_transform
        self.additional_features = additional_features
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target, *features = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.input_img_mode and not self.load_bytes:
            img = img.convert(self.input_img_mode)  # → (3, H, W)

        # CHECKPOINT: keep rgb transform
        if self.transform is not None:
            img_rgb = self.transform(img)
        else:
            img_rgb = img
        
        # CHECKPOINT: add canny image
        img_edge = get_canny(img)   # → (1, H, W)
        if self.transform:
            from torchvision.transforms.functional import pil_to_tensor
            from torchvision.transforms import ToTensor
            img_edge = pil_to_tensor(img_edge)
            img_edge = img_edge / 255.0
        img = (img_rgb, img_edge)   # return tuple

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        if self.additional_features is None:
            return img, target
        else:
            return img, target, *features

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


""" Dataset Factory

Hacked together by / Copyright 2021, Ross Wightman
"""
import os
# from typing import Optional

from torchvision.datasets import CIFAR100, CIFAR10, MNIST, KMNIST, FashionMNIST, ImageFolder
try:
    from torchvision.datasets import Places365
    has_places365 = True
except ImportError:
    has_places365 = False
try:
    from torchvision.datasets import INaturalist
    has_inaturalist = True
except ImportError:
    has_inaturalist = False
try:
    from torchvision.datasets import QMNIST
    has_qmnist = True
except ImportError:
    has_qmnist = False
try:
    from torchvision.datasets import ImageNet
    has_imagenet = True
except ImportError:
    has_imagenet = False


_TORCH_BASIC_DS = dict(
    cifar10=CIFAR10,
    cifar100=CIFAR100,
    mnist=MNIST,
    kmnist=KMNIST,
    fashion_mnist=FashionMNIST,
)
_TRAIN_SYNONYM = dict(train=None, training=None)
_EVAL_SYNONYM = dict(val=None, valid=None, validation=None, eval=None, evaluation=None)


def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root

    def _try(syn):
        for s in syn:
            try_root = os.path.join(root, s)
            if os.path.exists(try_root):
                return try_root
        return root
    if split_name in _TRAIN_SYNONYM:
        root = _try(_TRAIN_SYNONYM)
    elif split_name in _EVAL_SYNONYM:
        root = _try(_EVAL_SYNONYM)
    return root


def create_dataset(
        name: str,
        root: Optional[str] = None,
        split: str = 'validation',
        search_split: bool = True,
        class_map: dict = None,
        load_bytes: bool = False,
        is_training: bool = False,
        download: bool = False,
        batch_size: int = 1,
        num_samples: Optional[int] = None,
        seed: int = 42,
        repeats: int = 0,
        input_img_mode: str = 'RGB',
        trust_remote_code: bool = False,
        edge_aux: bool = False,     # CHECKPOINT: edge_aux mode
        **kwargs,
):
    """ Dataset factory method

    In parentheses after each arg are the type of dataset supported for each arg, one of:
      * Folder - default, timm folder (or tar) based ImageDataset
      * Torch - torchvision based datasets
      * HFDS - Hugging Face Datasets
      * HFIDS - Hugging Face Datasets Iterable (streaming mode, with IterableDataset)
      * TFDS - Tensorflow-datasets wrapper in IterabeDataset interface via IterableImageDataset
      * WDS - Webdataset
      * All - any of the above

    Args:
        name: Dataset name, empty is okay for folder based datasets
        root: Root folder of dataset (All)
        split: Dataset split (All)
        search_split: Search for split specific child fold from root so one can specify
            `imagenet/` instead of `/imagenet/val`, etc on cmd line / config. (Folder, Torch)
        class_map: Specify class -> index mapping via text file or dict (Folder)
        load_bytes: Load data, return images as undecoded bytes (Folder)
        download: Download dataset if not present and supported (HFIDS, TFDS, Torch)
        is_training: Create dataset in train mode, this is different from the split.
            For Iterable / TDFS it enables shuffle, ignored for other datasets. (TFDS, WDS, HFIDS)
        batch_size: Batch size hint for iterable datasets (TFDS, WDS, HFIDS)
        seed: Seed for iterable datasets (TFDS, WDS, HFIDS)
        repeats: Dataset repeats per iteration i.e. epoch (TFDS, WDS, HFIDS)
        input_img_mode: Input image color conversion mode e.g. 'RGB', 'L' (folder, TFDS, WDS, HFDS, HFIDS)
        trust_remote_code: Trust remote code in Hugging Face Datasets if True (HFDS, HFIDS)
        **kwargs: Other args to pass through to underlying Dataset and/or Reader classes

    Returns:
        Dataset object
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    name = name.lower()
    if name.startswith('torch/'):
        name = name.split('/', 2)[-1]
        torch_kwargs = dict(root=root, download=download, **kwargs)
        if name in _TORCH_BASIC_DS:
            ds_class = _TORCH_BASIC_DS[name]
            use_train = split in _TRAIN_SYNONYM
            ds = ds_class(train=use_train, **torch_kwargs)
        elif name == 'inaturalist' or name == 'inat':
            assert has_inaturalist, 'Please update to PyTorch 1.10, torchvision 0.11+ for Inaturalist'
            target_type = 'full'
            split_split = split.split('/')
            if len(split_split) > 1:
                target_type = split_split[0].split('_')
                if len(target_type) == 1:
                    target_type = target_type[0]
                split = split_split[-1]
            if split in _TRAIN_SYNONYM:
                split = '2021_train'
            elif split in _EVAL_SYNONYM:
                split = '2021_valid'
            ds = INaturalist(version=split, target_type=target_type, **torch_kwargs)
        elif name == 'places365':
            assert has_places365, 'Please update to a newer PyTorch and torchvision for Places365 dataset.'
            if split in _TRAIN_SYNONYM:
                split = 'train-standard'
            elif split in _EVAL_SYNONYM:
                split = 'val'
            ds = Places365(split=split, **torch_kwargs)
        elif name == 'qmnist':
            assert has_qmnist, 'Please update to a newer PyTorch and torchvision for QMNIST dataset.'
            use_train = split in _TRAIN_SYNONYM
            ds = QMNIST(train=use_train, **torch_kwargs)
        elif name == 'imagenet':
            torch_kwargs.pop('download')
            assert has_imagenet, 'Please update to a newer PyTorch and torchvision for ImageNet dataset.'
            if split in _EVAL_SYNONYM:
                split = 'val'
            ds = ImageNet(split=split, **torch_kwargs)
        elif name == 'image_folder' or name == 'folder':
            # in case torchvision ImageFolder is preferred over timm ImageDataset for some reason
            if search_split and os.path.isdir(root):
                # look for split specific sub-folder in root
                root = _search_split(root, split)
            ds = ImageFolder(root, **kwargs)
        else:
            assert False, f"Unknown torchvision dataset {name}"
    elif name.startswith('hfds/'):
        # NOTE right now, HF datasets default arrow format is a random-access Dataset,
        # There will be a IterableDataset variant too, TBD
        ds = ImageDataset(
            root,
            reader=name,
            split=split,
            class_map=class_map,
            input_img_mode=input_img_mode,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    elif name.startswith('hfids/'):
        ds = IterableImageDataset(
            root,
            reader=name,
            split=split,
            class_map=class_map,
            is_training=is_training,
            download=download,
            batch_size=batch_size,
            num_samples=num_samples,
            repeats=repeats,
            seed=seed,
            input_img_mode=input_img_mode,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    elif name.startswith('tfds/'):
        ds = IterableImageDataset(
            root,
            reader=name,
            split=split,
            class_map=class_map,
            is_training=is_training,
            download=download,
            batch_size=batch_size,
            num_samples=num_samples,
            repeats=repeats,
            seed=seed,
            input_img_mode=input_img_mode,
            **kwargs
        )
    elif name.startswith('wds/'):
        ds = IterableImageDataset(
            root,
            reader=name,
            split=split,
            class_map=class_map,
            is_training=is_training,
            batch_size=batch_size,
            num_samples=num_samples,
            repeats=repeats,
            seed=seed,
            input_img_mode=input_img_mode,
            **kwargs
        )
    else:
        # folder based dataset
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        if search_split and os.path.isdir(root):
            # look for split specific sub-folder in root
            root = _search_split(root, split)

        # CHECKPOINT: support edge_aux mode
        if edge_aux:
            ds = EdgeImageDataset(
                root,
                reader=name,
                class_map=class_map,
                load_bytes=load_bytes,
                input_img_mode=input_img_mode,
                **kwargs,
            )
        else:
            ds = ImageDataset(
                root,
                reader=name,
                class_map=class_map,
                load_bytes=load_bytes,
                input_img_mode=input_img_mode,
                **kwargs,
            )
    return ds

