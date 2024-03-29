"""NYUv2 dataset based on the https://github.com/xapharius/pytorch-nyuv2/tree/master"""


import os
import shutil
import tarfile
import typing as t
import zipfile

import h5py
import numpy as np
import requests
import torch
from PIL import Image
from torchvision.datasets.utils import download_url

from vision_mtl.cfg import nyuv2_data_cfg as data_cfg
from vision_mtl.data_modules.common_ds import MTLDataset


class NYUv2(MTLDataset):
    """
    PyTorch wrapper for the NYUv2 dataset focused on multi-task learning.
    Data sources available: RGB, Semantic Segmentation, Surface Normals, Depth Images.
    If no transformation is provided, the image type will not be returned.

    ### Output
    All images are of size: 640 x 480

    1. RGB: 3 channel input image

    2. Semantic Segmentation: 1 channel representing one of the 14 (0 -
    background) classes. Conversion to int will happen automatically if
    transformation ends in a tensor.

    3. Surface Normals: 3 channels, with values in [0, 1].

    4. Depth Images: 1 channel with floats representing the distance in meters.
    Conversion will happen automatically if transformation ends in a tensor.
    """

    benchmark_idxs: list[int] = [647, 584, 169, 768]

    def __init__(
        self,
        stage: str = "train",
        data_base_dir: str = data_cfg.data_dir,
        download: bool = False,
        use_sn: bool = False,
        transforms: t.Any = data_cfg.train_transform,
        max_depth: float = data_cfg.max_depth,
    ):
        """
        Will return tuples based on what data source has been enabled (rgb, seg etc).

        :param root: path to root folder (eg /data/NYUv2)
        :param train: whether to load the train or test set
        :param download: whether to download and process data if missing
        :param rgb_transform: the transformation pipeline for rbg images
        :param seg_transform: the transformation pipeline for segmentation images. If
        the transformation ends in a tensor, the result will be automatically
        converted to int in [0, 14)
        :param sn_transform: the transformation pipeline for surface normal images
        :param depth_transform: the transformation pipeline for depth images. If the
        transformation ends in a tensor, the result will be automatically converted
        to meters
        """

        assert stage in ["train", "test"], "stage must be either train or test"

        super().__init__(
            stage=stage,
            data_base_dir=data_base_dir,
            max_depth=max_depth,
            train_transform=transforms,
            test_transform=transforms,
        )
        self.use_sn = use_sn

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not complete." + " You can use download=True to download it"
            )

        # rgb folder as ground truth
        self.filenames = sorted(
            os.listdir(os.path.join(data_base_dir, f"{self.stage}_rgb"))
        )

    def __getitem__(self, index: int):
        raw_sample = self.load_raw_sample(index)

        sample = self.prepare_sample(raw_sample, self.transform)

        return sample

    def prepare_sample(self, raw_sample, transform=None):
        img, mask, depth, normals = (
            raw_sample["img"],
            raw_sample["mask"],
            raw_sample["depth"],
            raw_sample.get("normals"),
        )

        apply_transform = transform is not None
        if apply_transform:
            img = transform(img)
            mask = transform(mask)
            depth = transform(depth)
            if normals is not None:
                normals = transform(normals)

        img = self.convert_to_tensor(img)
        mask = self.convert_to_tensor(mask)
        depth = self.convert_to_tensor(depth)

        if img.max() > 1.0:
            img /= 255

        if mask.max() <= 1.0:
            # ToTensor transform scales to [0, 1] by default
            mask = mask * 255
        mask = mask.squeeze().long()

        # depth png is uint16
        depth = depth.float() / 1e4
        depth = self.normalize_depth(depth)

        if depth.shape[0] == 1:
            depth = depth.permute(1, 2, 0)

        sample = {"img": img, "mask": mask, "depth": depth}

        if normals is not None:
            normals = self.convert_to_tensor(normals)
            sample["normals"] = normals

        return sample

    def convert_to_tensor(self, x, dtype=torch.float32):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(dtype)
        return x

    def load_raw_sample(self, index: int) -> dict:
        def get_folder(name):
            return os.path.join(self.data_base_dir, f"{self.stage}_{name}")

        img = np.array(Image.open(os.path.join(get_folder("rgb"), self.filenames[index])))
        mask = np.array(
            Image.open(os.path.join(get_folder("seg13"), self.filenames[index]))
        )
        depth = np.array(
            Image.open(os.path.join(get_folder("depth"), self.filenames[index]))
        )
        sample = {"img": img, "mask": mask, "depth": depth}
        if self.use_sn:
            normals = np.array(
                Image.open(os.path.join(get_folder("sn"), self.filenames[index]))
            )
            sample["normals"] = normals

        return sample

    def __len__(self):
        return len(self.filenames)

    def __repr__(self) -> str:
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of data points: {self.__len__()}\n"
        fmt_str += f"    Split: {self.stage}\n"
        fmt_str += f"    Root Location: {self.data_base_dir}\n"
        tmp = "    Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        try:
            for split in ["train", "test"]:
                inputs = ["rgb", "seg13", "depth"]
                if self.use_sn:
                    inputs.append("sn")
                for part in inputs:
                    path = os.path.join(self.data_base_dir, f"{split}_{part}")
                    if not os.path.exists(path):
                        raise FileNotFoundError("Missing Folder")
        except FileNotFoundError:
            return False
        return True

    def download(self):
        if self._check_exists():
            return
        download_rgb(self.data_base_dir)
        download_seg(self.data_base_dir)
        download_depth(self.data_base_dir)
        if self.use_sn:
            download_sn(self.data_base_dir)
        print("Done!")


def download_rgb(root: str):
    train_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz"
    test_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[2])

    _proc(train_url, os.path.join(root, "train_rgb"))
    _proc(test_url, os.path.join(root, "test_rgb"))


def download_seg(root: str):
    train_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz"
    test_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[3])

    _proc(train_url, os.path.join(root, "train_seg13"))
    _proc(test_url, os.path.join(root, "test_seg13"))


def download_sn(root: str):
    url = "https://www.dropbox.com/s/dn5sxhlgml78l03/nyu_normals_gt.zip"
    train_dst = os.path.join(root, "train_sn")
    test_dst = os.path.join(root, "test_sn")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            req = requests.get(url + "?dl=1")  # dropbox
            with open(tar, "wb") as f:
                f.write(req.content)
        if os.path.exists(tar):
            _unpack(tar)
            if not os.path.exists(train_dst):
                _replace_folder(
                    os.path.join(root, "nyu_normals_gt", "train"), train_dst
                )
                _rename_files(train_dst, lambda x: x[1:])
            if not os.path.exists(test_dst):
                _replace_folder(os.path.join(root, "nyu_normals_gt", "test"), test_dst)
                _rename_files(test_dst, lambda x: x[1:])
            shutil.rmtree(os.path.join(root, "nyu_normals_gt"))


def download_depth(
    root: str,
    url: str = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
):
    """Downloads the official labelled depth dataset and extracts the images. The URL may not work and one may need to find another source for depths, like https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2."""

    train_dst = os.path.join(root, "train_depth")
    test_dst = os.path.join(root, "test_depth")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            download_url(url, root)
        if os.path.exists(tar):
            train_ids = [
                f.split(".")[0] for f in os.listdir(os.path.join(root, "train_rgb"))
            ]
            _create_depth_files(tar, root, train_ids)


def _unpack(file: str):
    """
    Unpacks tar and zip, does nothing for any other type
    :param file: path of file
    """
    path = file.rsplit(".", 1)[0]

    if file.endswith(".tgz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(path)
        tar.close()
    elif file.endswith(".zip"):
        zip = zipfile.ZipFile(file, "r")
        zip.extractall(path)
        zip.close()


def _rename_files(folder: str, rename_func: t.Callable):
    """
    Renames all files inside a folder based on the passed rename function
    :param folder: path to folder that contains files
    :param rename_func: function renaming filename (not including path) str -> str
    """
    imgs_old = os.listdir(folder)
    imgs_new = [rename_func(file) for file in imgs_old]
    for img_old, img_new in zip(imgs_old, imgs_new):
        shutil.move(os.path.join(folder, img_old), os.path.join(folder, img_new))


def _replace_folder(src: str, dst: str):
    """
    Rename src into dst, replacing/overwriting dst if it exists.
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)


def _create_depth_files(mat_file: str, root: str, train_ids: list):
    """
    Extract the depth arrays from the mat file into images
    :param mat_file: path to the official labelled dataset .mat file
    :param root: The root directory of the dataset
    :param train_ids: the IDs of the training images as string (for splitting)
    """
    os.mkdir(os.path.join(root, "train_depth"))
    os.mkdir(os.path.join(root, "test_depth"))
    train_ids = set(train_ids)

    depths = h5py.File(mat_file, "r")["depths"]
    for i in range(len(depths)):
        img = (depths[i] * 1e4).astype(np.uint16).T
        id_ = str(i + 1).zfill(4)
        folder = "train" if id_ in train_ids else "test"
        save_path = os.path.join(root, f"{folder}_depth", id_ + ".png")
        Image.fromarray(img).save(save_path)


if __name__ == "__main__":
    # executed for the first time, downloads the data requested by use_* flags
    data_dir_root = "./data"
    train_ds = NYUv2(
        data_base_dir=data_dir_root,
        download=True,
        stage="train",
        use_sn=False,
    )
    print(train_ds)
