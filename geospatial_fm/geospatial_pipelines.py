"""
This file holds pipeline components useful for loading remote sensing images and annotations.
"""
import numpy as np
import os.path as osp
import torch
import torchvision.transforms.functional as F

from tifffile import imread
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import PIPELINES
from torchvision import transforms


def open_tiff(fname):
    data = imread(fname)
    return data


@PIPELINES.register_module()
class ConstantMultiply(object):
    """Multiply image by constant.

    It multiplies an image by a constant

    Args:
        constant (float, optional): The constant to multiply by. 1.0 (e.g. no alteration if not specified)
    """

    def __init__(self, constant=1.0):
        self.constant = constant

    def __call__(self, results):
        """Call function to multiply by constant input img

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Results with image multiplied by constant
        """

        results["img"] = results["img"] * self.constant

        return results


@PIPELINES.register_module()
class BandsExtract(object):

    """Extract bands from image. Assumes channels last

    It extracts bands from an image. Assumes channels last.

    Args:
        bands (list, optional): The list of indexes to use for extraction. If not provided nothing will happen.
    """

    def __init__(self, bands=None):
        self.bands = bands

    def __call__(self, results):
        """Call function to multiply extract bands

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Results with extracted bands
        """

        if self.bands is not None:
            results["img"] = results["img"][..., self.bands]

        return results


@PIPELINES.register_module()
class TorchRandomCrop(object):

    """

    It randomly crops a multichannel tensor.

    Args:
        crop_size (tuple): the size to use to crop
    """

    def __init__(self, crop_size=(224, 224)):
        self.crop_size = crop_size

    def __call__(self, results):
        i, j, h, w = transforms.RandomCrop.get_params(results["img"], self.crop_size)
        results["img"] = F.crop(results["img"], i, j, h, w).float()
        results["gt_semantic_seg"] = F.crop(results["gt_semantic_seg"], i, j, h, w)

        return results


@PIPELINES.register_module()
class TorchNormalize(object):
    """Normalize the image.

    It normalises a multichannel image using torch

    Args:
        mean (sequence): Mean values .
        std (sequence): Std values of 3 channels.
    """

    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = F.normalize(results["img"], self.means, self.stds, False)
        results["img_norm_cfg"] = dict(mean=self.means, std=self.stds)
        return results


@PIPELINES.register_module()
class Reshape(object):
    """
    It reshapes a tensor.
    Args:
        new_shape (tuple): tuple with new shape
        keys (list): list with keys to apply reshape to
        look_up (dict): dictionary to use to look up dimensions when more than one is to be inferred from the original image, which have to be inputed as -1s in the new_shape argument. eg {'2': 1, '3': 2} would infer the new 3rd and 4th dimensions from the 2nd and 3rd from the original image.
    """

    def __init__(self, new_shape, keys, look_up=None):
        self.new_shape = new_shape
        self.keys = keys
        self.look_up = look_up

    def __call__(self, results):
        dim_to_infer = np.where(np.array(self.new_shape) == -1)[0]

        for key in self.keys:
            if (len(dim_to_infer) > 1) & (self.look_up is not None):
                old_shape = results[key].shape
                tmp = np.array(self.new_shape)
                for i in range(len(dim_to_infer)):
                    tmp[dim_to_infer[i]] = old_shape[self.look_up[str(dim_to_infer[i])]]
                self.new_shape = tuple(tmp)
            results[key] = results[key].reshape(self.new_shape)

        return results


@PIPELINES.register_module()
class CastTensor(object):
    """

    It casts a tensor.

    Args:
        new_type (str): torch type
        keys (list): list with keys to apply reshape to
    """

    def __init__(self, new_type, keys):
        self.new_type = new_type
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].type(self.new_type)

        return results


@PIPELINES.register_module()
class CollectTestList(object):
    """

    It processes the data in a way that conforms with inference and test pipelines.

    Args:

        keys (list): keys to collect (eg img/gt_semantic_seg)
        meta_keys (list): additional meta to collect and add to img_metas

    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        img_meta = [img_meta]
        data["img_metas"] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = [results[key]]
        return data

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"
        )


@PIPELINES.register_module()
class TorchPermute(object):
    """Permute dimensions.

    Particularly useful in going from channels_last to channels_first

    Args:
        keys (Sequence[str]): Keys of results to be permuted.
        order (Sequence[int]): New order of dimensions.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].permute(self.order)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys}, order={self.order})"


@PIPELINES.register_module()
class LoadGeospatialImageFromFile(object):
    """

    It loads a tiff image. Returns in channels last format, transposing if necessary according to channels_last argument.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        nodata (float/int): no data value to substitute to nodata_replace
        nodata_replace (float/int): value to use to replace no data
        channels_last (bool): whether the file has channels last format.
            If False, will transpose to channels last format. Defaults to True.
    """

    def __init__(
        self, to_float32=False, nodata=None, nodata_replace=0.0, channels_last=True
    ):
        self.to_float32 = to_float32
        self.nodata = nodata
        self.nodata_replace = nodata_replace
        self.channels_last = channels_last

    def __call__(self, results):
        if results.get("img_prefix") is not None:
            filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]
        img = open_tiff(filename)

        if not self.channels_last:
            img = np.transpose(img, (1, 2, 0))

        if self.to_float32:
            img = img.astype(np.float32)

        if self.nodata is not None:
            img = np.where(img == self.nodata, self.nodata_replace, img)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        results["flip"] = False
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}"
        return repr_str


@PIPELINES.register_module()
class LoadSpatioTemporalImagesFromFile(LoadGeospatialImageFromFile):
    """
    Load a time-series dataset from multiple files.

    Currently hardcoded to assume that GeoTIFF files are structured in four
    different 'monthX' folders like so:

    - month1/
      - scene_m01_XXXXXX_chip01.tif
      - scene_m01_XXXXXX_chip02.tif
    - month2/
      - scene_m02_XXXXXX_chip01.tif
      - scene_m02_XXXXXX_chip02.tif
    - month3/
      - scene_m03_XXXXXX_chip01.tif
      - scene_m03_XXXXXX_chip02.tif
    - month4/
      - scene_m04_XXXXXX_chip01.tif
      - scene_m04_XXXXXX_chip02.tif
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, results):
        """
        Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if results.get("img_prefix") is not None:
            img_prefix = results["img_prefix"]
            assert img_prefix.endswith("month1")
            filenames = [
                osp.join(img_prefix, results["img_info"]["filename_t1"]),  # June
                osp.join(
                    img_prefix.replace("month1", "month2"),  # July
                    results["img_info"]["filename_t2"],
                ),
                osp.join(
                    img_prefix.replace("month1", "month3"),  # August
                    results["img_info"]["filename_t3"],
                ),
                # osp.join(
                #     img_prefix.replace("month1", "month4"), # September
                #     results["img_info"]["filename_t4"],
                # ),
            ]
        else:
            raise NotImplementedError

        img = np.stack(arrays=list(map(open_tiff, filenames)), axis=0)
        assert img.shape == (3, 512, 512, 6)  # Time, Height, Width, Channels
        if not self.channels_last:
            img = np.transpose(a=img, axes=(0, 2, 3, 1))
            assert img.shape == (3, 6, 512, 512)  # Time, Channels, Height, Width
        if self.to_float32:
            img = img.astype(dtype=np.float32)
        if self.nodata is not None:
            img = np.where(img == self.nodata, self.nodata_replace, img)

        results["filename"] = filenames[0]
        results["ori_filename"] = results["img_info"]["filename_t1"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        results["flip"] = False
        num_channels = 1 if len(img.shape) < 3 else img.shape[0]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results


@PIPELINES.register_module()
class LoadGeospatialAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        nodata (float/int): no data value to substitute to nodata_replace
        nodata_replace (float/int): value to use to replace no data


    """

    def __init__(
        self,
        reduce_zero_label=False,
        nodata=None,
        nodata_replace=-1,
    ):
        self.reduce_zero_label = reduce_zero_label
        self.nodata = nodata
        self.nodata_replace = nodata_replace

    def __call__(self, results):
        if results.get("seg_prefix", None) is not None:
            filename = osp.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        else:
            filename = results["ann_info"]["seg_map"]

        gt_semantic_seg = open_tiff(filename)

        if self.nodata is not None:
            gt_semantic_seg = np.where(
                gt_semantic_seg == self.nodata, self.nodata_replace, gt_semantic_seg
            )
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        if results.get("label_map", None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results["label_map"].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id

        results["gt_semantic_seg"] = gt_semantic_seg
        results["seg_fields"].append("gt_semantic_seg")
        return results
