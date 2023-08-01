from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from .geospatial_pipelines import LoadGeospatialAnnotations

        
@DATASETS.register_module()
class GeospatialDataset(CustomDataset):
    """GeospatialDataset dataset.
    """

    def __init__(self, CLASSES=(0, 1), PALETTE=None, **kwargs):
        
        self.CLASSES = CLASSES

        self.PALETTE = PALETTE
        
        gt_seg_map_loader_cfg = kwargs.pop('gt_seg_map_loader_cfg') if 'gt_seg_map_loader_cfg' in kwargs else dict()
        reduce_zero_label = kwargs.pop('reduce_zero_label') if 'reduce_zero_label' in kwargs else False
        
        super(GeospatialDataset, self).__init__(
            reduce_zero_label=reduce_zero_label,
            # ignore_index=2,
            **kwargs)

        self.gt_seg_map_loader = LoadGeospatialAnnotations(
            reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg
        )


@DATASETS.register_module()
class SpatioTemporalDataset(GeospatialDataset):
    """
    Time-series dataset for irrigation data at
    https://huggingface.co/datasets/ibm-nasa-geospatial/hls_irrigation_scenes
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            raise NotImplementedError
        else:
            for img in self.file_client.list_dir_or_file(
                dir_path=img_dir, list_dir=False, suffix=img_suffix, recursive=True
            ):
                # Get 'T10SFG_chip22.tif' basename from 'scene_m01_T10SFG_chip22.tif'
                basename = "_".join(img.split(sep="_")[2:])
                img_info = dict(
                    filename_t1=f"scene_m01_{basename}",
                    filename_t2=f"scene_m02_{basename}",
                    filename_t3=f"scene_m03_{basename}",
                    filename_t4=f"scene_m04_{basename}",
                )
                if ann_dir is not None:
                    seg_map = f"mask_{basename.replace(img_suffix, seg_map_suffix)}"
                    img_info["ann"] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x["filename_t1"])

        return img_infos
