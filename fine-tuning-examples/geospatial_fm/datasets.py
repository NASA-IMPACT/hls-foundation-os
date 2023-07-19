from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from .geospatial_pipelines import LoadGeospatialAnnotations

@DATASETS.register_module()
class Sen1Floods11(CustomDataset):
    """
    Sen1Floods11 dataset.
    """

    CLASSES = (0, 1)

    PALETTE = None

    def __init__(self, **kwargs):    

        gt_seg_map_loader_cfg = kwargs.pop('gt_seg_map_loader_cfg') if 'gt_seg_map_loader_cfg' in kwargs else dict()
        reduce_zero_label = kwargs.pop('reduce_zero_label') if 'reduce_zero_label' in kwargs else False
        
        super(Sen1Floods11, self).__init__(
            **kwargs)
        
        self.gt_seg_map_loader = LoadGeospatialAnnotations(reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg)
        

@DATASETS.register_module()
class FireScars(CustomDataset):
    """
    Sen1Floods11 dataset.
    """

    CLASSES = (0, 1)

    PALETTE = None

    def __init__(self, **kwargs):    

        gt_seg_map_loader_cfg = kwargs.pop('gt_seg_map_loader_cfg') if 'gt_seg_map_loader_cfg' in kwargs else dict()
        reduce_zero_label = kwargs.pop('reduce_zero_label') if 'reduce_zero_label' in kwargs else False
        
        super(FireScars, self).__init__(
            **kwargs)
        
        self.gt_seg_map_loader = LoadGeospatialAnnotations(reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg)

        
@DATASETS.register_module()
class LULC(CustomDataset):
    """
    Sen1Floods11 dataset.
    """

    CLASSES=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

    PALETTE = None

    def __init__(self, **kwargs):    

        gt_seg_map_loader_cfg = kwargs.pop('gt_seg_map_loader_cfg') if 'gt_seg_map_loader_cfg' in kwargs else dict()
        reduce_zero_label = kwargs.pop('reduce_zero_label') if 'reduce_zero_label' in kwargs else False
        
        super(LULC, self).__init__(
            **kwargs)
        
        self.gt_seg_map_loader = LoadGeospatialAnnotations(reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg)
