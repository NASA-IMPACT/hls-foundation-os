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

        self.gt_seg_map_loader = LoadGeospatialAnnotations(reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg)