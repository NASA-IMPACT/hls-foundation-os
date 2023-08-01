# Image segmentation by foundation model finetuning

This repository shows three examples of how [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) can be finetuned for downstream tasks. These are flood detection using the Sentinel 2 data from [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset, burn scars detection using the [NASA HLS fire scars dataset](https://huggingface.co/datasets/nasa-impact/hls_burn_scars) and multi-temporal crop classification using the [NASA HLS multi-temporal crop classification dataset](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification).

## The approach
### Background
To finetune for these tasks in this repository, we make use of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/), which provides an extensible framework for segmentation tasks.

[MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) allows us to concatenate necks and heads appropriate for any segmentation downstream task to the encoder, and then perform the finetuning. This only requires the setup of a config file detailing the desired model architecture, dataset setup and training strategy.

We build extensions on top of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) to support our encoder and provide classes to read and augment remote sensing data (from .tiff files) using [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) data pipelines. These extensions can be found in the [geospatial_fm](./geospatial_fm/) directory, and they are installed as a package on the top of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) for easy to use. If more advanced functionality is necessary, it should be added there.

### The pretrained backbone
The pretrained model we work with is a [ViT](https://arxiv.org/abs/2010.11929) trained as a [Masked Auto Encoder](https://arxiv.org/abs/2111.06377). This is trained on [HLS](https://hls.gsfc.nasa.gov/) data. The encoder from this model is made available as the backbone and the weights can be downloaded from Hugging Face [here](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/blob/main/Prithvi_100M.pt).


### The architectures
We use a simple architecture that adds a neck and segmentation head to the backbone. The neck concatenates and processes the transformer's token based embeddings into one that can be fed into convolutional layers. The head processes this embedding into a segmentation mask. The code for these can be found in [this file](./geospatial_fm/geospatial_fm.py).

### The pipeline
We additionally provide extra components for data loading pipelines in [geospatial_pipelines.py](./geospatial_fm/geospatial_pipelines.py). These are documented in the file.
We observe the MMCV convention that all operations assume channels last format. Our tiff loader also assumes this is the format files are written in, and offers a flag to automatically transpose to channels last format if this is not the case.
*However*, we also introduce some components with the prefix `Torch`, e.g. `TorchNormalize`. These components assume the torch convention of channels first.

At some point during the pipeline, before feeding the data to the model, it is necessary to change to channels first format.
We reccomend doing after the `ToTensor` operation (which is also necessary at some point), using the `TorchPermute` operation.
## Setup
### Dependencies
1. Clone this repository
2. `conda create -n <environment-name> python==3.9`
3. `conda activate <environment-name>`
4. Install torch (tested for >=1.7.1 and <=11) and torchvision (tested for >=0.8.2 and <=0.12). May vary with your system. Please check at: https://pytorch.org/get-started/previous-versions/.
    1. e.g.: `pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115`
5. `cd` into the cloned repo
5. `pip install -e .`
6. `pip install -U openmim`
7. `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/{torch_version}/index.html`. Note that pre-built wheels (fast installs without needing to build) only exist for some versions of torch and CUDA. Check compatibilities here: https://mmcv.readthedocs.io/en/v1.6.2/get_started/installation.html
    1. e.g.: `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/11.5/1.11.0/index.html`

### Data

- Download the flood detection dataset from [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11). Splits in the `mmsegmentation` format are available in the `data_splits` folders.
- Download the [NASA HLS fire scars dataset](https://huggingface.co/datasets/nasa-impact/hls_burn_scars) from Hugging Face.
- Download the [NASA HLS multi-temporal crop classification dataset](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification) fro Hugging Face.
- Download the [NASA HLS irrigation_scenes dataset](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_irrigation_scenes) from HuggingFace.

## Running the finetuning
1. In the `configs` folder there are the three config examples for the three segmentation tasks. Complete the configs with your setup specifications. Parts that must be completed are marked with `#TO BE DEFINED BY USER`. They relate to where you downloaded the dataset, pretrained model weights, test set (e.g. regular one or Bolivia out of bag data) and where you are going to save the experiment outputs.

2.
  a. With the conda env created above activated, run either of the commands below:

    mim train mmsegmentation --launcher pytorch configs/sen1floods11_config.py
    mim train mmsegmentation --launcher pytorch configs/burn_scars_config.py
    mim train mmsegmentation --launcher pytorch configs/multi_temporal_crop_classification.py
    mim train mmsegmentation --launcher pytorch configs/irrigation_scenes_config.py

  b. To run testing:

    mim test mmsegmentation configs/sen1floods11_config.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"
    mim test mmsegmentation configs/burn_scars_config.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"
    mim test mmsegmentation configs/multi_temporal_crop_classification.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"

## Checkpoints on Hugging Face
We also provide checkpoints on Hugging Face for the [burn scars detection](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-burn-scar) and the [multi temporal crop classification tasks](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification).

## Running the inference
We provide a script to run inference on new data in geotiff format. The data can be of any shape (e.g. height and width) as long as it follows the bands/channels of the original dataset. Below an example:

```
python model_inference.py -config /path/to/config/config.py -ckpt /path/to/checkpoint/checkpoint.pth -input /input/folder/ -output /output/folder/ -input_type tif -bands "[0,1,2,3,4,5]"
```

The `bands` parameter is useful in case the files to run inference on have the data in different order/indexes than the original dataset.

## Additional documentation
This project builds on [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) and [MMCV](https://mmcv.readthedocs.io/en/v1.5.0/). For additional documentation, consult their docs (please note this is currently version 0.30.0 of MMSegmentation and version 1.5.0 of MMCV, not latest).
