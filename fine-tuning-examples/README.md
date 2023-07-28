# Image segmentation by foundation model finetuning
This repository shows two examples of how the geospatial foundation model can be finetuned for downstream tasks. These are flood detection using the [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset and fire scars detection using the NASA fire scars dataset [NASA fire scars dataset](https://huggingface.co/datasets/nasa-impact/hls_burn_scars)

## The approach
### Background
We provide a pretrained backbone that can be used for various downstream remote sensing tasks.

To finetune for these tasks in this repository, we make use of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/), which provides an extensible framework for segmentation tasks. 

[MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) allows us to concatenate necks and heads appropriate for any segmentation downstream task to the encoder, and then perform the finetuning. This only requires the setup of a config file detailing the desired model architecture, dataset setup and training strategy. 

We build extensions on top of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) to support our encoder and provide classes to read and augment remote sensing data (from .tiff files) using [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) data pipelines. These extensions can be found in the [geospatial_fm](./geospatial_fm/) directory, and they are installed as a package on the top of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) for easy to use. If more advanced functionality is necessary, it should be added there.

### The pretrained backbone
The pretrained model we work with is a [ViT](https://arxiv.org/abs/2010.11929) trained as a [Masked Auto Encoder](https://arxiv.org/abs/2111.06377). This is trained on [HLS](https://hls.gsfc.nasa.gov/) data. The encoder from this model is made available as the backbone and the weights can be downloaded from Hugging Face [here](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/blob/main/Prithvi_100M.pt).

### The architecture
We provide a simple architecture in [the configuration file](./configs/config.py) that adds a neck and segmentation head to the backbone. The neck concatenates and processes the transformer's token based embeddings into one that can be fed into convolutional layers. The head processes this embedding into a segmentation mask. The code for these can be found in [this file](./geospatial_fm/geospatial_fm.py).

## Setup
### Dependencies
1. Clone this repository
2. `conda create -n <environment-name> python==3.9`
3. `conda activate <environment-name>`
4. `cd fine-tuning-examples`
5. Install torch and torchvision: `pip install torch==1.7.1 torchvision==0.8.2` (May vary with your system. Please check at https://pytorch.org/get-started/locally/)
6. `pip install --editable .`
7. `pip install -U openmim`
8. `mim install mmcv-full==1.5.0` (This may take a while for torch > 1.7.1, as wheel must be built)

### Data

- Download the flood detection dataset from [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11).
- Download the fire scars detection dataset from [Hugging Face](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars).
- Download the irrigation_scenes dataset from [HuggingFace](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_irrigation_scenes)

## Running the code
1. Complete the configs with your setup specifications. Parts that must be completed are marked with `#TO BE DEFINED BY USER`. They relate to where you downloaded the dataset, pretrained model weights, test set (e.g. regular one or Bolivia out of bag data) and where you are going to save the experiment outputs.

2.
  a. With the conda env created above activated and from the `fine-tuning-examples` folder, run either of the commands below:

    mim train mmsegmentation --launcher pytorch configs/sen1floods11_config.py
    mim train mmsegmentation --launcher pytorch configs/firescars_config.py
    mim train mmsegmentation --launcher pytorch configs/irrigation_scenes_config.py

  b. To run testing:

    mim test mmsegmentation configs/sen1floods11_config.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"
    mim test mmsegmentation configs/firescars_config.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"

## Additional documentation
This project builds on [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) and [MMCV](https://mmcv.readthedocs.io/en/v1.5.0/). For additional documentation, consult their docs (please note this is currently version 0.30.0 of MMSegmentation and version 1.5.0 of MMCV, not latest).
