# Flood segmentation by foundation model finetuning
This repository shows an example of how the geospatial foundation model can be finetuned for a downstream task.
In this case, we showcase flood segmentation with the [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset.

## The approach
### Background
We provide a pretrained backbone that can be used for various downstream remote sensing tasks.

To finetune for these tasks in this repository, we make use of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/), which provides an extensible framework for segmentation tasks. 

[MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) allows us to concatenate necks and heads appropriate for any segmentation downstream task to the encoder, and then perform the finetuning. This only requires the setup of a [configuration file](./configs/geospatial_fm_sen1floods11_finetune.py) detailing the desired model architecture, dataset setup and training strategy. 

We build extensions on top of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) to support our encoder and provide classes to read and augment remote sensing data (from .tiff files) using [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) data pipelines. These extensions can be found in the [sen1floods11](../sen1floods11/) directory. If more advanced functionality is necessary, it should be added there.

### The pretrained backbone
The pretrained model we work with is a [ViT](https://arxiv.org/abs/2010.11929) trained as a [Masked Auto Encoder](https://arxiv.org/abs/2111.06377). This is trained on [HLS](https://hls.gsfc.nasa.gov/) data. The encoder from this model is made available as the backbone.

### The architecture
We provide a simple architecture in [the configuration file](./configs/geospatial_fm_sen1floods11_finetune.py) that adds a neck and segmentation head to the backbone. The neck concatenates and processes the transformer's token based embeddings into one that can be fed into convolutional layers. The head processes this embedding into a segmentation mask. The code for these can be found in [this file](./geospatial_fm/geospatial_fm.py).

## Setup
### Dependencies
1. Clone this repository
2. `conda create -n <environment-name> python==3.7.1`
3. `conda activate <environment-name>`
4. `cd mmsegmentation`
5. `pip install -r projects/sen1floods11/requirements.txt`
6. `pip install -e .`
7. `pip install -U openmim`
8. `mim install mmcv-full==1.5.0`

### Data
Download the dataset from [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11).


## Running the code
1. Complete the [configuration file](./configs/geospatial_fm_sen1floods11_finetune.py) with your setup specifications. Parts that must be completed are marked with `#TO BE DEFINED BY USER`. 

2.
    a. With the conda env activated, run `python tools/train.py projects/sen1floods11/configs/geospatial_fm_sen1floods11_finetune.py`. All model artifacts, logs and tensorflow logs will be saved in the directory defined in the config file.

    b. To run testing: `python tools/train.py projects/sen1floods11/configs/geospatial_fm_sen1floods11_finetune.py <path-to-train-file>`
    

## Additional documentation
This project builds on [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/). For additional documentation, consult their docs (please note this is currently version 0.30.0 of MMSegmentation, not latest).