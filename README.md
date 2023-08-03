# Image segmentation by foundation model finetuning

This repository shows three examples of how [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) can be finetuned for downstream tasks. The examples include flood detection using Sentinel-2 data from the [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset, burn scars detection using the [NASA HLS fire scars dataset](https://huggingface.co/datasets/nasa-impact/hls_burn_scars) and multi-temporal crop classification using the [NASA HLS multi-temporal crop classification dataset](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification).

## The approach
### Background
To finetune for these tasks in this repository, we make use of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/), which provides an extensible framework for segmentation tasks. 

[MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) allows us to concatenate necks and heads appropriate for any segmentation downstream task to the encoder, and then perform the finetuning. This only requires setting up a config file detailing the desired model architecture, dataset setup and training strategy. 

We build extensions on top of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) to support our encoder and provide classes to read and augment remote sensing data (from .tiff files) using [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) data pipelines. These extensions can be found in the [geospatial_fm](./geospatial_fm/) directory, and they are installed as a package on the top of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) for ease of use. If more advanced functionality is necessary, it should be added there.

### The pretrained backbone
The pretrained model we work with is a [ViT](https://arxiv.org/abs/2010.11929)operating as a [Masked Autoencoder](https://arxiv.org/abs/2111.06377), trained on [HLS](https://hls.gsfc.nasa.gov/) data. The encoder from this model is made available as the backbone and the weights can be downloaded from Hugging Face [here](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/blob/main/Prithvi_100M.pt).


### The architectures
We use a simple architecture that adds a neck and segmentation head to the backbone. The neck concatenates and processes the transformer's token based embeddings into an embedding that can be fed into convolutional layers. The head processes this embedding into a segmentation mask. The code for the architecture can be found in [this file](./geospatial_fm/geospatial_fm.py).

### The pipeline
Additionally, we provide extra components for data loading pipelines in [geospatial_pipelines.py](./geospatial_fm/geospatial_pipelines.py). These are documented in the file.
We observe the MMCV convention that all operations assumes a channel-last format. Our tiff loader also assumes this is the format in which files are written, and offers a flag to automatically transpose a to channel-last format if this is not the case.
*However*, we also introduce some components with the prefix `Torch`, such as `TorchNormalize`. These components assume the torch convention of channel-first.

At some point during the pipeline, before feeding the data to the model, it is necessary to change to channel-first format.
We reccomend implementing the change after the `ToTensor` operation (which is also necessary at some point), using the `TorchPermute` operation.
## Setup
### Dependencies
1. Clone this repository
2. `conda create -n <environment-name> python==3.9`
3. `conda activate <environment-name>`
4. Install torch (tested for >=1.7.1 and <=1.11.0) and torchvision (tested for >=0.8.2 and <=0.12). May vary with your system. Please check at: https://pytorch.org/get-started/previous-versions/.
    1. e.g.: `pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115`
5. `cd` into the cloned repo
5. `pip install -e .`
6. `pip install -U openmim`
7. `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/{torch_version}/index.html`. Note that pre-built wheels (fast installs without needing to build) only exist for some versions of torch and CUDA. Check compatibilities here: https://mmcv.readthedocs.io/en/v1.6.2/get_started/installation.html
    1. e.g.: `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html`

### Data

The flood detection dataset can be downloaded from [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11). Splits in the `mmsegmentation` format are available in the `data_splits` folders.


The [NASA HLS fire scars dataset](https://huggingface.co/datasets/nasa-impact/hls_burn_scars) can be downloaded from Hugging Face.

The [NASA HLS multi-temporal crop classification dataset](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification) can be downloaded from Hugging Face.


## Running the finetuning
1. In the `configs` folder there are three config examples for the three segmentation tasks. Complete the configs with your setup specifications. Parts that must be completed are marked with `#TO BE DEFINED BY USER`. They relate to the location where you downloaded the dataset, pretrained model weights, the test set (e.g. regular one or Bolivia out of bag data) and where you are going to save the experiment outputs.

2. 
    a. With the conda env created above activated, run:
    
    `mim train mmsegmentation --launcher pytorch configs/sen1floods11_config.py` or 
    
    `mim train mmsegmentation --launcher pytorch configs/burn_scars.py` or
    
    `mim train mmsegmentation --launcher pytorch configs/multi_temporal_crop_classification.py`
    
    b. To run testing: 
    
    `mim test mmsegmentation configs/sen1floods11_config.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"` or 
    
    `mim test mmsegmentation configs/burn_scars.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"` or
    
    `mim test mmsegmentation configs/multi_temporal_crop_classification.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"`

## Checkpoints on Hugging Face
We also provide checkpoints on Hugging Face for the [burn scars detection](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-burn-scar) and the [multi temporal crop classification tasks](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification).

## Running the inference
We provide a script to run inference on new data in GeoTIFF format. The data can be of any shape (e.g. height and width) as long as it follows the bands/channels of the original dataset. An example is shown below.

```
python model_inference.py -config /path/to/config/config.py -ckpt /path/to/checkpoint/checkpoint.pth -input /input/folder/ -output /output/folder/ -input_type tif -bands "[0,1,2,3,4,5]"
```

The `bands` parameter is useful in case the files used to run inference have the data in different orders/indexes than the original dataset.

## Additional documentation
This project builds on [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) and [MMCV](https://mmcv.readthedocs.io/en/v1.5.0/). For additional documentation, consult their docs (please note this is currently version 0.30.0 of MMSegmentation and version 1.5.0 of MMCV, not latest).

## Citation

If this repository helped your research, please cite `HLS foundation` in your publications. Here is an example BibTeX entry:

```
@software{HLS_Foundation_2023,
    author          = {Jakubik, Johannes and Chu, Linsong and Fraccaro, Paolo and Bangalore, Ranjini and Lambhate, Devyani and Das, Kamal and Oliveira Borges, Dario and Kimura, Daiki and Simumba, Naomi and Szwarcman, Daniela and Muszynski, Michal and Weldemariam, Kommy and Zadrozny, Bianca and Ganti, Raghu and Costa, Carlos and Watson, Campbell and Mukkavilli, Karthik and Roy, Sujit and Phillips, Christopher and Ankur, Kumar and Ramasubramanian, Muthukumaran and Gurung, Iksha and Leong, Wei Ji and Avery, Ryan and Ramachandran, Rahul and Maskey, Manil and Olofossen, Pontus and Fancher, Elizabeth and Lee, Tsengdar and Murphy, Kevin and Duffy, Dan and Little, Mike and Alemohammad, Hamed and Cecil, Michael and Li, Steve and Khallaghi, Sam and Godwin, Denys and Ahmadi, Maryam and Kordi, Fatemeh and Saux, Bertrand and Pastick, Neal and Doucette, Peter and Fleckenstein, Rylie and Luanga, Dalton and Corvin, Alex and Granger, Erwan},
    doi             = {10.57967/hf/0952},
    month           = aug,
    title           = {{HLS Foundation}},
    repository-code = {https://github.com/nasa-impact/hls-foundation-os},
    year            = {2023}
}
```
