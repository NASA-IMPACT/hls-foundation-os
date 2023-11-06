import argparse
import glob
import os
import time

import numpy as np
import rasterio
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose, LoadImageFromFile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference on flood detection fine-tuned model"
    )
    parser.add_argument("-config", help="path to model configuration file")
    parser.add_argument("-ckpt", help="path to model checkpoint")
    parser.add_argument("-input", help="path to input images folder for inference")
    parser.add_argument("-output", help="path to save output image")
    parser.add_argument("-input_type", help="file type of input images", default="tif")
    parser.add_argument(
        "-bands",
        help="bands in the file where to find the relevant data",
        type=int,
        nargs="+",
    )
    parser.add_argument("-device", help="device", default="cuda", type=str)

    args = parser.parse_args()

    return args


def open_tiff(fname):
    with rasterio.open(fname, "r") as src:
        data = src.read()

    return data


def write_tiff(img_wrt, filename, metadata):
    """
    It writes a raster image to file.

    :param img_wrt: numpy array containing the data (can be 2D for single band or 3D for multiple bands)
    :param filename: file path to the output file
    :param metadata: metadata to use to write the raster to disk
    :return:
    """

    with rasterio.open(filename, "w", **metadata) as dest:
        if len(img_wrt.shape) == 2:
            img_wrt = img_wrt[None]

        for i in range(img_wrt.shape[0]):
            dest.write(img_wrt[i, :, :], i + 1)

    return filename


def get_meta(fname):
    with rasterio.open(fname, "r") as src:
        meta = src.meta

    return meta


def inference_segmentor(model, imgs, custom_test_pipeline=None):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = (
        [LoadImageFromFile()] + cfg.data.test.pipeline[1:]
        if custom_test_pipeline == None
        else custom_test_pipeline
    )
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    imgs = imgs if isinstance(imgs, list) else [imgs]
    for img in imgs:
        img_data = {"img_info": {"filename": img}}
        img_data = test_pipeline(img_data)
        data.append(img_data)
    # print(data.shape)

    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # data = collate(data, samples_per_gpu=len(imgs))
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # img_metas = scatter(data['img_metas'],'cpu')
        # data['img_metas'] = [i.data[0] for i in data['img_metas']]

        img_metas = data["img_metas"].data[0]
        img = data["img"]
        data = {"img": img, "img_metas": img_metas}

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def inference_on_file(model, target_image, output_image, custom_test_pipeline):
    time_taken = -1
    try:
        st = time.time()
        print("Running inference...")
        result = inference_segmentor(model, target_image, custom_test_pipeline)
        print("Output has shape: " + str(result[0].shape))

        ##### get metadata mask
        mask = open_tiff(target_image)
        meta = get_meta(target_image)
        mask = np.where(mask == meta["nodata"], 1, 0)
        mask = np.max(mask, axis=0)[None]

        result[0] = np.where(mask == 1, -1, result[0])

        ##### Save file to disk
        meta["count"] = 1
        meta["dtype"] = "int16"
        meta["compress"] = "lzw"
        meta["nodata"] = -1
        print("Saving output...")
        write_tiff(result[0], output_image, meta)
        et = time.time()
        time_taken = np.round(et - st, 1)
        print(
            f"Inference completed in {str(time_taken)} seconds. Output available at: "
            + output_image
        )

    except:
        print(f"Error on image {target_image} \nContinue to next input")

    return time_taken


def process_test_pipeline(custom_test_pipeline, bands=None):
    # change extracted bands if necessary
    if bands is not None:
        extract_index = [
            i for i, x in enumerate(custom_test_pipeline) if x["type"] == "BandsExtract"
        ]

        if len(extract_index) > 0:
            custom_test_pipeline[extract_index[0]]["bands"] = bands

    collect_index = [
        i for i, x in enumerate(custom_test_pipeline) if x["type"].find("Collect") > -1
    ]

    # adapt collected keys if necessary
    if len(collect_index) > 0:
        keys = [
            "img_info",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ]
        custom_test_pipeline[collect_index[0]]["meta_keys"] = keys

    return custom_test_pipeline


def inference_on_files(
    config_path, ckpt, input_type, input_path, output_path, bands, device
):
    # load model
    config = Config.fromfile(config_path)
    config.model.backbone.pretrained = None
    model = init_segmentor(config, ckpt, device)

    # identify images to predict on
    target_images = glob.glob(os.path.join(input_path, "*." + input_type))

    print("Identified images to predict on: " + str(len(target_images)))

    # check if output folder available
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # modify test pipeline if necessary
    custom_test_pipeline = process_test_pipeline(model.cfg.data.test.pipeline, bands)

    # for each image predict and save to disk
    for i, target_image in enumerate(target_images):
        print(f"Working on Image {i}")
        output_image = os.path.join(
            output_path,
            target_image.split("/")[-1].replace(
                "." + input_type, "_pred." + input_type
            ),
        )

        inference_on_file(model, target_image, output_image, custom_test_pipeline)


def main():
    # unpack args
    args = parse_args()
    config_path = args.config
    ckpt = args.ckpt
    input_type = args.input_type
    input_path = args.input
    output_path = args.output
    bands = args.bands
    device = args.device

    inference_on_files(
        config_path, ckpt, input_type, input_path, output_path, bands, device
    )


if __name__ == "__main__":
    main()
