
### Model and Input
The model expects remote sensing data in a video format (B, C, T, H, W). Note that the temporal dimension is very important here and not present in most 
other works around remote sensing modeling. Being able to handle a time series of remote sensing images can be very helpful to a variety of downstream tasks. The model can also handle static image which can be simply fed into the model with T=1.

### Code
The model follows [original mae repo](https://github.com/facebookresearch/mae) with modifications including:
1. replace 2D patch embed with 3D patch embed
2. replace 2D positional embed with 3D positional embed
3. replace 2D patchify and unpatchify with 3D
4. etc.

### Pre-training
The model was pre-trained with Harmonised Landsat and Sentinel 2 data from NASA using the following bands:

* Blue
* Green
* Red
* Narrow NIR
* SWIR 1
* SWIR 2

### Download the model
You can download the Prithvi model from [here](https://ibm.ent.box.com/s/vwcyi2wtt31db20m4nnordmufgf9sv65/file/1255258189196).