# CNN Style Transfer

Implemented image style transfer in Keras using pre-trained VGGNet. 

## Usage

Run style_transfer.py on CL with the following arguments:

--contentPath: path/to/contentImage (default Paris)
--stylePath: path/to/styleImage (default StarryNight)
--outWidth: width of output image (int)
--outHeight: height of output image (int)
--a: content weight (float)
--b: style weight (float)
--includeTotalVar: whether or not to include regularization loss for total variation (bool, default True)
--t: total variation loss weight (float)
--rounds: number of iterations for gradient descent (int)
--save: whether or not to save the output image (bool)
--randOut: whether or not to randomly initialize the output image (bool)
