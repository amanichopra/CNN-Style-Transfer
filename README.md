# CNN Style Transfer

Implemented image style transfer in Keras using pre-trained VGGNet. 

## Usage

Run style_transfer.py on CL with the following arguments:

--contentPath: path/to/contentImage (default Paris)<br/>
--stylePath: path/to/styleImage (default StarryNight)<br/>
--outWidth: width of output image (int)<br/>
--outHeight: height of output image (int)<br/>
--a: content weight (float)<br/>
--b: style weight (float)<br/>
--includeTotalVar: whether or not to include regularization loss for total variation (bool, default True)<br/>
--t: total variation loss weight (float)<br/>
--rounds: number of iterations for gradient descent (int)<br/>
--save: whether or not to save the output image (bool)<br/>
--randOut: whether or not to randomly initialize the output image (bool)
