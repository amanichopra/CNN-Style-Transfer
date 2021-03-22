import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import random
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.framework.ops import disable_eager_execution
import imageio
import argparse
import numpy

disable_eager_execution()
random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#=============================<Helper Fuctions>=================================
def deprocessImage(img, height, width):
    img = img.reshape((height, width, 3))
    # Remove zero-center by mean pixel with respect to imageNet dataset
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68

    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def getGramMatrix(features):
    if K.image_data_format() == 'channels_first': features = K.flatten(features)
    else: features = K.batch_flatten(K.permute_dimensions(features, (2, 0, 1)))
    return K.dot(features, K.transpose(features)) # compute dot product across all channels

def strToBool(str):
    if str == 'True': return True
    else: return False


#========================<Loss Function Builder Functions>======================
def getStyleLoss(styleFeatures, transferFeatures):
    styleGram = getGramMatrix(styleFeatures)
    transferGram = getGramMatrix(transferFeatures)
    size = styleFeatures.shape[0] * styleFeatures.shape[1]
    loss = K.sum(K.square(styleGram - transferGram)) / (4.0 * (styleFeatures.shape[2]**2) * (size**2))
    return loss


def getContentLoss(contentFeatures, transferFeatures):
    return K.sum(K.square(transferFeatures - contentFeatures))


def getTotalVariationLoss(transferFeatures):
    shape = transferFeatures.shape
    a = K.square(transferFeatures[:,:shape[1]-1,:shape[2]-1,:] - transferFeatures[:,1:,:shape[2]-1, :])
    b = K.square(transferFeatures[:,:shape[1]-1,:shape[2]-1,:] - transferFeatures[:,:shape[1]-1,1:,:])
    return K.sum(K.pow(a+b, 1.25))


def getTotalLoss(mod, styleWeight, contentWeight, incorporateTotalVariationLoss=False, transferImage=None, totalVarWeight=None,
                 styleLayerNames=["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"],
                 contentLayerName="block5_conv2"):
    loss = 0.0

    print("   Calculating content loss.")
    contentLayer = mod[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    transferOutput = contentLayer[2, :, :, :]
    loss += contentWeight * getContentLoss(contentOutput, transferOutput)

    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        styleLayer = mod[layerName]
        styleOutput = styleLayer[1, :, :, :]
        transferOutput = styleLayer[2, :, :, :]
        loss += (styleWeight / len(styleLayerNames)) * getStyleLoss(styleOutput, transferOutput)

    if incorporateTotalVariationLoss and transferImage is not None and totalVarWeight is not None:  # ensure transferImage is tensor
        print("   Calculating total variation loss.")
        loss += totalVarWeight * getTotalVariationLoss(transferImage)

    return loss


#=========================<Pipeline Functions>==================================
def getRawData(CONTENT_IMG_PATH, STYLE_IMG_PATH, tHeight, tWidth):
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH, target_size=(tHeight, tWidth))
    cImgW, cImgH = cImg.size
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH, target_size=(tHeight, tWidth))
    sImgW, sImgH = sImg.size
    print("      Images have been loaded.")
    return ((cImg, cImgH, cImgW), (sImg, sImgH, sImgW), (tImg, cImgH, cImgW))


def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def styleTransfer(cData, sData, tData, tHeight, tWidth, incorporateTotalVariationLoss, rounds, styleWeight, contentWeight, totalVarWeight, outFolder):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, tHeight, tWidth, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)

    model = vgg19.VGG19(weights="imagenet", include_top=False, input_tensor=inputTensor)
    modOutputs = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")

    loss = getTotalLoss(modOutputs, styleWeight, contentWeight, totalVarWeight=totalVarWeight, incorporateTotalVariationLoss=incorporateTotalVariationLoss,
                        transferImage=genTensor)
    gradients = K.gradients(loss, genTensor)[0]
    getLossAndGradients = K.function([genTensor], [loss, gradients])

    print("   Beginning transfer.")
    for i in range(rounds):
        print("   Step %d." % i)

        class LossGradComputer(object):
            def loss(self, x):
                x = x.reshape((1, tHeight, tWidth, 3))
                outs = getLossAndGradients([x])
                self.loss_value = outs[0]
                self.grad_values = outs[1].flatten().astype('float64')
                return self.loss_value

            def grads(self, x):
                grad_values = np.copy(self.grad_values)
                self.loss_value = None
                self.grad_values = None
                return grad_values

        computer = LossGradComputer()
        # gradient descent on loss
        tData, tLoss, _ = fmin_l_bfgs_b(computer.loss, tData.flatten(), fprime=computer.grads, maxfun=20, maxiter=1300)
        print("      Loss: %f." % tLoss)

        if outFolder:
            # save image
            img = deprocessImage(tData, tHeight, tWidth)
            saveFile = '{}/iter_{}.png'.format(outFolder, i)
            imageio.imwrite(saveFile, img)
            print("      Image saved to \"%s\"." % saveFile)

    print("   Transfer complete.")


#=========================<Main>================================================
def main():
    CONTENT_IMG_PATH = tf.keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
    STYLE_IMG_PATH = tf.keras.utils.get_file("starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg")

    # Dimensions of the picture that will be generated.
    tWidth, tHeight = load_img(CONTENT_IMG_PATH).size

    parser = argparse.ArgumentParser(description='StyleTransfer')

    parser.add_argument('--contentPath', default=CONTENT_IMG_PATH)  # path to content image
    parser.add_argument('--stylePath', default=STYLE_IMG_PATH)  # path to style image
    parser.add_argument('--outWidth', type=int, default=tWidth)  # desired width of output image
    parser.add_argument('--outHeight', type=int, default=tHeight)  # desired height of output image

    parser.add_argument('--a', type=float, default=0.1)  # content weight
    parser.add_argument('--b', type=float, default=1)  # style weight
    parser.add_argument('--includeTotalVar', default='True')  # whether or not to include total variation as regularizer in loss
    parser.add_argument('--t', type=float, default=1)  # total variation weight
    parser.add_argument('--rounds', type=int, default=3)  # number of transfer rounds
    parser.add_argument('--save', default='True')  # whether or not to save the output image
    parser.add_argument('--randOut', default='False') # whether or not generated image should be initialized randomly

    args = parser.parse_args()
    outFolder = None
    if strToBool(args.save):
        try:
            outFolder = args.contentPath[:args.contentPath.index('/')]
        except ValueError:
            raise Exception('Ensure content and style images are in a folder labeled "[content]-[style]" stored in the cwd!')

    print("Starting style transfer program.")
    raw = getRawData(args.contentPath, args.stylePath, args.outHeight, args.outWidth)
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    # Transfer image.
    if strToBool(args.randOut):
        tData = numpy.random.rand(100,100,3) * 255
    else:
        tData = preprocessData(raw[2])

    styleTransfer(cData, sData, tData, args.outHeight, args.outWidth, strToBool(args.includeTotalVar), args.rounds, args.b, args.a, args.t, outFolder)
    print("Done. Goodbye.")


if __name__ == "__main__":
    main()
