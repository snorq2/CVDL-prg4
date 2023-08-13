from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

#  TensorFlow is fairly verbose.  Suppresses standard messages from terminal.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Function to generate needed files from picture index
# Input is the picture number from the assignment
# Outputs are the unprocessed and processed image


def open_process_img(picnum):
    img = Image.open("images/gc" + str(picnum) + ".jpg")
    img_prc = img.resize((224, 224))
    img_prc = np.reshape(np.asarray(img_prc), (1, 224, 224, 3))
    img_prc = preprocess_input(img_prc)
    return img, img_prc

# Converts and normalizes inputs, overlays heatmap onto image, and saves image per Grad_CAM run parameters
# Uses cv2.COLORMAP_JET to color the heatmap
# Input is the original image, a Grad-CAM to overlay, name of the predicted class, assignment picture number, and the prediction place number
# No function output.  Saves to the file and prints the file path/name


def save_with_overlay(img, grad_cam, class_name, picnum, prednum):
    extension = ".jpg"
    img_np = np.array(img)
    grad_cam = cv2.resize(grad_cam, img.size)
    grad_cam = (grad_cam / grad_cam.max()) * 255
    grad_cam = (np.rint(grad_cam)).astype(np.uint8)
    heatmap = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET)
    img_save = cv2.addWeighted(img_np, 0.8, heatmap, 0.6, 0)
    save_name = "grad_cam/gc" + \
        str(picnum) + "_prediction-num-" + \
        str(prednum) + "_" + class_name
    cv2.imwrite(save_name + extension, img_save)
    cv2.imwrite(save_name + "_heatmap" + extension, heatmap)
    print("Saved " + save_name)

# Extracts nth prediction from the included set of target classes
# Uses technique from this post to develop the Numpy index extraction:
# https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
# Input is the class array from the VGG16 model and the nth prediction
# Output is the predicted class name and associated index


def get_specific_prediction(classes, n_prediction):
    classes = classes[0].numpy()
    top_k = decode_predictions(classes, top=n_prediction)
    class_name = top_k[0][n_prediction-1][1]
    # class_name = class_name[1]
    classes = classes.flatten()
    indexes = np.argpartition(classes, -n_prediction)[-n_prediction:]
    indexes_sorted = indexes[np.argsort(classes[indexes])]
    index_predict = indexes_sorted[-n_prediction]
    return class_name, int(index_predict)

# Executes Grad_CAM on input image
# Input is the processed image and nth prediction desired
# Output is the predicted class and the coarse saliency map
# Gradient/activation extraction developed from primary Tensorflow documentation and a Stackoverflow post:
# https://stackoverflow.com/questions/63233460/how-to-take-gradient-in-tensorflow-for-vgg16


def grad_CAM(img_prc, n_prediction):
    # Instantiate model
    model = VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    # Find target layer
    last_layer = model.get_layer('block5_conv3')

    # Generate model based on VGG16, but with two outputs - one at the activation layer, one at the output
    # This lets us capture the activations and gradients at the desired layer while also getting our class predictions at the same time
    grad_model = tf.keras.models.Model(
        model.inputs, [last_layer.output, model.outputs])

    # Run our predictions in a GradientTape instance
    # Extracts the nth prediction from the class prediction output
    with tf.GradientTape() as tape:
        activations, classes = grad_model(img_prc)
        prediction, predict_index = get_specific_prediction(
            classes, n_prediction)
        output = classes[0]
        output = output[:, predict_index]

    # Capture the activations and the gradient from the chosen prediction
    ak = activations[0]
    gradients = tape.gradient(output, activations)

    # Execute global pooling of the gradient by summing each layer with respect to the layer axis and dividing by 14*14
    ack = tf.reduce_sum(gradients, axis=(0, 1, 2)) * (1/(14*14))

    # Set up the grad_cam pool
    ack_ak = np.zeros((14, 14))

    # Weigh and sum the activation layers
    for i in range(0, 512):
        ack_ak += ack[i] * ak[:, :, i]

    # Execute RELU to generate coarse saliency map
    grad_cam = np.maximum(ack_ak, 0)

    return prediction, grad_cam


# Run an image through all the processing, Grad-CAM, and processing/saving steps
# Input is the assignment image number, and the desired prediction number to target
# No output, prints steps and output files to terminal, and saves the heatmap/image overlay
def run_grad_CAM(picnum, n_prediction):
    img, img_prc = open_process_img(picnum)
    prediction, grad_cam = grad_CAM(img_prc, n_prediction)
    print("Prediction " + str(n_prediction) + " for this image: " + prediction)
    print("Saving overlay")
    save_with_overlay(img, grad_cam, prediction, picnum, n_prediction)


# Loop for all the assignment requirements
for picnum in range(1, 6):
    for pred_place in range(1, 4):
        print("Running place " + str(pred_place) +
              " prediction on picture #" + str(picnum))
        run_grad_CAM(picnum, pred_place)

print("All runs finished")
