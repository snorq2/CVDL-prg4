from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras import layers
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

model = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)


img = Image.open("images/gc5.jpg")
img_dims = img.size
img = img.resize((224, 224))
img = np.reshape(np.asarray(img), (1, 224, 224, 3))
img_prc = preprocess_input(img)
prediction = model.predict(img_prc)
max_prediction = np.argmax(prediction)
last_layer = model.get_layer('block5_conv3')
gradient = K.gradients(model.output[:, max_prediction], last_layer.output)[0]
# img_cast = tf.cast(img_prc, tf.float32)
# # img_cast = img_prc
# with tf.GradientTape() as tape:
#     tape.watch(img_cast)
#     preds = model(img_cast)
#     top_class = preds[:, max_prediction]
    
# gradient = tape.gradient(top_class, img_cast)
ack = (1/(14 * 14)) * gradient
sum = tf.math.reduce_sum((ack * gradient), axis = 3)
rel = tf.nn.relu(sum)
# keras_resizer = layers.Resizing(img_dims[0], img_dims[1])
# resized = keras_resizer(rel)
resized = tf.image.resize(rel, img_dims)
# sal_map = resized.numpy()
# sal_map = np.array(resized, dtype=np.float32)
# sess = tf.compat.v1.Session()
sess = tf.session()
sal_map = sess.run(resized)
# with tf.compat.v1.Session() as sess:
#     sal_map = resized.eval(session = sess)
plt.imshow(sal_map)
plt.show()
print(decode_predictions(prediction, top=3))

# imtest = ImageDataGenerator()
# imgen = imtest.flow_from_directory(directory="images", target_size=(224, 224))

# img = np.asarray(Image.open("gc1.jpg"))
# img_prc = preprocess_input(img)
# prediction = model.predict(imgen)
# print(decode_predictions(prediction, top=3))