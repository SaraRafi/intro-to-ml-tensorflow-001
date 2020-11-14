

import tensorflow as tf
import numpy as np

image_size = 224

def process_image(image):
    image = tf.image.resize(image, (image_size, image_size))
    image = image/255.
    return image.numpy()
