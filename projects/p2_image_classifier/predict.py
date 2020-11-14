
import argparse


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow_hub as hub
from tensorflow import keras
import os

import tensorflow_datasets as tfds

from PIL import Image

import image_preprocess


parser = argparse.ArgumentParser()

parser.add_argument("image_path" )
parser.add_argument("model_path")
parser.add_argument("--top_k" , dest = "top_k", type = int, required = False, default = 5)
parser.add_argument("--category_names", dest = "category_names", required = False)
args = parser.parse_args()


def predict(image_path, model_path, top_k, category_names):
    model = tf.keras.models.load_model(model_path , custom_objects={'KerasLayer':hub.KerasLayer})
    image  = Image.open(image_path)
    image_np = np.asarray(image)
    image_np = np.expand_dims(image_np, axis = 0)
    processed_image = image_preprocess.process_image(image_np)
    probs = model.predict(processed_image)
    top_k_values, top_k_indices = tf.nn.top_k(probs, k = top_k)

    if category_names == None:

        return top_k_values.numpy(), top_k_indices.numpy()

    # probs, classes = predict(image_path, model, top_k)
    elif category_names != None:

        objects = top_k_indices.numpy().flatten()
        y_pos = np.arange(top_k_indices.shape[1])
        performance = top_k_values.numpy().flatten()

        with open(category_names, 'r') as f:
            class_names = json.load(f)

        y_labels =[]

        for i in objects:
            label = str(i + 1)
            y_labels.append(class_names[label])

        return top_k_values.numpy().flatten(), y_labels


if __name__=='__main__':
    print(predict(args.image_path, args.model_path, args.top_k, args.category_names))
