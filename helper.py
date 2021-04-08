import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from model import Unet
from data import Dataset


def process(image_path, model):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img)
    height, width = tf.shape(img)[:2]
    img = tf.image.resize(img, (128, 128))
    img = img / 255.
    img = np.expand_dims(img, axis=0)

    pred_mask = model(img)
    pred_mask = tf.argmax(pred_mask, axis=-1).numpy()
    pred_mask = tf.transpose(pred_mask, (1, 2, 0))
    pred_mask = tf.cast((pred_mask * 100), tf.uint8)
    pred_mask = tf.image.resize(pred_mask, (height, width))

    return pred_mask.numpy()


def progressBar(epoch, current, total, loss, accuracy,
                name="Training", barLength=20):
    percent = current * 100 / total
    arrow = '=' * int(percent / 100 * barLength - 1) + '>'
    spaces = '.' * (barLength - len(arrow))
    print(f'{name} - Epoch {epoch}: [{arrow}{spaces}] {round(percent,2)}% - Loss: {loss} - Accuracy: {accuracy}',
          end='\r')


def creat_example(collection, labels, demo_dir, name, col=3):

    fig, axs = plt.subplots(1, col, figsize=(20, 10))
    axs = axs.flatten()

    for i in range(len(axs)):
        axs[i].imshow(collection[i])
        axs[i].set_title(labels[i])
        axs[i].axis("off")

    fig.savefig(os.path.join(demo_dir, name))


def example_random(model,
                   image_path,
                   demo_dir="./examples",
                   name="random_example01.png"):

    if not(os.path.exists(demo_dir)):
        os.mkdir(demo_dir)

    orig_img = cv2.imread(image_path)[..., ::-1]
    pred_mask = process(image_path, model)
    labels = ["Original Image", "Predicted Mask"]
    collection = [orig_img, pred_mask]

    creat_example(collection, labels, demo_dir, name, 2)


def example_test(model,
                 images_dir="./data/images",
                 annotations_dir="./data/annotations",
                 demo_dir="./examples",
                 name="example01.png"):

    if not(os.path.exists(demo_dir)):
        os.mkdir(demo_dir)

    dataset = Dataset(training=False,
                      images_dir=images_dir,
                      annotations_dir=annotations_dir)

    image_name = np.random.choice(dataset.get_images_list())
    image_path = os.path.join(images_dir, image_name + ".jpg")
    mask_path = os.path.join(annotations_dir, "trimaps", image_name + ".png")

    orig_img = cv2.imread(image_path)[..., ::-1]
    true_mask = (cv2.imread(mask_path) * 100)[..., ::-1]
    pred_mask = process(image_path, model)

    labels = ["Original Image", "True Mask", "Predicted Mask"]
    collection = [orig_img, true_mask, pred_mask]

    creat_example(collection, labels, demo_dir, name, 3)
