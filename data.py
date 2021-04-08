import os
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE


class Dataset:
    def __init__(self,
                 training=True, 
                 images_dir="./data/images", 
                 annotations_dir="./data/annotations",
                 height=128,
                 width=128):

        self.image_dir = images_dir
        self.annotations_dir = annotations_dir
        self.height = height
        self.width = width
        self.training = training

        if self.training:
            self.path = os.path.join(self.annotations_dir, "trainval.txt")
        else:
            self.path = os.path.join(self.annotations_dir, "test.txt")

        self.images_list = self.get_images_list()

    def get_dataset(self, batch_size=20, buffer_size=10000):
        ds = tf.data.Dataset.zip((self.get_inputs(), self.get_labels()))
        if self.training:
            ds = ds.map(self._random_flip, num_parallel_calls=AUTOTUNE)
            ds = ds.cache().shuffle(buffer_size).batch(batch_size).repeat()
            ds = ds.prefetch(buffer_size=AUTOTUNE)
        else:
            ds = ds.batch(batch_size)
        return ds

    def get_inputs(self):
        image_paths = [os.path.join(self.image_dir, name + ".jpg") for name in self.images_list]
        ds = tf.data.Dataset.from_tensor_slices(image_paths)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda img: tf.image.decode_jpeg(img, channels=3), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda img: tf.image.resize(img, [self.height, self.width]), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda img: tf.math.divide(img, tf.constant(255.)), num_parallel_calls=AUTOTUNE)
        return ds

    def get_labels(self):
        mask_paths = [os.path.join(self.annotations_dir, "trimaps", name + ".png") for name in self.images_list]
        ds = tf.data.Dataset.from_tensor_slices(mask_paths)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda img: tf.image.decode_png(img, channels=3), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda img: tf.image.resize(img, [self.height, self.width]), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda img: tf.cast(img, tf.int32))
        ds = ds.map(lambda img: img[:, :, :1])
        ds = ds.map(lambda img: tf.math.subtract(img, tf.constant(1)), num_parallel_calls=AUTOTUNE)
        return ds

    def get_images_list(self):
        images_list = []
        with open(self.path, "r") as f:
            for line in f.readlines():
                name = line.strip().split(" ")[0]
                images_list.append(name)
        return images_list

    def _random_flip(self, image, mask):
        if tf.random.uniform(()) < 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        return image, mask
