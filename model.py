import tensorflow as tf
from tensorflow.keras import models, layers


class DownScalingBlock(layers.Layer):
    def __init__(self, depth, kernel_size=3, **kwargs):
        super(DownScalingBlock, self).__init__(**kwargs)
        self.depth = depth
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.depth,
                                   self.kernel_size,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   activation='relu')
        self.conv2 = layers.Conv2D(self.depth,
                                   self.kernel_size,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   activation='relu')
        self.pool = layers.MaxPool2D()
        super(DownScalingBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x_concat = self.conv2(x)
        x = self.pool(x_concat)
        return x, x_concat

    def get_config(self):
        config = super(DownScalingBlock, self).get_config()
        config.update({
            'depth': self.depth,
            'kernel_size': self.kernel_size
        })
        return config


class UpScalingBlock(layers.Layer):
    def __init__(self, depth, kernel_size=3, **kwargs):
        super(UpScalingBlock, self).__init__(**kwargs)
        self.depth = depth
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.depth,
                                   self.kernel_size,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   activation='relu')
        self.conv2 = layers.Conv2D(self.depth,
                                   self.kernel_size,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   activation='relu')
        self.conv3 = layers.Conv2D(self.depth // 2,
                                   self.kernel_size,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   activation='relu')
        self.upsampling = layers.UpSampling2D()
        super(UpScalingBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.upsampling(x)
        return x

    def get_config(self):
        config = super(UpScalingBlock, self).get_config()
        config.update({
            'depth': self.depth,
            'kernel_size': self.kernel_size
        })
        return config


class Unet(models.Model):
    def __init__(self):
        super(Unet, self).__init__()

    def build(self, input_shape):
        self.downscale1 = DownScalingBlock(64)
        self.downscale2 = DownScalingBlock(128)
        self.downscale3 = DownScalingBlock(256)
        self.upscale1 = UpScalingBlock(512)
        self.upscale2 = UpScalingBlock(256)
        self.upscale3 = UpScalingBlock(128)
        self.conv1 = layers.Conv2D(64, 3,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   activation='relu')
        self.conv2 = layers.Conv2D(64, 3,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   activation='relu')
        self.conv3 = layers.Conv2D(3, 3, 
                                   padding='same', 
                                   kernel_initializer='he_normal')
        self.shape = input_shape
        super(Unet, self).build(input_shape)

    def call(self, inputs):
        x, c1 = self.downscale1(inputs)
        x, c2 = self.downscale2(x)
        x, c3 = self.downscale3(x)
        x = layers.concatenate([self.upscale1(x), c3])
        x = layers.concatenate([self.upscale2(x), c2])
        x = layers.concatenate([self.upscale3(x), c1])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def summary(self):
        inputs = layers.Input(shape=self.shape[1:])
        outputs = self.call(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.summary()
