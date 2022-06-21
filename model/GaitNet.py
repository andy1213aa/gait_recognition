import tensorflow as tf
from tensorflow.keras import layers, initializers
import tensorflow_addons as tfa
from model.residual_block import Residual_Block


class GaitNet(tf.keras.Model):

    # def __init__(self, num_class: int, num_pei_channel: int, view_dim: int):
    def __init__(self):
        super(GaitNet, self).__init__()

        ch = 16

        self.res0 = Residual_Block(ch*2, ksize= 3, shortcut=True)
        self.res1 = Residual_Block(ch*2, ksize= 3, shortcut=True)
        self.res2 = Residual_Block(ch*2, ksize= 3, shortcut=True)
        self.res3 = Residual_Block(ch*2, ksize= 3, shortcut=True)


        self.flatten =  layers.Flatten()
        self.f1 =  layers.Dense(units=128, name='F1')
        self.tanh = tf.keras.activations.tanh
    def call(self, input):

        x = self.res0(input)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.tanh(x)
        return x

    def model(self, inputsize: int) -> tf.keras.models:
        input = tf.keras.Input(
            shape=(inputsize[0], inputsize[1], 3), name='input_layer')

        return tf.keras.models.Model(inputs=input, outputs=self.call(input))
