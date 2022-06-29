import tensorflow as tf
from tensorflow.keras import layers, initializers
import tensorflow_addons as tfa
from model.residual_block import Residual_Block
from tensorflow.keras.utils import plot_model

class GaitNet(tf.keras.Model):

    # def __init__(self, num_class: int, num_pei_channel: int, view_dim: int):
    def __init__(self, ch):
        super(GaitNet, self).__init__()

       

        self.res0 = Residual_Block(ch*2, ksize= 3, shortcut=True)
        self.res1 = Residual_Block(ch*4, ksize= 3, shortcut=True)
        self.res2 = Residual_Block(ch*4, ksize= 3, shortcut=True)
        self.res3 = Residual_Block(ch*8, ksize= 3, shortcut=True)
        self.res4 = Residual_Block(ch*8, ksize= 3, shortcut=True)


        self.flatten =  layers.Flatten() 
        self.f1 =  layers.Dense(units=256, name='F1')
        self.L2_normalize = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        # self.tanh = tf.keras.activations.tanh
    def call(self, input):

        x = self.res0(input)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.L2_normalize(x)
        return x

    def model(self, inputsize: int) -> tf.keras.models:
        input = tf.keras.Input(
            shape=(inputsize[0], inputsize[1], inputsize[2]), name='input_layer')
        model = tf.keras.models.Model(inputs=input, outputs=self.call(input))
        plot_model(model, to_file='model.png', show_shapes=True)
        return model
