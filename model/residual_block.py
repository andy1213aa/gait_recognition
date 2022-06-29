import tensorflow as tf
from tensorflow.keras import layers, initializers
import tensorflow_addons as tfa

class Residual_Block(layers.Layer):
  def __init__(self, out_shape, strides=1, ksize = 3, shortcut = True):
    super(Residual_Block, self).__init__()
    self.shortcut = shortcut
    
    self.conv1_1 =  layers.Conv2D(filters=out_shape,
                                kernel_size=(3, 3), strides=1, padding='same',
                                name='conv1_1',  use_bias=False)
    self.leakyReLU1_1 = layers.LeakyReLU(name='leakyReLU1_1')
    # self.BN_1 = layers.BatchNormalization()
    self.conv1_2 =  layers.Conv2D(filters=out_shape,
                                kernel_size=(3, 3), strides=1, padding='same',
                                name='conv1_2',  use_bias=False)
    self.leakyReLU1_2 = layers.LeakyReLU(name='leakyReLU1_2')
    # self.BN_2 = layers.BatchNormalization()
    self.MaxPooling2D = layers.MaxPooling2D()

    if shortcut:
 
      self.conv_shortcut = layers.Conv2D(out_shape,kernel_size=1,strides=1, padding='same', use_bias=False)
      # self.BN_shortcut = layers.BatchNormalization()
      self.MaxPooling2D_shortcut = layers.MaxPooling2D()
    
    self.leakyReLU3 = layers.LeakyReLU(name='leakyReLU3')

  def call(self, inputs):

    x = self.conv1_1(inputs)
    # x = self.BN_1(x)
    x = self.leakyReLU1_1(x)

    x = self.conv1_2(x)
    # x = self.BN_2(x)
    x = self.leakyReLU1_2(x)

    x = self.MaxPooling2D(x)     
    
    if self.shortcut:
  
      shortcut = self.conv_shortcut(inputs)
      # shortcut = self.BN_shortcut(shortcut)
      shortcut = self.MaxPooling2D_shortcut(shortcut)
      x = layers.add([x,shortcut])

    outputs = self.leakyReLU3(x)
    return outputs