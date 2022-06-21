import tensorflow as tf
from tensorflow.keras import layers, initializers
import tensorflow_addons as tfa

class Residual_Block(layers.Layer):
  def __init__(self, out_shape, strides=1, ksize = 3, shortcut = False):
    super(Residual_Block, self).__init__()
    self.shortcut = shortcut
    
    self.conv1_1 =  layers.Conv2D(filters=32,
                                kernel_size=(3, 3), strides=1, padding='same',
                                name='conv1_1',  use_bias=False)
    self.leakyReLU1_1 = layers.LeakyReLU(name='leakyReLU1_1')
    self.conv1_2 =  layers.Conv2D(filters=32,
                                kernel_size=(3, 3), strides=1, padding='same',
                                name='conv1_2',  use_bias=False)
    self.leakyReLU1_2 = layers.LeakyReLU(name='leakyReLU1_2')
    
    self.AveragePooling2D = layers.AveragePooling2D()

    if shortcut:
 
        self.conv_shortcut = layers.Conv2D(out_shape,kernel_size=1,strides=1, padding='same', use_bias=False)
        self.AveragePooling2D_shortcut = layers.AveragePooling2D()
    
    self.leakyReLU3 = layers.LeakyReLU(name='leakyReLU3')

  def call(self, inputs):

    x = self.conv1_1(inputs)
    x = self.leakyReLU1_1(x)

    x = self.conv1_2(x)
    x = self.leakyReLU1_2(x)

    x = self.AveragePooling2D(x)     
    
    if self.shortcut:
  
      shortcut = self.conv_shortcut(inputs)
      shortcut = self.AveragePooling2D_shortcut(shortcut)
      x = layers.add([x,shortcut])

    outputs = self.leakyReLU3(x)
    return outputs