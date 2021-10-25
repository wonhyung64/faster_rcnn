import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D 

class RPN(Layer):
    def __init__(self, hyper_params, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        
    def get_config(self):
        config = super(RPN, self).get_config()
        config.update({"hyper_params": self.hyper_params})
        return config
        
    def call(self):
        base_model = VGG16(include_top=False, input_shape=(self.hyper_params["img_size"], self.hyper_params["img_size"], 3))
        #
        feature_extractor = base_model.get_layer("block5_conv3")
        feature_extractor.trainable = False
        #
        output = Conv2D(512,(3, 3), activation='relu', padding='same', name='rpn_conv')(feature_extractor.output)
        #
        rpn_cls_output = Conv2D(self.hyper_params['anchor_count'], (1, 1), activation='sigmoid', name='rpn_cls')(output)
        #
        rpn_reg_output = Conv2D(self.hyper_params['anchor_count']*4, (1,1), activation='linear', name='rpn_reg')(output)
        #
        return [rpn_cls_output, rpn_reg_output, feature_extractor]