import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D 

class RPN(Model):
    
    def __init__(self, hyper_params):
        super(RPN, self).__init__()
        self.hyper_params = hyper_params

        self.base_model = VGG16(include_top=False, input_shape=(self.hyper_params["img_size"], self.hyper_params["img_size"], 3))        
        self.layer = self.base_model.get_layer('block5_conv3').output

        self.feature_extractor = Model(inputs=self.base_model.input, outputs=self.layer)
        self.feature_extractor.trainable = False

        self.conv = Conv2D(512,(3, 3), activation='relu', padding='same', name='rpn_conv')
        self.rpn_cls_output = Conv2D(self.hyper_params['anchor_count'], (1, 1), activation='sigmoid', name='rpn_cls')
        self.rpn_reg_output = Conv2D(self.hyper_params['anchor_count']*4, (1,1), activation='linear', name='rpn_reg')

    def call(self,inputs):
        feature_map = self.feature_extractor(inputs) 
        x = self.conv(feature_map)
        cls = self.rpn_cls_output(x)
        reg = self.rpn_reg_output(x)
        return [cls, reg, feature_map]