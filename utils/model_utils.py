import tensorflow as tf

from tensorflow.keras import Model, VGG16, Conv2D, Layer


class RPN(Layer):
    def __init__(self, hyper_params, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        
    def get_config(self):
        config = super(RPN, self).get_config()
        config.update({"hyper_params": self.hyper_params})
        return config
        
    def call(self, inputs):
        base_model = VGG16(include_top=False, input_shape=(img_size, img_size, 3))
        #
        feature_extractor = base_model.get_layer("block5_conv3")
        feature_extractor.trainable = False
        #
        output = Conv2D(512,(3, 3), activation='relu', padding='same', name='rpn_conv')(feature_extractor.output)
        #
        rpn_cls_output = Conv2D(hyper_params['anchor_count'], (1, 1), activation='sigmoid', name='rpn_cls')(output)
        #
        rpn_reg_output = Conv2D(hyper_params['anchor_count']*4, (1,1), activation='linear', name='rpn_reg')(output)
        #
        rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output, feature_extractor.output])