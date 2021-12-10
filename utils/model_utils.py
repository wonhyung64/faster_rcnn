#%%
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, TimeDistributed, Dense, Flatten, Dropout
#%%
class RPN(Model):
    
    def __init__(self, hyper_params):
        super(RPN, self).__init__()
        self.hyper_params = hyper_params

        self.base_model = VGG16(include_top=False, input_shape=(self.hyper_params["img_size"], 
                                                                self.hyper_params["img_size"],
                                                                3))        

        self.layer = self.base_model.get_layer('block5_conv3').output

        self.feature_extractor = Model(inputs=self.base_model.input, outputs=self.layer)
        self.feature_extractor.trainable = False

        self.conv = Conv2D(filters=512, kernel_size=(3, 3), 
                           activation='relu', padding='same', 
                           name='rpn_conv')

        self.rpn_cls_output = Conv2D(filters=self.hyper_params['anchor_count'], 
                                     kernel_size=(1, 1), 
                                     activation='sigmoid', 
                                     name='rpn_cls')

        self.rpn_reg_output = Conv2D(filters=self.hyper_params['anchor_count']*4, 
                                     kernel_size=(1,1), 
                                     activation='linear', 
                                     name='rpn_reg')

    def call(self,inputs):
        feature_map = self.feature_extractor(inputs) 
        x = self.conv(feature_map)
        cls = self.rpn_cls_output(x)
        reg = self.rpn_reg_output(x)
        return [reg, cls, feature_map]
#%%
class DTN(Model):
    def __init__(self, hyper_params):
        super(DTN, self).__init__()
        self.hyper_params = hyper_params
        #
        self.FC1 = TimeDistributed(Flatten(), name='frcnn_flatten')
        self.FC2 = TimeDistributed(Dense(4096, activation='relu'), name='frcnn_fc1')
        self.FC3 = TimeDistributed(Dropout(0.5), name='frcnn_dropout1')
        self.FC4 = TimeDistributed(Dense(4096, activation='relu'), name='frcnn_fc2')
        self.FC5 = TimeDistributed(Dropout(0.5), name='frcnn_dropout2')
        #
        self.cls = TimeDistributed(Dense(self.hyper_params['total_labels'], 
                                         activation='softmax'), 
                                         name='frcnn_cls')
        self.reg = TimeDistributed(Dense(self.hyper_params['total_labels'] * 4, 
                                         activation='linear'), 
                                         name='frcnn_reg')

    def call(self, inputs):
        fc1 = self.FC1(inputs)
        fc2 = self.FC2(fc1)
        fc3 = self.FC3(fc2)
        fc4 = self.FC4(fc3)
        fc5 = self.FC5(fc4)
        cls = self.cls(fc5)
        reg = self.reg(fc5)
        return [reg, cls]