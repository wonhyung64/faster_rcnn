#%%
from six import MovedModule
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from utils import bbox_utils, data_utils, hyper_params_utils, rpn_utils


#%%
batch_size = 4

hyper_params = hyper_params_utils.get_hyper_params()

epochs = hyper_params['epochs']

hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])


#%% DATA IMPORT
train_data, dataset_info = tfds.load("voc/2007", split="train+validation", data_dir = "E:\Data\\tensorflow_datasets", with_info=True)
val_data, _ = tfds.load("voc/2007", split="test", data_dir = "E:\Data\\tensorflow_datasets", with_info=True)

train_total_items = dataset_info.splits["train"].num_examples + dataset_info.splits["validation"].num_examples
val_total_items = dataset_info.splits["test"].num_examples

labels = dataset_info.features["labels"].names

hyper_params["total_labels"] = len(labels) + 1



#%% DATA PREPROCESSING
img_size = hyper_params["img_size"]

train_data = train_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, apply_augmentation=True))
val_data = val_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))

data_shapes = ([None, None, None], [None, None], [None,])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))

train_data = train_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values) # batch size = 8 한번에 8개의 사진을 사용
val_data = val_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)


#%% ANCHOR
anchors = bbox_utils.generate_anchors(hyper_params)


#%% Generating Region Proposal
frcnn_train_feed = rpn_utils.faster_rcnn_generator(train_data, anchors, hyper_params)
frcnn_val_feed = rpn_utils.faster_rcnn_generator(val_data, anchors, hyper_params)


#%% RPN Model
base_model = VGG16(include_top=False, input_shape=(img_size, img_size, 3))

feature_extractor = base_model.get_layer("block5_conv3")

output = Conv2D(512,(3, 3), activation='relu', padding='same', name='rpn_conv')(feature_extractor.output)

rpn_cls_output = Conv2D(hyper_params['anchor_count'], (1, 1), activation='sigmoid', name='rpn_cls')(output)

rpn_reg_output = Conv2D(hyper_params['anchor_count'] * 4, (1,1), activation='linear', name='rpn_reg')(output)

rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])

rpn_model.summary()
rpn_model.input

#%%
class RoIBBox(layer):
    
    def __init__(self, anchors, mode, hyper_params, **kwargs):
        super(RoIBBox, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        self.mode = mode
        self.anchors = tf.constant(anchors, dtype=tf.float32)
        
    

#%% Faster R-CNN Model
frcnn_model = fater_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params)

input_img = rpn_model.input

rpn_reg_predictions, rpn_cls_predictions = rpn_model.output
#
roi_bboxes = RoIBBox(anchors, mode='training', hyper_params, name='roi_bboxes')([rpn_reg_predictions, rpn_cls_predictions])

roi_pooled = RoIPooling(hyper_params, name='roi_pooling')([feature_extractor.output, roi_bboxes])