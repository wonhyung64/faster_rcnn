#%%
# 수정사항
#  - preprocessing 함수 모듈화
#  - generating anchors 함수 모듈화
#  - hyper parameters 불러오는 함수 모듈화
#  - training + test
#  - test 10개만 출력, gt_bbox 표시
#%% Module

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.backend import set_value

from utils import data_utils, bbox_utils, hyper_params_utils, rpn_utils


#%% OPTION 
batch_size = 8

hyper_params = hyper_params_utils.get_hyper_params()

hyper_params["anchor_count"] = len(hyper_params["anchor_ratios"]) * len(hyper_params["anchor_scales"])

epochs = hyper_params["epochs"]
attempt = str(hyper_params["attempt"])
#%% DATA IMPORT
train_data, dataset_info = tfds.load("voc/2007", split="train+validation", data_dir = "C:\won\data\pascal_voc\\tensorflow_datasets", with_info=True)
val_data, _ = tfds.load("voc/2007", split="test", data_dir = "C:\won\data\pascal_voc\\tensorflow_datasets", with_info=True)

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
rpn_train_feed = rpn_utils.rpn_generator(train_data, anchors, hyper_params)
rpn_val_feed = rpn_utils.rpn_generator(val_data, anchors, hyper_params)


#%% RPN MODEL with VGG-16
img_size = hyper_params['img_size']

base_model = VGG16(include_top=False, input_shape=(img_size, img_size, 3))
# include_top ?
# input shape : (img_size) x  (img_size) x 3channel

feature_extractor = base_model.get_layer("block5_conv3")
feature_extractor.output.shape
# VGG16의 마지막 합성곱 층인 "blcok5_conv3" 를 feature_extractor에 할당

output = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="rpn_conv")(feature_extractor.output)
output.shape
# output의 차원이 512 = feature map 하나에 512 차원의 정보를 가짐.

rpn_cls_output = Conv2D(filters=hyper_params["anchor_count"], kernel_size=(1,1), activation="sigmoid", name="rpn_cls")(output)
rpn_cls_output.shape
# rpn model 에서 나오는 classification layer

rpn_reg_output = Conv2D(filters=hyper_params["anchor_count"]*4, kernel_size=(1,1), activation="linear", name="rpn_reg")(output)
rpn_reg_output.shape
# rpn model 에서 나오는 regression layer

rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])
# rpn model 구축

# rpn_model.summary()


#%% Regression Loss Function
def rpn_reg_loss(*args):
    y_true, y_pred = args if len(args) == 2 else args[0]
    #
    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
    # Huber : SmoothL1 loss function

    loss_for_all = loss_fn(y_true[0], y_pred[0])
    loss_for_all = tf.reduce_sum(loss_for_all, axis=-1)
    # sum of SmoothL1
    
    pos_cond = tf.reduce_any(tf.not_equal(y_true[1], tf.constant(1.0)), axis=-1)
    # tf.reduce_any?
    
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    # positive label
    #
    loc_loss = tf.reduce_sum(pos_mask * loss_for_all)

    total_pos_bboxes = tf.maximum(1.0, tf.reduce_sum(pos_mask))

    return loc_loss / total_pos_bboxes


#%% Objectness Loss Function
def rpn_cls_loss(*args):
    y_true, y_pred = args if len(args) == 2 else args[0]

    indices = tf.where(tf.not_equal(y_true[1], tf.constant(-1.0, dtype=tf.float32)))

    target = tf.gather_nd(y_true[1], indices)
    output = tf.gather_nd(y_pred[1], indices)
    # tf.gather_nd ?
    
    lf = tf.losses.BinaryCrossentropy()
    return lf(target, output)


# %% Training
rpn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5),
                  loss=[rpn_reg_loss, rpn_cls_loss])

main_path = "C:/Users/USER/Documents/GitHub/faster_rcnn/rpn"
if not os.path.exists(main_path):
    os.makedirs(main_path)
rpn_model_path = os.path.join(main_path, "{}_{}_model_weights_attempt{}.h5".format("rpn", "vgg16",attempt))

checkpoint_callback = ModelCheckpoint(rpn_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)

step_size_train = math.ceil(train_total_items / batch_size)
step_size_val = math.ceil(val_total_items / batch_size)

#%%
rpn_model.fit(rpn_train_feed,
              steps_per_epoch=step_size_train,
              validation_data=rpn_val_feed,
              validation_steps=step_size_val,
              epochs=epochs,
              callbacks=[checkpoint_callback])


# %% Test
tf.keras.backend.clear_session()

batch_size = 4




# data_dir = 'E:\Data\\tensorflow_datasets'

data_dir = 'C:\won\data\pascal_voc\\tensorflow_datasets'

test_data, dataset_info = tfds.load(name='voc/2007', split='test', data_dir=data_dir, with_info=True)

labels = dataset_info.features['labels'].names
labels = ['bg'] + labels
hyper_params['total_labels'] = len(labels)
img_size = hyper_params['img_size']

data_types = (tf.float32, tf.float32, tf.int32)
data_shapes = ([None, None, None], [None, None], [None,])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))

test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))

test_data = test_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)


#%% Model
base_model = VGG16(include_top=False, input_shape=(img_size, img_size, 3))
feature_extractor = base_model.get_layer('block5_conv3')
output = Conv2D(512, (3, 3), activation='relu', padding='same', name='rpn_conv')(feature_extractor.output)
rpn_cls_output = Conv2D(hyper_params['anchor_count'], (1, 1), activation='sigmoid', name='rpn_cls')(output)
rpn_reg_output = Conv2D(hyper_params['anchor_count'] * 4, (1, 1), activation='linear', name='rpn_reg')(output)
rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])


# %%
# main_path = "E:\Github\\faster_rcnn\\rpn"
main_path = "C:/Users/USER/Documents/GitHub/faster_rcnn/rpn"
model_path = os.path.join(main_path, "{}_{}_model_weights_attempt{}.h5".format('rpn', 'vgg16', attempt))
rpn_model.load_weights(model_path, by_name=True)


#%%
anchors = bbox_utils.generate_anchors(hyper_params)


#%%
result_dir = "C:\won\\rpn_res_att" + attempt + '_with_gt'

os.mkdir(result_dir)
os.chdir(result_dir)

i = 0
for image_data in test_data:
    if i >= 10 : break
    imgs, gt_bbox, gt_labels = image_data
    rpn_bbox_deltas, rpn_labels = rpn_model.predict_on_batch(imgs)
    #
    rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, -1, 4))
    rpn_labels = tf.reshape(rpn_labels, (batch_size, -1))
    #
    rpn_bbox_deltas += hyper_params['variances']
    
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[...,0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height
    #
    all_bbox_width = tf.exp(rpn_bbox_deltas[..., 3]) * all_anc_width
    all_bbox_height = tf.exp(rpn_bbox_deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (rpn_bbox_deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (rpn_bbox_deltas[..., 0] * all_anc_height) + all_anc_ctr_y
    #
    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    
    rpn_bboxes = tf.stack([y1, x1, y2, x2], axis=-1)
    #
    _, top_indices = tf.nn.top_k(rpn_labels, gt_bbox.shape[1] + 2)
    selected_rpn_bboxes = tf.gather(rpn_bboxes, top_indices, batch_dims=1)
    #
    bboxes = tf.reshape(selected_rpn_bboxes[:,0,:], [4,1,4])
    for j in range(gt_bbox.shape[1]):
        bboxes = tf.concat([bboxes, tf.reshape(gt_bbox[:,j,:], [4,1,4])], axis=1)
        bboxes = tf.concat([bboxes, tf.reshape(selected_rpn_bboxes[:,j+1,:], [4,1,4])], axis=1)
    #
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, bboxes, colors)
    plt.figure()
    file_layout = 'test_'
    for img_with_bb in imgs_with_bb :
        filename = file_layout + str(i) + '.png'
        plt.imshow(img_with_bb)
        plt.savefig(filename)
        i += 1


#%%
result_dir = "C:\won\\rpn_res_att" + attempt + '_without_gt'
os.mkdir(result_dir)
os.chdir(result_dir)

i = 0
for image_data in test_data:
    if i >= 10 :  break
    imgs, _, _ = image_data
    rpn_bbox_deltas, rpn_labels = rpn_model.predict_on_batch(imgs)
    #
    rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, -1, 4))
    rpn_labels = tf.reshape(rpn_labels, (batch_size, -1))
    #
    rpn_bbox_deltas += hyper_params['variances']
    
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[...,0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height
    #
    all_bbox_width = tf.exp(rpn_bbox_deltas[..., 3]) * all_anc_width
    all_bbox_height = tf.exp(rpn_bbox_deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (rpn_bbox_deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (rpn_bbox_deltas[..., 0] * all_anc_height) + all_anc_ctr_y
    #
    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    
    rpn_bboxes = tf.stack([y1, x1, y2, x2], axis=-1)
    #
    _, top_indices = tf.nn.top_k(rpn_labels, 20)
    #
    selected_rpn_bboxes = tf.gather(rpn_bboxes, top_indices, batch_dims=1)
    #
    colors = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, selected_rpn_bboxes, colors)
    plt.figure()
    file_layout = 'test_'
    for img_with_bb in imgs_with_bb :
        filename = file_layout + str(i) + '.png'
        plt.imshow(img_with_bb)
        plt.savefig(filename)
        i += 1
