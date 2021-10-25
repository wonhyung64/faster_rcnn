#%% MODULE IMPORT

import os
import math
import tensorflow as tf
import tensorflow_datasets as tfds
#
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Layer, Conv2D, Lambda, Input, TimeDistributed, Dense, Flatten, BatchNormalization, Dropout
#
from utils import bbox_utils, data_utils, hyper_params_utils, rpn_utils, model_utils

#%% DATA IMPORT

data_dir = "E:\Data\\tensorflow_datasets"
# data_dir = "C:\won\data\pascal_voc\\tensorflow_datasets"
#
train_data, dataset_info = tfds.load("voc/2007", split="train+validation", data_dir = data_dir, with_info=True)
val_data, _ = tfds.load("voc/2007", split="test", data_dir = data_dir, with_info=True)
#
train_total_items = dataset_info.splits["train"].num_examples + dataset_info.splits["validation"].num_examples
val_total_items = dataset_info.splits["test"].num_examples
#
labels = dataset_info.features["labels"].names
#
#%% HYPER PARAMETERS

hyper_params = hyper_params_utils.get_hyper_params()
#
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])
#

hyper_params["total_labels"] = len(labels) # background label
#
epochs = hyper_params['epochs']
#
batch_size = 4 
#
img_size = hyper_params["img_size"]
#%% DATA PREPROCESSING

train_data_tmp = train_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, apply_augmentation=True))
val_data_tmp = val_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))
#
data_shapes = ([None, None, None], [None, None], [None,])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
#
train_data = train_data_tmp.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values) # batch size = 8 한번에 8개의 사진을 사용
val_data = val_data_tmp.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
#
anchors = bbox_utils.generate_anchors(hyper_params)
#
frcnn_train_feed = rpn_utils.faster_rcnn_generator(train_data, anchors, hyper_params)
#(img, gt_boxes, gt_labels, bbox_deltas, bbox_labels), ()
frcnn_val_feed = rpn_utils.faster_rcnn_generator(val_data, anchors, hyper_params)
#
#%%
class RoIBBox(Layer):
    
    def __init__(self, anchors, hyper_params, **kwargs):
        super(RoIBBox, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        self.anchors = tf.constant(anchors, dtype=tf.float32)

    def get_config(self):
        config = super(RoIBBox, self).get_config()
        config.update({"hyper_params": self.hyper_params, "anchors": self.anchors.numpy()})
        return config

    def call(self, inputs):
        rpn_bbox_deltas = inputs[0]
        rpn_probs = inputs[1]
        anchors = self.anchors
        #
        pre_nms_topn = self.hyper_params["pre_nms_topn"] # pre_nms_topn : 6000
        post_nms_topn = self.hyper_params["train_nms_topn"]
        # train_nms_topn : 1500, test_nms_topn : 300
        nms_iou_threshold = self.hyper_params["nms_iou_threshold"] # nms_iou_threshold : 0.7
        # nms_iou_threshold = tf.constant(nms_iou_threshold, dtype=tf.float32)
        variances = self.hyper_params["variances"]
        total_anchors = anchors.shape[0]
        batch_size = tf.shape(rpn_bbox_deltas)[0]
        rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, total_anchors, 4))
        rpn_probs = tf.reshape(rpn_probs, (batch_size, total_anchors))
        #
        rpn_bbox_deltas *= variances
        #
        all_anc_width = anchors[..., 3] - anchors[..., 1]
        all_anc_height = anchors[..., 2] - anchors[..., 0]
        all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
        all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height

        all_bbox_width = tf.exp(rpn_bbox_deltas[..., 3]) * all_anc_width
        all_bbox_height = tf.exp(rpn_bbox_deltas[..., 2]) * all_anc_height
        all_bbox_ctr_x = (rpn_bbox_deltas[..., 1] * all_anc_width) + all_anc_ctr_x
        all_bbox_ctr_y = (rpn_bbox_deltas[..., 0] * all_anc_height) + all_anc_ctr_y

        y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
        x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
        y2 = all_bbox_height + y1
        x2 = all_bbox_width + x1
        
        rpn_bboxes = tf.stack([y1, x1, y2, x2], axis=-1)
        #
        _, pre_indices = tf.nn.top_k(rpn_probs, pre_nms_topn)
        #
        pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
        pre_roi_probs = tf.gather(rpn_probs, pre_indices, batch_dims=1)
        #
        pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
        pre_roi_probs = tf.reshape(pre_roi_probs, (batch_size, pre_nms_topn, 1))
        #
        roi_bboxes, _, _, _ = tf.image.combined_non_max_suppression(pre_roi_bboxes, pre_roi_probs,
                                                             max_output_size_per_class=post_nms_topn,
                                                             max_total_size = post_nms_topn,
                                                             iou_threshold=nms_iou_threshold)
        #
        return roi_bboxes # rpn과 classification 을 따로 학습 시키기 위해
#    
#%%
class RoIPooling(Layer):
    #
    def __init__(self, hyper_params, **kwargs):
        super(RoIPooling, self).__init__(**kwargs)
        self.hyper_params = hyper_params
    #
    def get_config(self):
        config = super(RoIPooling, self).get_config()
        config.update({"hyper_params": self.hyper_params})
        return config
    #
    def call(self, inputs):
        feature_map = inputs[0]
        roi_bboxes = inputs[1]
        pooling_size = self.hyper_params["pooling_size"]
        batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
        #
        row_size = batch_size * total_bboxes
        #
        pooling_bbox_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes))
        pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1, ))
        pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))
        # (roi_1500 * batch_size_4, bbox_coordinate_4)
        #

        pooling_feature_map = tf.image.crop_and_resize(
            feature_map,
            pooling_bboxes,
            pooling_bbox_indices,
            pooling_size
        ) # (roi_1500 * batch_size_4, pooling_size, pooling_size, feature_map_channel_512)
        final_pooling_feature_map = tf.reshape(pooling_feature_map, (batch_size,
                                                                     total_bboxes,
                                                                     pooling_feature_map.shape[1],
                                                                     pooling_feature_map.shape[2],
                                                                     pooling_feature_map.shape[3]))
        #
        return final_pooling_feature_map
#
#%%
class RoIDelta(Layer):
    def __init__(self, hyper_params, **kwargs):
        super(RoIDelta, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        
    def get_config(self):
        config = super(RoIDelta, self).get_config()
        config.update({"hyper_params": self.hyper_params})
        return config
    
    def call(self, inputs):
        roi_bboxes = inputs[0]
        gt_boxes = inputs[1]
        gt_labels = inputs[2]

        total_labels = self.hyper_params["total_labels"]
        total_pos_bboxes = self.hyper_params["total_pos_bboxes"]
        total_neg_bboxes = self.hyper_params["total_neg_bboxes"]
        variances = self.hyper_params["variances"]
        batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
        #
        bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(roi_bboxes, 4, axis=-1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)

        gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x2), axis=-1)
        bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)

        x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1]))
        y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
        x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
        y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))

        intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)

        union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)

        iou_map = intersection_area / union_area
        #
        max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
        #
        merged_iou_map = tf.reduce_max(iou_map, axis=2)
        #
        pos_mask = tf.greater(merged_iou_map, 0.5)
        pos_mask = rpn_utils.randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
        #
        neg_mask = tf.logical_and(tf.less(merged_iou_map, 0.5), tf.greater(merged_iou_map, 0.1))
        neg_mask = rpn_utils.randomly_select_xyz_mask(neg_mask, tf.constant([total_neg_bboxes], dtype=tf.int32))
        #
        gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
        expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, axis=-1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
        #
        gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
        pos_gt_labels = tf.where(pos_mask, gt_labels_map, tf.constant(-1, dtype=tf.int32))
        neg_gt_labels = tf.cast(neg_mask, dtype=tf.int32)
        expanded_gt_labels = pos_gt_labels + neg_gt_labels # IoU 가 ~0.1 은 -1, 0.1~0.5 는 0, 0.5~ 는 1 이상의 클래스
        #
        bbox_width = roi_bboxes[..., 3] - roi_bboxes[..., 1]
        bbox_height = roi_bboxes[..., 2] - roi_bboxes[..., 0]
        bbox_ctr_x = roi_bboxes[..., 1] + 0.5 * bbox_width
        bbox_ctr_y = roi_bboxes[..., 0] + 0.5 * bbox_height

        gt_width = expanded_gt_boxes[..., 3] - expanded_gt_boxes[..., 1]
        gt_height = expanded_gt_boxes[..., 2] - expanded_gt_boxes[..., 0]
        gt_ctr_x = expanded_gt_boxes[..., 1] + 0.5 * gt_width
        gt_ctr_y = expanded_gt_boxes[..., 0] + 0.5 * gt_height
        
        bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
        bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
        delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
        delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
        delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_width))
        delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))
        
        roi_bbox_deltas = tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)
        #
        roi_bbox_labels = tf.one_hot(expanded_gt_labels, total_labels) # 21개 클래스로 인코딩
        scatter_indices = tf.tile(tf.expand_dims(roi_bbox_labels, -1), (1, 1, 1, 4))
        roi_bbox_deltas = scatter_indices * tf.expand_dims(roi_bbox_deltas, -2)
        # roi_bbox_deltas = tf.reshape(roi_bbox_deltas, (batch_size, total_bboxes * total_labels, 4))
        # 
        # return tf.stop_gradient(roi_bbox_deltas), tf.stop_gradient(roi_bbox_labels)
        return roi_bbox_deltas, roi_bbox_labels
#
#%%
#%%
def region_reg_loss(pred, bbox_deltas, bbox_labels):
    #
    pred = tf.reshape(pred, (batch_size, hyper_params['feature_map_shape'],
                                         hyper_params['feature_map_shape'],
                                         hyper_params['anchor_count'], 4))
    bbox_deltas = tf.reshape(bbox_deltas, (batch_size, hyper_params['feature_map_shape'],
                                                       hyper_params['feature_map_shape'],
                                                       hyper_params['anchor_count'], 4))

    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
    # Huber : SmoothL1 loss function
    loss_for_all = loss_fn(bbox_deltas, pred)
    
    pos_cond = tf.equal(bbox_labels, tf.constant(1.0))
    # tf.reduce_any?
    
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    # positive label
    #
    loc_loss = tf.reduce_sum(pos_mask * loss_for_all)

    total_pos_bboxes = tf.reduce_sum(pos_mask)

    return loc_loss / total_pos_bboxes
#%%
def dtn_reg_loss(pred, frcnn_reg_actuals, frcnn_cls_actuals):

    pred = tf.reshape(pred, (batch_size, hyper_params['train_nms_topn'],
                                  hyper_params['total_labels'],4))
    #
    frcnn_reg_actuals = tf.reshape(frcnn_reg_actuals, (batch_size, hyper_params['train_nms_topn'],
                                  hyper_params['total_labels'],4))
    #
    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
    # Huber : SmoothL1 loss function
    loss_for_all = loss_fn(frcnn_reg_actuals, pred)
    
    pos_cond = tf.equal(frcnn_cls_actuals, tf.constant(1.0))
    
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    
    # tf.reduce_any?
    loc_loss = tf.reduce_sum(pos_mask * loss_for_all) 
    # positive label
    total_pos_bboxes = tf.reduce_sum(pos_mask)
    #
    return loc_loss / total_pos_bboxes 
#%%
def region_cls_loss(pred, bbox_labels):

    indices = tf.where(tf.not_equal(bbox_labels, tf.constant(-1.0, dtype = tf.float32)))
    
    target = tf.gather_nd(bbox_labels, indices)
    output = tf.gather_nd(pred, indices)

    lf = tf.losses.BinaryCrossentropy()
    return lf(target, output)
#%%
def dtn_cls_loss(pred, true):
    # y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, 4))

    loss_fn = tf.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
    
    loss_for_all = loss_fn(true, pred)
    
    cond = tf.reduce_any(tf.not_equal(true, tf.constant(0.0)), axis=-1)
    mask = tf.cast(cond, dtype=tf.float32)
    
    conf_loss = tf.reduce_sum(mask * loss_for_all)
    total_boxes = tf.maximum(1.0, tf.reduce_sum(mask))
    
    return conf_loss / total_boxes
#%% RPN Model
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

#%% Faster R-CNN Model
class Recog(Model):
    def __init__(self, hyper_params):
        super(Recog, self).__init__()
        self.hyper_params = hyper_params
        self.roi_pooled = RoIPooling(self.hyper_params, name='roi_pooling')
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
        roi_pooled = self.roi_pooled(inputs)
        fc1 = self.FC1(roi_pooled)
        fc2 = self.FC2(fc1)
        fc3 = self.FC3(fc2)
        fc4 = self.FC4(fc3)
        fc5 = self.FC5(fc4)
        cls = self.cls(fc5)
        reg = self.reg(fc5)
        return [reg, cls]

#%%
rpn_model = RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)

NMS = RoIBBox(anchors, hyper_params, name='roi_bboxes')
Delta = RoIDelta(hyper_params, name='roi_deltas')

frcnn_model = Recog(hyper_params)
input_shape = [(None, hyper_params['feature_map_shape'], 
                hyper_params['feature_map_shape'], 512), 
               (None, hyper_params['train_nms_topn'], 4)]
frcnn_model.build(input_shape)

#%%
optimizer1 = keras.optimizers.Adam(learning_rate=1e-5)
optimizer2 = keras.optimizers.Adam(learning_rate=1e-5)
#%%
@tf.function
def train_step1(img, bbox_deltas, bbox_labels):
    with tf.GradientTape(persistent=True) as tape:
        '''RPN'''
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
        
        rpn_reg_loss = region_reg_loss(rpn_reg_output, bbox_deltas, bbox_labels)
        rpn_cls_loss = region_cls_loss(rpn_cls_output, bbox_labels)
        rpn_loss = rpn_reg_loss + rpn_cls_loss
        
    grads_rpn = tape.gradient(rpn_loss, rpn_model.trainable_weights)

    optimizer1.apply_gradients(zip(grads_rpn, rpn_model.trainable_weights))

    return rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map

#%%
@tf.function
def train_step2(nms_output, roi_delta):
    with tf.GradientTape(persistent=True) as tape:
        '''Recognition'''
        frcnn_pred = frcnn_model([feature_map, tf.stop_gradient(nms_output)], training=True)
        
        frcnn_reg_loss = dtn_reg_loss(frcnn_pred[0], roi_delta[0], roi_delta[1])
        frcnn_cls_loss = dtn_cls_loss(frcnn_pred[1], roi_delta[1])
        frcnn_loss = frcnn_reg_loss + frcnn_cls_loss

    grads_frcnn = tape.gradient(frcnn_loss, frcnn_model.trainable_weights)
    optimizer2.apply_gradients(zip(grads_frcnn, frcnn_model.trainable_weights))

    return frcnn_reg_loss, frcnn_cls_loss
#%%
for epoch in range(epochs):
    tf.print("\nStart of epoch %d" % (epoch + 1,))
    
    for step, ((img, gt_boxes, gt_labels, bbox_deltas, bbox_labels), ()) in enumerate(frcnn_train_feed):
        try:
            rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map = train_step1(img, bbox_deltas, bbox_labels)
            nms_output = NMS([rpn_reg_output, rpn_cls_output])
            roi_delta = Delta([nms_output, gt_boxes, gt_labels])
            frcnn_reg_loss, frcnn_cls_loss = train_step2(nms_output, roi_delta)

            
            if step % 10 == 0:
                tf.print(
                    "Training loss (for one batch) at step %d: rpn_reg - %.4f, rpn_cls - %.4f, rpn - %.4f, frcnn_reg - %.4f, frcnn_cls - %.4f, frcnn - %.4f, loss - %.4f"
                    % (step, float(rpn_reg_loss), float(rpn_cls_loss), float(rpn_reg_loss + rpn_cls_loss), float(frcnn_reg_loss), float(frcnn_cls_loss), float(frcnn_reg_loss + frcnn_cls_loss), float(rpn_reg_loss + rpn_cls_loss + frcnn_reg_loss + frcnn_cls_loss)))
                
                tf.print("Seen so far: %d samples" % ((step + 1) * batch_size))
        except: break
#%%
rpn_model.save_weights(r'C:\Users\USER\Documents\GitHub\faster_rcnn\assets\rpn_weights2\weights')
frcnn_model.save_weights(r'C:\Users\USER\Documents\GitHub\faster_rcnn\assets\frcnn_weights2\weights')