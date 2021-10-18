#%%

import os
import math
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
#
from PIL import Image, ImageDraw
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Layer, Conv2D, Lambda, Input, TimeDistributed, Dense, Flatten, BatchNormalization, Dropout
#
from utils import bbox_utils, data_utils, hyper_params_utils, rpn_utils


#%%
batch_size = 1

hyper_params = hyper_params_utils.get_hyper_params()
data_dir = "C:\won\data\pascal_voc\\tensorflow_datasets"
# data_dir = 'E:\Data\\tensorflow_datasets'

test_data, dataset_info = tfds.load(name='voc/2007', split='test', data_dir=data_dir, with_info=True)

total_items = dataset_info.splits['test'].num_examples

hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])

labels = dataset_info.features["labels"].names

labels = ["bg"] + labels

hyper_params["total_labels"] = len(labels)

img_size = hyper_params["img_size"]

data_types = (tf.float32, tf.float32, tf.int32)
data_shapes = ([None, None, None], [None, None], [None,])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))

test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, data_types, data_shapes))


test_data = test_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)

anchors = bbox_utils.generate_anchors(hyper_params)

frcnn_test_feed = rpn_utils.faster_rcnn_generator(test_data, anchors, hyper_params)
#%%
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

rpn_model.load_weights(r'C:\Users\USER\Documents\GitHub\faster_rcnn\assets\rpn_weights2\weights')

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
#%%
class Decoder(Layer):
    def __init__(self, variances, total_labels, max_total_size=200, score_threshold=0.5, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.variances = variances
        self.total_labels = total_labels
        self.max_total_size = max_total_size
        self.score_threshold = score_threshold
        
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            "variances" : self.variances,
            "total_labels": self.total_labels,
            "max_total_size": self.max_total_size,
            "score_threshold" : self.score_threshold
        })
        return config

    def call(self, inputs):
        roi_bboxes = inputs[0]
        pred_deltas = inputs[1]
        pred_label_probs = inputs[2]
        batch_size = tf.shape(pred_deltas)[0]

        pred_deltas = tf.reshape(pred_deltas, (batch_size, -1, self.total_labels, 4))
        total_labels = hyper_params["total_labels"]#
        pred_deltas = tf.reshape(pred_deltas, (batch_size, -1, total_labels, 4))#
        pred_deltas *= self.variances
        pred_deltas *= hyper_params["variances"]#

        expanded_roi_bboxes = tf.tile(tf.expand_dims(roi_bboxes, -2), (1, 1, self.total_labels, 1))
        expanded_roi_bboxes = tf.tile(tf.expand_dims(roi_bboxes, -2), (1, 1, total_labels, 1))#
        
        all_anc_width = expanded_roi_bboxes[..., 3] - expanded_roi_bboxes[..., 1]
        all_anc_height = expanded_roi_bboxes[..., 2] - expanded_roi_bboxes[..., 0]
        all_anc_ctr_x = expanded_roi_bboxes[..., 1] + 0.5 * all_anc_width
        all_anc_ctr_y = expanded_roi_bboxes[..., 0] + 0.5 * all_anc_height
        
        all_bbox_width = tf.exp(pred_deltas[..., 3]) * all_anc_width
        all_bbox_height = tf.exp(pred_deltas[..., 2]) * all_anc_height
        all_bbox_ctr_x = (pred_deltas[..., 1] * all_anc_width) + all_anc_ctr_x
        all_bbox_ctr_y = (pred_deltas[..., 0] * all_anc_height) + all_anc_ctr_y

        y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
        x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
        y2 = all_bbox_height + y1
        x2 = all_bbox_width + x1
        pred_bboxes = tf.stack([y1, x1, y2, x2], axis=-1)
        pred_label_probs.shape
        tf.argmax(pred_label_probs[0][2])#
        pred_labels_map = tf.expand_dims(tf.argmax(pred_label_probs, -1), -1)
        pred_labels = tf.where(tf.not_equal(pred_labels_map, 0), pred_label_probs, tf.zeros_like(pred_label_probs))
        
        final_bboxes, final_scores, final_labels, _ = tf.image.combined_non_max_suppression(
                            pred_bboxes, pred_labels,
                            max_output_size_per_class = self.max_total_size,
                            max_total_size = self.max_total_size,
                            score_threshold=self.score_threshold
                        )
        return final_bboxes, final_labels, final_scores
#
#%% NMS
NMS = RoIBBox(anchors, hyper_params, name='roi_bboxes')
Delta = RoIDelta(hyper_params, name='roi_deltas')
decode = Decoder(hyper_params["variances"], hyper_params["total_labels"], name="faster_rcnn_decoder")
#%% Faster R-CNN Model
feature_map = Input(shape=(hyper_params['feature_map_shape'], 
                           hyper_params['feature_map_shape'], 512), name='feature_map', dtype=tf.float32)
#
nms_input = Input(shape=(1500, 4), name='nms_input', dtype=tf.float32)
#
roi_pooled = RoIPooling(hyper_params, name='roi_pooling')([feature_map, nms_input])
#
output = TimeDistributed(Flatten(), name='frcnn_flatten')(roi_pooled)
output = TimeDistributed(Dense(4096, activation='relu'), name='frcnn_fc1')(output)
output = TimeDistributed(Dropout(0.5), name='frcnn_dropout1')(output)
output = TimeDistributed(Dense(4096, activation='relu'), name='frcnn_fc2')(output)
output = TimeDistributed(Dropout(0.5), name='frcnn_dropout2')(output)
#
frcnn_cls_pred = TimeDistributed(Dense(hyper_params['total_labels'], activation='softmax'), name='frcnn_cls')(output)
frcnn_reg_pred = TimeDistributed(Dense(84, activation='linear'), name='frcnn_reg')(output)

frcnn_model = Model(inputs=[feature_map, nms_input],
                    outputs=[frcnn_reg_pred, frcnn_cls_pred]
                    )
frcnn_model.load_weights(r'C:\Users\USER\Documents\GitHub\faster_rcnn\assets\frcnn_weights2\weights')
#%%
(img, gt_boxes, gt_labels, bbox_deltas, bbox_labels), () = next(frcnn_test_feed)

rpn_reg_pred, rpn_cls_pred, feature_map = rpn_model.predict(img)

roi_bboxes = NMS([rpn_reg_pred, rpn_cls_pred])

pred_deltas, pred_label_probs = frcnn_model.predict([feature_map, roi_bboxes])

final_bboxes, final_labels, final_scores = decode([roi_bboxes, pred_deltas, pred_label_probs])

img_size = img.shape[1]

for i , image in enumerate(img):

    y1 = final_bboxes[i][...,0] * img_size
    x1 = final_bboxes[i][...,1] * img_size
    y2 = final_bboxes[i][...,2] * img_size
    x2 = final_bboxes[i][...,3] * img_size

    denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)

    image = tf.keras.preprocessing.image.array_to_img(image)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    
    for index, bbox in enumerate(denormalized_box):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)
        width = x2 - x1
        height = y2 - y1
        # if width <= 0 or height <=0:
        #     continue
        final_labels = tf.reshape(final_labels, shape=(200,))
        final_scores = tf.reshape(final_scores, shape=(200,))
        label_index = int(final_labels[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    
    plt.figure()
    plt.imshow(image)
    plt.show()


# test_data[0][1]

# # %%
# for batch_id, image_data in enumerate(test_data):
#     imgs, _, _ = image_data
#     img_size = imgs.shape[1]
#     start = batch_id * batch_size
#     end = start + batch_size
#     batch_bboxes, batch_labels, batch_scores = pred_bboxes[start:end], pred_labels[start:end], pred_scores[start:end]
#     for i, img in enumerate(imgs):
#         y1 = batch_bboxes[i][..., 0] * img_size 
#         x1 = batch_bboxes[i][..., 1] * img_size
#         y2 = batch_bboxes[i][..., 2] * img_size
#         x2 = batch_bboxes[i][..., 3] * img_size
        
#         colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)
#         image = tf.keras.preprocessing.image.array_to_img(img)
#         width, height = image.size
#         draw = ImageDraw.Draw(image)

#         for index, bbox in enumerate(bboxes):
#             y1, x1, y2, x2 = tf.split(denormalized_bboxes, 4)
#             width = x2 - x1
#             height = y2 - y1
#             if width <= 0 or height <= 0:
#                 continue
#             label_index = int(batch_labels[i][index])
#             color = tuple(colors[label_index].numpy())
#             label_text = "{0} {1:0.3f}".format(labels[label_index], batch_scores[index])
#             draw.text((x1 + 4, y1 + 2), label_text, fill=color)
#             draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            
#         plt.figure()
#         plt.imshow(img)
#         plt.show()
        
        

# %%
