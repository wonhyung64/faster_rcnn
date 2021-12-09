#%% MODULE IMPORT
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#
from tqdm import tqdm
from PIL import ImageDraw
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, TimeDistributed, Dense, Flatten, Dropout
#
from utils import bbox_utils, rpn_utils, loss_utils
#%% HYPER PARAMETERS
hyper_params = {"img_size": 500,
                "feature_map_shape": 31,
                "anchor_ratios": [1., 2., 1./2.],
                "anchor_scales": [128, 256, 512],
                "pre_nms_topn": 6000,
                "train_nms_topn": 10,
                "test_nms_topn": 16,
                "nms_iou_threshold": 0.9,
                "total_pos_bboxes": 128,
                "total_neg_bboxes": 128,
                "pooling_size": (7,7),
                "variances": [0.1, 0.1, 0.2, 0.2],
                "iters" : 20000,
                "pos_threshold" : 0.6,
                "neg_threshold" : 0.25,
                "batch_size" : 1,
                "background" : False,
                "dtn_with_binary" : True,
                "nms_by_class" : False,
                "atmp" : 11
                }
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])
#
iters = hyper_params['iters']
batch_size = hyper_params['batch_size']
img_size = hyper_params["img_size"]
background = hyper_params["background"]
dtn_with_binary = hyper_params["dtn_with_binary"]
nms_by_class = hyper_params["nms_by_class"]
#%%
info_dir = r"C:\won\data\pascal_voc\voc2007_np"
info = np.load(info_dir + r"\info.npy", allow_pickle=True)

labels = info[0]["labels"]
train_filename = info[0]['train_filename'] + info[0]['val_filename']
test_filename = info[0]['test_filename']

train_total_items = len(train_filename)
test_total_items = len(test_filename)

labels = ["bg"] + labels

hyper_params["total_labels"] = len(labels)
#%%
anchors = bbox_utils.generate_anchors(hyper_params)
#%%
class RoIBBox(Layer):
    
    def __init__(self, anchors, hyper_params, test=False, **kwargs):
        super(RoIBBox, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        self.anchors = tf.constant(anchors, dtype=tf.float32)
        self.test = test

    def get_config(self):
        config = super(RoIBBox, self).get_config()
        config.update({"hyper_params": self.hyper_params, "anchors": self.anchors.numpy()})
        return config

    def call(self, inputs):
        rpn_bbox_deltas = inputs[0]
        rpn_probs = inputs[1]
        gt_labels = inputs[2]
        anchors = self.anchors
        #
        pre_nms_topn = self.hyper_params["pre_nms_topn"] # pre_nms_topn : 6000
        post_nms_topn = self.hyper_params["train_nms_topn"]
        if self.test == True: post_nms_topn = self.hyper_params["test_nms_topn"]
        # train_nms_topn : 1500, test_nms_topn : 300
        nms_iou_threshold = self.hyper_params["nms_iou_threshold"] # nms_iou_threshold : 0.7
        # nms_iou_threshold = tf.constant(nms_iou_threshold, dtype=tf.float32)
        variances = self.hyper_params["variances"]
        # non_nms = self.hyper_params["non_nms"]
        total_anchors = anchors.shape[0]
        batch_size = tf.shape(rpn_bbox_deltas)[0]
        rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, total_anchors, 4))
        rpn_probs = tf.reshape(rpn_probs, (batch_size, total_anchors))
        #
        rpn_bbox_deltas *= variances
        #
        rpn_bboxes = rpn_utils.delta_to_bbox(anchors, rpn_bbox_deltas)

        if self.hyper_params["nms_by_class"] == True:
            iou_map = rpn_utils.generate_iou(anchors, rpn_bboxes)
            #
            max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
            # 1500개의 roi_bbox 와의 iou가 가장 큰 gtbox 인덱스
            merged_iou_map = tf.reduce_max(iou_map, axis=2)
            
            rpn_class = tf.where(merged_iou_map != 0, tf.gather(gt_labels, max_indices_each_gt_box, axis=1, batch_dims=1), tf.constant(0, dtype=tf.int32))

            rpn_probs[0][rpn_class[0]==8]
            rpn_probs[1][rpn_class[1]==16]
            rpn_probs[2][rpn_class[2]==19]
            rpn_probs[3][rpn_class[3]==15]
            rpn_probs[3][rpn_class[3]==20]
            merged_iou_map[3]
            gt_labels
            tf.unique(gt_labels[0])
            gt_boxes[2]
        #
        else:
            _, pre_indices = tf.nn.top_k(rpn_probs, pre_nms_topn)
            #
            pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
            pre_roi_probs = tf.gather(rpn_probs, pre_indices, batch_dims=1)
            #
            pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
            pre_roi_probs = tf.reshape(pre_roi_probs, (batch_size, pre_nms_topn, 1))
            #
            # roi_bboxs
            
            roi_bboxes, roi_scores, _, _ = tf.image.combined_non_max_suppression(pre_roi_bboxes, pre_roi_probs,
                                                                max_output_size_per_class=post_nms_topn,
                                                                max_total_size = post_nms_topn,
                                                                iou_threshold=nms_iou_threshold)
        #
        return roi_bboxes, roi_scores

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
        if hyper_params["dtn_with_binary"] == True:
            self.cls = TimeDistributed(Dense(self.hyper_params['total_labels'],
                                             activation='sigmoid'),
                                       name='frcnn_cls')

    def call(self, inputs):
        fc1 = self.FC2(inputs)
        fc2 = self.FC3(fc1)
        fc3 = self.FC4(fc2)
        fc4 = self.FC5(fc3)
        cls = self.cls(fc4)
        reg = self.reg(fc4)
        return [reg, cls]

#%%
rpn_model = RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)

NMS = RoIBBox(anchors, hyper_params, test=False, name='roi_bboxes')
pooling = RoIPooling(hyper_params)

res_dir = r'C:\won\frcnn\atmp'
atmp = hyper_params["atmp"]
rpn_model.load_weights(res_dir + str(atmp) + r'\rpn_weights\weights')

#%%
train_dir = r"C:\won\data\pascal_voc\voc2007_np\train_val\\"
#%%
i = atmp
roi_dir = r'C:\won\data\pascal_voc\voc_2007_roi_atmp'

tmp = True
while tmp :
    if os.path.isdir(roi_dir + str(i)) : 
        i+= 1
    else: 
        os.makedirs(roi_dir + str(i))
        print("Generated roi" + str(i))
        tmp = False

roi_dir = roi_dir + str(i) 

#%%
progress_bar = tqdm(range(train_total_items))
for attempt in progress_bar:
    extract_dic = dict()
    
    res_filename = [train_filename[i] for i in range(attempt*batch_size, attempt*batch_size + batch_size)]
    batch_data = np.array([np.load(train_dir + train_filename[i] + ".npy", allow_pickle=True) for i in range(attempt*batch_size, attempt*batch_size+batch_size)])

    img, gt_boxes, gt_labels = rpn_utils.preprocessing(batch_data, hyper_params["batch_size"], hyper_params["img_size"], hyper_params["img_size"], evaluate=True)
    
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model.predict(img)
    roi_bboxes, roi_scores = NMS([rpn_reg_output, rpn_cls_output, gt_labels])
    pooled_roi = pooling([feature_map, roi_bboxes])
    extract_dic["pooled_roi"] = pooled_roi.numpy()
    extract_dic["gt_boxes"] = gt_boxes.numpy()
    extract_dic["gt_labels"] = gt_labels.numpy()
    extract = np.array([extract_dic], dtype=object)

    iou_map = rpn_utils.generate_iou(roi_bboxes, gt_boxes)
    #
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # 1500개의 roi_bbox 와의 iou가 가장 큰 gtbox 인덱스
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    
    rpn_class = tf.where(merged_iou_map >= 0.5, tf.gather(gt_labels, max_indices_each_gt_box, axis=1, batch_dims=1), tf.constant(0, dtype=tf.int32))
    rpn_class = rpn_class.numpy()

    save_dir = roi_dir + r"\\" + res_filename[0]
    os.makedirs(save_dir)

    np.save((save_dir + "\\" + res_filename[0] + "_label.npy"), rpn_class, allow_pickle=True)
    np.save((save_dir + "\\" + res_filename[0] + "_feature.npy"), extract, allow_pickle=True)

    for i in range(10):
        roi_bboxes_ = roi_bboxes * img_size
        tmp = tf.round(roi_bboxes_[0][i])
        img_ = img[0][int(tmp[0]):int(tmp[2]), int(tmp[1]):int(tmp[3])]
        img_ = img_.numpy()
        np.save((save_dir + "\\" + res_filename[0] + "_" + str(i) + '.npy'), img_, allow_pickle=True)
        
        # img_ = tf.keras.preprocessing.image.array_to_img(img_)
        # plt.imshow(img_)
        # plt.savefig(save_dir + "\\" + res_filename[0] + "_" + str(i) + '.png')
# %%

# pooled_roi_ = pooled_roi.numpy()
# a = np.load()
# a = np.load(roi_dir + "\\" + res_filename[0] + ".npy", allow_pickle=True)

# rois = a[0]["pooled_roi"]
# rois = tf.convert_to_tensor(rois)
# roi = rois[0][0]

# classification = VGG16(include_top=True, input_shape=(hyper_params["img_size"], 
#                                                                 hyper_params["img_size"],
#                                                                 3))        
# classification.summay()
# classification = VGG16()
# classification.build(input_shape=(None, 500, 500, 3))
# classification.summary()

# model_input = classification.get_layer("flatten")
# model_output = classification.output
# model = tf.keras.Model(inputs=model_input, outputs=model_output)

# from keras.applications.vgg16 import decode_predictions

