#%% Module
import tensorflow as tf

#%% OPTION 
batch_size = 8
epochs = 50
load_weights = False

hyper_params = {"img_size": 500,
                "feature_map_shape": 31,
                "anchor_ratios": [1., 2., 1./2.],
                "anchor_scales": [128, 256, 512],
                "pre_nms_topn": 6000,
                "train_nms_topn": 1500,
                "test_nms_topn": 300,
                "nms_iou_threshold": 0.7,
                "total_pos_bboxes": 128,
                "total_neg_bboxes": 128,
                "pooling_size": (7,7),
                "variances": [0.1, 0.1, 0.2, 0.2],
                }

hyper_params["anchor_count"] = len(hyper_params["anchor_ratios"]) * len(hyper_params["anchor_scales"])

#%% DATA IMPORT
import tensorflow_datasets as tfds

train_data, dataset_info = tfds.load("voc/2007", split="train+validation", data_dir = "~tensorflow_datasets", with_info=True)
val_data, _ = tfds.load("voc/2007", split="test", data_dir = "~tensorflow_datasets", with_info=True)
# under bar 로 데이터를 불러오지 않기
train_total_items = dataset_info.splits["train"].num_examples + dataset_info.splits["validation"].num_examples
val_total_items = dataset_info.splits["test"].num_examples

#%% EDA
import matplotlib.pyplot as plt
import numpy as np

dataset_info

labels = dataset_info.features["labels"].names
labels

for data in train_data.take(1):
    image, label = data['image'], data['labels']
    print(image)
    print('Image size :', image.shape)
    plt.imshow(image.numpy())
    plt.axis('off')
    print('Label size :', label.shape)
    print("Label: %s, %s" % (labels[label.numpy()[0]], labels[label.numpy()[1]]))

for data in val_data.take(1):
    image, label = data['image'], data['labels']
    print('Image size :', image.shape)
    plt.imshow(image.numpy())
    plt.axis('off')
    print('Label size :', label.shape)
    print("Label: %s, %s" % (labels[label.numpy()[0]], labels[label.numpy()[1]]))

hyper_params["total_labels"] = len(labels) + 1

#%% DATA PREPROCESSING DEF
def preprocessing(image_data, final_height, final_width, apply_augmentation=False, evaluate=False):
    img = image_data['image']
    gt_boxes = image_data['objects']['bbox']
    gt_labels = tf.cast(image_data['objects']['label'] + 1, tf.int32)
    
    if evaluate:
        not_diff = tf.logical_not(image_data['objects']['is_difficult'])
        gt_boxes = gt_boxes[not_diff]
        gt_labels = gt_labels[not_diff]
    
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    
    if apply_augmentation:
        img, gt_boxes = randomly_apply_operation(flip_horizontally, img, gt_boxes)
    
    return img, gt_boxes, gt_labels


def flip_horizontally(img, gt_boxes):
    flipped_img = tf.image.flip_left_right(img) # 이미지 좌우반전 함수
    flipped_gt_boxes = tf.stack([gt_boxes[..., 0],
                                  1.0 - gt_boxes[..., 3],
                                  gt_boxes[...,2],
                                  1.0 - gt_boxes[..., 1]], -1)
    return flipped_img, flipped_gt_boxes


def randomly_apply_operation(operation, img, gt_boxes):
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gt_boxes),
        lambda: (img, gt_boxes)
    )
    

def get_random_bool():
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)

#%% DATA PREPROCESSING
img_size = hyper_params["img_size"]

train_data = train_data.map(lambda x : preprocessing(x, img_size, img_size, apply_augmentation=True))
val_data = val_data.map(lambda x : preprocessing(x, img_size, img_size))

#%%
for data in train_data.take(1):
    image, label = data[0], data[2]
    print(image)
    print('Image size :', image.shape)
    plt.imshow(image.numpy())
    plt.axis('off')
    print('Label size :', label.shape)
    print("Label: %s, %s" % (labels[label.numpy()[0]-1], labels[label.numpy()[1]-1]))

for data in val_data.take(1):
    image, label = data[0], data[2]
    print('Image size :', image.shape)
    plt.imshow(image.numpy())
    plt.axis('off')
    print('Label size :', label.shape)
    print("Label: %s, %s" % (labels[label.numpy()[0]-1], labels[label.numpy()[1]-1]))

# %%
data_shapes = ([None, None, None], [None, None], [None,])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
train_data = train_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
val_data = val_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)

#%% ANCHOR
anchor_count = hyper_params['anchor_count']
feature_map_shape = hyper_params['feature_map_shape']

stride = 1 / feature_map_shape
# 여기서 stride가 뭐지 ?? => stride : 보폭

grid_coords = tf.cast(tf.range(0, feature_map_shape) / feature_map_shape + stride / 2, dtype=tf.float32)
print(grid_coords)

grid_x, grid_y = tf.meshgrid(grid_coords, grid_coords) # tf.meshgrid : 공간상에서 격자를 만드는 함수
print(grid_x)

flat_grid_x, flat_grid_y = tf.reshape(grid_x, (-1, )), tf.reshape(grid_y, (-1, ))
print(flat_grid_x)
# 2차원 구조를 1차원으로 만들기

grid_map = tf.stack([flat_grid_y, flat_grid_x, flat_grid_y, flat_grid_x], axis=-1)
# 왜 두개씩 쌓지 ?
print(grid_map)

base_anchors = []
for scale in hyper_params['anchor_scales']:
    scale /= hyper_params['img_size']
    for ratio in hyper_params['anchor_ratios']:
        w = tf.sqrt(scale **2 / ratio)
        h = w * ratio
        base_anchors.append([-h/2, -w/2, h/2, w/2])
base_anchors = tf.cast(base_anchors, dtype=tf.float32)        
print(base_anchors)

anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))
print(tf.reshape(base_anchors, (1, -1, 4)))
print(tf.reshape(grid_map, (1, -1, 4)))
print(anchors)

anchors = tf.reshape(anchors, (-1, 4))
print(anchors)
anchors = tf.clip_by_value(t=anchors, clip_value_min=0, clip_value_max=1) # tf.clip_by_value : min, max값보다 작거나 같은 값을 clip 값으로 대체
print(anchors)

#%% Generating Residual Proposal DEF
def rpn_generator(dataset, anchors, hyper_params):
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            bbox_deltas, bbox_labels = calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params)
            yield img, (bbox_deltas, bbox_labels)

def calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params):
    batch_size = tf.shape(gt_boxes)
    feature_map_shape = hyper_params['feature_map_shape']
    anchor_count = hyper_params['anchor_count']
    total_pos_bboxes = hyper_params['total_pos_bboxes']
    total_neg_bboxes = hyper_params['total_neg_bboxes']
    variances = hyper_params['variances']
    

    # Generating IoU map
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)
    
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1) # tf.squeeze : 텐서에서 사이즈 차원이 1이 아닌 부분만 짜낸다.
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis = -1)
    
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, perm=[0, 2, 1]))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))
    
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)
    
    iou_map = intersection_area / union_area
    #
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    #
    merged_iou_map = tf.reduce_max(iou_map, axis=2) # tf.reduce_max 의 의미 정확하지 않음..
    #
    pos_mask = tf.greater(merged_iou_map, 0.7)
    #
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    #
    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    max_pos_mask = tf.scatter_nd(scatter_bbox_indices, tf.fill((tf.shape(valid_indices)[0], ), True), tf.shape(pos_mask))
    
    
    
    

#%% RPN MODEL with VGG-16
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential

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

# feature_extractor 를 왜 return ?

#%%
