#%%
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, TimeDistributed, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
#%%
def generate_anchors(): 
    feature_map_shape = 31
    stride = 1 / feature_map_shape
    grid_coords_ctr = tf.cast(tf.range(0, feature_map_shape) / feature_map_shape + stride / 2, dtype=tf.float32)
    grid_x_ctr, grid_y_ctr = tf.meshgrid(grid_coords_ctr, grid_coords_ctr) # tf.meshgrid : 공간상에서 격자를 만드는 함수
    flat_grid_x_ctr, flat_grid_y_ctr = tf.reshape(grid_x_ctr, (-1, )), tf.reshape(grid_y_ctr, (-1, ))
    grid_map = tf.stack([flat_grid_y_ctr, flat_grid_x_ctr, flat_grid_y_ctr, flat_grid_x_ctr], axis=-1)

    base_anchors = []
    for scale in [128, 256, 512]:
        scale /= 500
        for ratio in [1., 2., 1./2.]:
            w = tf.sqrt(scale **2 / ratio)
            h = w * ratio
            base_anchors.append([-h/2, -w/2, h/2, w/2])
    base_anchors = tf.cast(base_anchors, dtype=tf.float32)        
    anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))
    anchors = tf.reshape(anchors, (-1, 4))
    anchors = tf.clip_by_value(t=anchors, clip_value_min=0, clip_value_max=1) # tf.clip_by_value : min, max값보다 작거나 같은 값을 clip 값으로 대체
    return anchors

def bbox_to_delta(anchors, gt_boxes):
    bbox_width = anchors[..., 3] - anchors[..., 1]
    bbox_height = anchors[..., 2] - anchors[...,0]
    bbox_ctr_x = anchors[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = anchors[..., 0] + 0.5 * bbox_height
    
    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height
    
    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))
    
    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)

def generate_iou(anchors, gt_boxes):
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1) 
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1) 
    
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1) 
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis = -1)
    
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1])) 
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))
    
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)
    
    return intersection_area / union_area 

def delta_to_bbox(anchors, bbox_deltas):
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[..., 0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height

    all_bbox_width = tf.exp(bbox_deltas[..., 3]) * all_anc_width
    all_bbox_height = tf.exp(bbox_deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (bbox_deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (bbox_deltas[..., 0] * all_anc_height) + all_anc_ctr_y

    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    
    return tf.stack([y1, x1, y2, x2], axis=-1)

def NMS(anchors, rpn_reg_output, rpn_cls_output):
    rpn_reg_output = tf.reshape(rpn_reg_output, (4, 31*31*9, 4))
    rpn_cls_output = tf.reshape(rpn_cls_output, (4, 31*31*9))
    rpn_bboxes = delta_to_bbox(anchors, rpn_reg_output)
    _, pre_indices = tf.nn.top_k(rpn_cls_output, 6000)
    pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
    pre_roi_probs = tf.gather(rpn_cls_output, pre_indices, batch_dims=1)
    pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (4, 6000, 1, 4))
    pre_roi_probs = tf.reshape(pre_roi_probs, (4, 6000, 1))
    roi_bboxes, _, _, _ = tf.image.combined_non_max_suppression(pre_roi_bboxes, pre_roi_probs,
                                                        max_output_size_per_class=1500,
                                                        max_total_size=1500,
                                                        iou_threshold=0.7)
    return roi_bboxes


def RoIPooling(feature_map, roi_bboxes):
    pooling_size = (7,7)
    batch_size=4
    total_bboxes = 1500
    pooling_bbox_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes))
    pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1,))
    pooling_bboxes = tf.reshape(roi_bboxes, (batch_size*total_bboxes, 4))
    pooled_roi = tf.image.crop_and_resize(feature_map,
                                                    pooling_bboxes,
                                                    pooling_bbox_indices,
                                                    pooling_size) 
    pooled_roi = tf.reshape(pooled_roi, (batch_size, total_bboxes,
                                                    pooled_roi.shape[1],
                                                    pooled_roi.shape[2],
                                                    pooled_roi.shape[3]))
    return pooled_roi
#%% data import and preprocessing

train, dataset_info = tfds.load("voc/2007", split="train", with_info=True)

labels = dataset_info.features["labels"].names
labels = ["bg"] + labels

def preprocessing(image_data):# resize, flip_left_right
    img = image_data["image"]
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (500, 500)) # 서로 다른 크기의 이미지를 500 X 500 사이즈로 맞추기
    if (np.random.uniform(0,1,1) > 0.5) == True:
        img = tf.image.flip_left_right(img)
        gt_boxes = tf.stack([gt_boxes[...,0],
                            1.0 - gt_boxes[...,3],
                            gt_boxes[...,2],
                            1.0 - gt_boxes[...,1]], -1)
    return img, gt_boxes, gt_labels

train = train.map(lambda x: preprocessing(x))
data_shapes = ([None, None, None], [None, None], [None]) # image, gt_boxs, gt_labels
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
train = train.padded_batch(4, padded_shapes=data_shapes, padding_values=padding_values)
# 한 image가 갖고있는 gt_box의 갯수가 다르기 때문에 배치로 묶어주기 위해 padding_values로 채우고 4개의 배치로 묶어주기
train = iter(train)
anchors = generate_anchors()

# %% bounding box target
def rpn_target(anchors, gt_boxes, gt_labels):
    iou_map = generate_iou(anchors, gt_boxes)
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    merged_iou_map = tf.reduce_max(iou_map, axis=2) 

    pos_mask = tf.greater(merged_iou_map, 0.7)
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    max_pos_mask = tf.scatter_nd(indices=scatter_bbox_indices, updates=tf.fill((tf.shape(valid_indices)[0], ), True), shape=tf.shape(pos_mask))
    pos_mask = tf.logical_or(pos_mask, max_pos_mask)
    pos_mask = randomly_select_mask(pos_mask, tf.constant([128], dtype=tf.int32))
    pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
    neg_count = (128 + 128) - pos_count
    neg_mask = tf.logical_and(tf.less(merged_iou_map, 0.3), tf.logical_not(pos_mask))
    neg_mask = randomly_select_mask(neg_mask, neg_count)
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)

    gt_boxes_map = tf.gather(params=gt_boxes, indices=max_indices_each_row, batch_dims=1)
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    bbox_deltas = bbox_to_delta(anchors, expanded_gt_boxes)
    bbox_deltas = tf.reshape(bbox_deltas, (4, 31, 31, 9*4))
    bbox_labels = tf.reshape(bbox_labels, (4, 31, 31, 9))

    return bbox_deltas, bbox_labels

def randomly_select_mask(mask, num):
    maxval = tf.reduce_max(num) * 10
    random_mask = tf.random.uniform(tf.shape(mask), minval=1, maxval=maxval, dtype=tf.int32)
    multiplied_mask = tf.cast(mask, tf.int32) * random_mask
    sorted_mask = tf.argsort(multiplied_mask, direction="DESCENDING")
    sorted_mask_indices = tf.argsort(sorted_mask)
    selected_mask = tf.less(sorted_mask_indices, tf.expand_dims(num, 1))
    return tf.logical_and(mask, selected_mask)
    
class RPN(Model):
    def __init__(self):
        super(RPN, self).__init__()
        self.base_model = VGG19(include_top=False, input_shape=(500, 500, 3))        
        self.layer = self.base_model.get_layer('block5_conv3').output
        self.feature_extractor = Model(inputs=self.base_model.input, outputs=self.layer)
        self.feature_extractor.trainable = False
        self.conv = Conv2D(filters=512, kernel_size=(3, 3), 
                           activation='relu', padding='same', 
                           name='rpn_conv')
        self.rpn_cls_output = Conv2D(filters=9, 
                                     kernel_size=(1, 1), 
                                     activation='sigmoid', 
                                     name='rpn_cls')
        self.rpn_reg_output = Conv2D(filters=9*4, 
                                     kernel_size=(1,1), 
                                     activation='linear', 
                                     name='rpn_reg')
    @tf.function
    def call(self,inputs):
        feature_map = self.feature_extractor(inputs) 
        x = self.conv(feature_map)
        cls = self.rpn_cls_output(x)
        reg = self.rpn_reg_output(x)
        return [reg, cls, feature_map]

rpn_model = RPN()
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)

def rpn_reg_loss_fn(pred, bbox_deltas, bbox_labels):
    pred = tf.reshape(pred, (4, 31, 31, 9, 4))
    bbox_deltas = tf.reshape(bbox_deltas, (4, 31, 31, 9, 4))
    total_anchors_loc = 31 * 31
    tune_param = tf.constant(total_anchors_loc / (128 + 128), tf.float32)
    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
    loss_for_all = loss_fn(bbox_deltas, pred)
    pos_cond = tf.equal(bbox_labels, tf.constant(1.0))
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    loc_loss = tf.reduce_sum(pos_mask * loss_for_all)
    return loc_loss * tune_param / total_anchors_loc

def rpn_cls_loss_fn(pred, bbox_labels):
    indices = tf.where(tf.not_equal(bbox_labels, tf.constant(-1.0, dtype = tf.float32)))
    target = tf.gather_nd(bbox_labels, indices)
    output = tf.gather_nd(pred, indices)
    lf =  -tf.reduce_sum(target * tf.math.log(tf.clip_by_value(output, 1e-9, 1e+9)) + (1-target) * tf.math.log(1 - tf.clip_by_value(output, 1e-9, 1e+9)))
    return lf
    
optimizer_rpn = keras.optimizers.SGD(learning_rate=1e-5)

@tf.function
def train_step1(img, bbox_deltas, bbox_labels):
    with tf.GradientTape(persistent=True) as tape:
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
        rpn_reg_loss = rpn_reg_loss_fn(rpn_reg_output, bbox_deltas, bbox_labels)
        rpn_cls_loss = rpn_cls_loss_fn(rpn_cls_output, bbox_labels)
        rpn_loss = rpn_reg_loss + rpn_cls_loss
    grads_rpn = tape.gradient(rpn_loss, rpn_model.trainable_weights)
    optimizer_rpn.apply_gradients(zip(grads_rpn, rpn_model.trainable_weights))
    return rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map
    
#%%
def frcnn_target(roi_bboxes, gt_boxes, gt_labels):
    iou_map = generate_iou(roi_bboxes, gt_boxes)
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    merged_iou_map = tf.reduce_max(iou_map, axis=2)

    pos_mask = tf.greater(merged_iou_map, 0.5)
    pos_mask = randomly_select_mask(pos_mask, tf.constant([128], dtype=tf.int32))
    neg_mask = tf.logical_and(tf.less(merged_iou_map, 0.5), tf.greater(merged_iou_map, 0.1))
    neg_mask = randomly_select_mask(neg_mask, tf.constant([128], dtype=tf.int32))

    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, axis=-1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
    pos_gt_labels = tf.where(pos_mask, gt_labels_map, tf.constant(-1, dtype=tf.int32))
    neg_gt_labels = tf.cast(neg_mask, dtype=tf.int32)
    expanded_gt_labels = pos_gt_labels + neg_gt_labels 

    roi_bbox_deltas = bbox_to_delta(roi_bboxes, expanded_gt_boxes)
    roi_bbox_labels = tf.one_hot(expanded_gt_labels, 21)
    scatter_indices = tf.tile(tf.expand_dims(roi_bbox_labels, -1), (1, 1, 1, 4))
    roi_bbox_deltas = scatter_indices * tf.expand_dims(roi_bbox_deltas, -2)

    return roi_bbox_deltas, roi_bbox_labels

class FRCNN(Model):
    def __init__(self):
        super(FRCNN, self).__init__()
        self.FC1 = TimeDistributed(Flatten(), name='frcnn_flatten')
        self.FC2 = TimeDistributed(Dense(4096, activation='relu'), name='frcnn_fc1')
        self.FC3 = TimeDistributed(Dropout(0.5), name='frcnn_dropout1')
        self.FC4 = TimeDistributed(Dense(4096, activation='relu'), name='frcnn_fc2')
        self.FC5 = TimeDistributed(Dropout(0.5), name='frcnn_dropout2')
        self.cls = TimeDistributed(Dense(21, 
                                         activation='softmax'), 
                                         name='frcnn_cls')
        self.reg = TimeDistributed(Dense(21*4, 
                                         activation='linear'), 
                                         name='frcnn_reg')
    @tf.function
    def call(self, inputs):
        fc1 = self.FC1(inputs)
        fc2 = self.FC2(fc1)
        fc3 = self.FC3(fc2)
        fc4 = self.FC4(fc3)
        fc5 = self.FC5(fc4)
        cls = self.cls(fc5)
        reg = self.reg(fc5)
        return [reg, cls]

frcnn_model = FRCNN()
input_shape = (None, 1500, 7, 7, 512)
frcnn_model.build(input_shape)

def frcnn_reg_loss_fn(pred, frcnn_reg_actuals, frcnn_cls_actuals):
    pred = tf.reshape(pred, (4, 1500, 21, 4))
    frcnn_reg_actuals = tf.reshape(frcnn_reg_actuals, (4, 1500, 21,4))
    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
    loss_for_all = loss_fn(frcnn_reg_actuals, pred)
    pos_cond = tf.equal(frcnn_cls_actuals, tf.constant(1.0))
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    loc_loss = tf.reduce_sum(pos_mask * loss_for_all) 
    total_pos_bboxes = tf.reduce_sum(pos_mask)
    return loc_loss / total_pos_bboxes * tf.constant(0.5, tf.float32)

def frcnn_cls_loss_fn(pred, true):
    loss_for_all = -tf.math.reduce_sum(true * tf.math.log(pred + 1e-7), axis=-1)
    cond = tf.reduce_any(tf.not_equal(true, tf.constant(0.0)), axis=-1)
    mask = tf.cast(cond, dtype=tf.float32)
    conf_loss = tf.reduce_mean(mask * loss_for_all)
    total_boxes = tf.maximum(1.0, tf.reduce_sum(mask))
    return conf_loss / total_boxes

optimizer_frcnn = keras.optimizers.SGD(learning_rate=1e-5)

@tf.function
def train_step2(pooled_roi, roi_bbox_deltas, roi_bbox_labels):
    with tf.GradientTape(persistent=True) as tape:
        frcnn_reg_output, frcnn_cls_output = frcnn_model(pooled_roi, training=True)
        frcnn_reg_loss = frcnn_reg_loss_fn(frcnn_reg_output, roi_bbox_deltas, roi_bbox_labels)
        frcnn_cls_loss = frcnn_cls_loss_fn(frcnn_cls_output, roi_bbox_labels)
        frcnn_loss = frcnn_reg_loss + frcnn_cls_loss
    grads_frcnn = tape.gradient(frcnn_loss, frcnn_model.trainable_weights)
    optimizer_frcnn.apply_gradients(zip(grads_frcnn, frcnn_model.trainable_weights))
    return frcnn_reg_loss, frcnn_cls_loss

#%%
iterations = 100
for iteration in range(iterations):
    img, gt_boxes, gt_labels = next(train) 
    bbox_deltas, bbox_labels= rpn_target(anchors, gt_boxes, gt_labels)
    rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map = train_step1(img, bbox_deltas, bbox_labels)
    roi_bboxes = NMS(anchors, rpn_reg_output, rpn_cls_output)
    pooled_roi = RoIPooling(feature_map, roi_bboxes)
    roi_bbox_deltas, roi_bbox_labels = frcnn_target(roi_bboxes, gt_boxes, gt_labels)
    frcnn_reg_loss, frcnn_cls_loss = train_step2(pooled_roi, roi_bbox_deltas, roi_bbox_labels)
    
    print('iteration {}/{} | rpn_reg {:.4f}, rpn_cls {:.4f}, frcnn_reg {:.4f}, frcnn_cls {:.4f}'.format(
        iteration, iterations, 
        float(rpn_reg_loss), float(rpn_cls_loss), float(frcnn_reg_loss), float(frcnn_cls_loss)
    ))
