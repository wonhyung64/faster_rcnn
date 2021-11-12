import numpy as np
import tensorflow as tf
#%%
def preprocessing(data, batch_size, final_height, final_width, evaluate):
    img_ = data[:,0]
    gt_boxes_ = data[:,1]
    gt_labels_ = data[:,2]
    is_difficult_ = data[:,3]

    for i in range(batch_size):
        
        if evaluate:
            not_diff = np.logical_not(is_difficult_[i])
            gt_boxes_[i] = gt_boxes_[i][not_diff]
            gt_labels_[i] = gt_labels_[i][not_diff]

        gt_boxes_[i] = tf.cast(gt_boxes_[i], tf.float32)
        gt_labels_[i] = tf.cast(gt_labels_[i], tf.int32)

        img_[i] = tf.image.convert_image_dtype(img_[i], tf.float32)
        img_[i] = tf.image.resize(img_[i], (final_height, final_width))

        img_[i] = tf.reshape(img_[i], shape=(1, final_height, final_width, 3))
        gt_boxes_[i] = tf.reshape(gt_boxes_[i], shape=(1, gt_boxes_[i].shape[0], 4))
        gt_labels_[i] = tf.reshape(gt_labels_[i], shape=(1, gt_labels_[i].shape[0]))

    max_label_num = max([gt_labels_[i].shape[1] for i in range(batch_size)])

    for i in range(batch_size):
        gt_boxes_[i] = tf.concat([gt_boxes_[i], tf.constant(0, dtype=tf.float32, shape=(1, max_label_num - gt_boxes_[i].shape[1], 4))], axis = -2)
        gt_labels_[i] = tf.concat([gt_labels_[i], tf.constant(0, dtype=tf.int32, shape=(1, max_label_num - gt_labels_[i].shape[1]))], axis=-1)
    
    img = tf.concat([img_[i] for i in range(batch_size)], axis=0)
    gt_boxes = tf.concat([gt_boxes_[i] for i in range(batch_size)], axis=0)
    gt_labels = tf.concat([gt_labels_[i] for i in range(batch_size)], axis=0)

    return img, gt_boxes, gt_labels

#%%
def faster_rcnn_generator(batch_data, anchors, hyper_params, train=True):
    chk_pos_num = []
    evaluate = False
    if train == False: 
        evaluate = True
    img, gt_boxes, gt_labels = preprocessing(batch_data, hyper_params["batch_size"], hyper_params["img_size"], hyper_params["img_size"], evaluate=evaluate) 
    bbox_deltas, bbox_labels, chk_pos_num = calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params, chk_pos_num)

    return img, gt_boxes, gt_labels, bbox_deltas, bbox_labels, chk_pos_num

#%%

def calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params, chk_pos_num):
    batch_size = hyper_params['batch_size'] # gt_boxes 는 [batch_size, 이미지 데이터 한장에 있는 라벨의 갯수(4개) , 좌표(4)]
    feature_map_shape = hyper_params['feature_map_shape'] # feature_map_shape = 31
    anchor_count = hyper_params['anchor_count'] # anchor_count = 3 * 3 
    total_pos_bboxes = hyper_params['total_pos_bboxes'] 
    total_neg_bboxes = hyper_params['total_neg_bboxes']
    variances = hyper_params['variances'] # variances 가 무슨값인지 알 수 없음
    pos_threshold = hyper_params["pos_threshold"]
    neg_threshold = hyper_params["neg_threshold"]

    # Generating IoU map
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1) # C X C  X anchor_count 개의 reference anchors 의 x, y 좌표
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1) # gt_boxes에 있는 박스들 각각의 x, y 좌표
    
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1) # tf.squeeze : 텐서에서 사이즈 차원이 1이 아닌 부분만 짜낸다.
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis = -1)
    
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1])) # tf.transpose : 텐서를 [] 순서의 모양으로 transpose
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))
    
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)
    
    iou_map = intersection_area / union_area 
    # 8장의 사진에 대한, C X C X 9 개의 reference anchors 와, ground truth box n개(최대4) 와의 IoU계산
    #
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # 각 사진의 reference anchor 에서 IoU가 가장 높게 나오는 gt_box의 index 추출 (8, 8649, 1)

    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    # 각 사진의 gt_box에서 IoU가 가장 높게 나오는 reference anchor 의 index 추출 (8, 1, 4)
    #
    merged_iou_map = tf.reduce_max(iou_map, axis=2) 
    # 8장의 사진 에서 하나의 reference anchor와 4개의 gt_boxes 들 중 가장 높은값만 남기기
    #
    pos_mask = tf.greater(merged_iou_map, pos_threshold)
    chk_pos_num = np.size(np.where(pos_mask[0] == True))
    # 각 사진에서 가장 높은 IoU가 threshold 보다 높은지 
    #
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    # 왜 -1? : gt_labels 중 라벨이 없는 값은 -1 로 입력
    
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    # valid_indices 에서 label이 있는 부분의 tensor_index 반환
    
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    # 8장의 사진에 15개의 라벨이 있고 이들과의 IoU가 가장높은 사진에서의 reference anchor index 반환
    #

    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    # 8장의 사진 에서 라벨이 존재하는 사진의 index와 해당 라벨의 gt_box와 가장 높은 IoU를 가지는 reference anchor의 index 반환 
    max_pos_mask = tf.scatter_nd(indices=scatter_bbox_indices, updates=tf.fill((tf.shape(valid_indices)[0], ), True), shape=tf.shape(pos_mask))
    # 8장의 사진 각각의 reference anchors에서 gt_box와의 가장 높은 IoU 를 가지는 reference anchor의 index 만 True
    pos_mask = tf.logical_or(pos_mask, max_pos_mask)
    # pos_mask에 threshold 이상의 IoU를 가지는 reference anchor 와, 가장 높은 IoU를 가지는 reference anchor 만 True 반환
    pos_mask = randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
    #
    pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
    # 8장의 pos_mask 에 대해서 각각 true의 개수 반환
    neg_count = (total_pos_bboxes + total_neg_bboxes) - pos_count
    
    neg_mask = tf.logical_and(tf.less(merged_iou_map, neg_threshold), tf.logical_not(pos_mask))
    # 8장의 사진에서 Iou가 0.3보다 작고 pos_mask가 false인 부분만 False
    neg_mask = randomly_select_xyz_mask(neg_mask, neg_count)
    #
    
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    # ?
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    # 정리 자료에서 cls loss의 p*_i
    gt_boxes_map = tf.gather(params=gt_boxes, indices=max_indices_each_row, batch_dims=1)
    # 8장의 사진에서 8649개의 reference anchor 중 4개의 gt_box와 IoU가 가장 높은 IoU값 4개의 gt_box에 대해 각각 반환

    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    # 정리 자료에서 reg loss의 p*_i
    
    bbox_width = anchors[..., 3] - anchors[..., 1]
    bbox_height = anchors[..., 2] - anchors[...,0]
    bbox_ctr_x = anchors[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = anchors[..., 0] + 0.5 * bbox_height
    
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
    
    bbox_deltas = tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1) / variances
    bbox_deltas = tf.reshape(bbox_deltas, (batch_size, feature_map_shape, feature_map_shape, anchor_count* 4))
    bbox_labels = tf.reshape(bbox_labels, (batch_size, feature_map_shape, feature_map_shape, anchor_count))
    
    return bbox_deltas, bbox_labels, chk_pos_num


def randomly_select_xyz_mask(mask, select_xyz):
    maxval = tf.reduce_max(select_xyz) * 10
    random_mask = tf.random.uniform(tf.shape(mask), minval=1, maxval=maxval, dtype=tf.int32)
    multiplied_mask = tf.cast(mask, tf.int32) * random_mask
    sorted_mask = tf.argsort(multiplied_mask, direction="DESCENDING")
    sorted_mask_indices = tf.argsort(sorted_mask)
    selected_mask = tf.less(sorted_mask_indices, tf.expand_dims(select_xyz, 1))
    return tf.logical_and(mask, selected_mask)
    
