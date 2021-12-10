#%%
import tensorflow as tf
#%%
def generate_anchors(hyper_params): 
    feature_map_shape = hyper_params['feature_map_shape']

    stride = 1 / feature_map_shape

    grid_coords_ctr = tf.cast(tf.range(0, feature_map_shape) / feature_map_shape + stride / 2, dtype=tf.float32)

    grid_x_ctr, grid_y_ctr = tf.meshgrid(grid_coords_ctr, grid_coords_ctr) # tf.meshgrid : 공간상에서 격자를 만드는 함수

    flat_grid_x_ctr, flat_grid_y_ctr = tf.reshape(grid_x_ctr, (-1, )), tf.reshape(grid_y_ctr, (-1, ))

    grid_map = tf.stack([flat_grid_y_ctr, flat_grid_x_ctr, flat_grid_y_ctr, flat_grid_x_ctr], axis=-1)

    base_anchors = []
    for scale in hyper_params['anchor_scales']:
        scale /= hyper_params['img_size']
        for ratio in hyper_params['anchor_ratios']:
            w = tf.sqrt(scale **2 / ratio)
            h = w * ratio
            base_anchors.append([-h/2, -w/2, h/2, w/2])
    base_anchors = tf.cast(base_anchors, dtype=tf.float32)        

    anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))

    anchors = tf.reshape(anchors, (-1, 4))
    anchors = tf.clip_by_value(t=anchors, clip_value_min=0, clip_value_max=1) # tf.clip_by_value : min, max값보다 작거나 같은 값을 clip 값으로 대체
    return anchors
#%%
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