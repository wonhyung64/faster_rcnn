#%%
import tensorflow as tf
def generate_anchors(hyper_params): 
    anchor_count = hyper_params['anchor_count']
    feature_map_shape = hyper_params['feature_map_shape']

    stride = 1 / feature_map_shape
    # 여기서 stride가 뭐지 ?? => stride : 보폭

    grid_coords = tf.cast(tf.range(0, feature_map_shape) / feature_map_shape + stride / 2, dtype=tf.float32)
    # print(grid_coords)

    grid_x, grid_y = tf.meshgrid(grid_coords, grid_coords) # tf.meshgrid : 공간상에서 격자를 만드는 함수
    # print(grid_x)

    flat_grid_x, flat_grid_y = tf.reshape(grid_x, (-1, )), tf.reshape(grid_y, (-1, ))
    # print(flat_grid_x)
    # 2차원 구조를 1차원으로 만들기

    grid_map = tf.stack([flat_grid_y, flat_grid_x, flat_grid_y, flat_grid_x], axis=-1)
    # 왜 두개씩 쌓지 ?
    # print(grid_map)

    base_anchors = []
    for scale in hyper_params['anchor_scales']:
        scale /= hyper_params['img_size']
        for ratio in hyper_params['anchor_ratios']:
            w = tf.sqrt(scale **2 / ratio)
            h = w * ratio
            base_anchors.append([-h/2, -w/2, h/2, w/2])
    base_anchors = tf.cast(base_anchors, dtype=tf.float32)        
    # print(base_anchors)

    anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))
    # print(tf.reshape(base_anchors, (1, -1, 4)))
    # print(tf.reshape(grid_map, (1, -1, 4)))
    # print(anchors)

    anchors = tf.reshape(anchors, (-1, 4))
    # print(anchors)
    anchors = tf.clip_by_value(t=anchors, clip_value_min=0, clip_value_max=1) # tf.clip_by_value : min, max값보다 작거나 같은 값을 clip 값으로 대체
    # print(anchors)
    return anchors