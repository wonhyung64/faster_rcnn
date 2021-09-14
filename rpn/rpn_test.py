#%%
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.python.keras.backend import set_value
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential

from utils import data_utils, bbox_utils, hyper_params_utils


# %%
batch_size = 4

hyper_params = hyper_params_utils.get_hyper_params()

hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])

attempt = str(hyper_params["attempt"])


# %%
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

anchors = bbox_utils.generate_anchors(hyper_params)


#%%
result_dir = "C:\won\\rpn_result_attempt" + attempt
os.mkdir(result_dir)

i = 0
for image_data in test_data:
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
    _, top_indices = tf.nn.top_k(rpn_labels, 10)
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
# %%
