#%% 
import os
import time
import tensorflow as tf
import numpy as np

from PIL import Image
from tqdm import tqdm

import utils, model_utils, preprocessing_utils, postprocessing_utils, anchor_utils, test_utils

from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
#%% 
hyper_params = utils.get_hyper_params()
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])

hyper_params["batch_size"] = batch_size = 1
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]

if dataset_name == "ship":
    import ship
    dataset, labels = ship.fetch_dataset(dataset_name, "test", img_size)
    dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing_ship(x, y, z, w))
else:
    import data_utils
    dataset, labels = data_utils.fetch_dataset(dataset_name, "test", img_size)
    dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))

data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
dataset = dataset.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
dataset = iter(dataset)
# next(dataset)
labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)

anchors = anchor_utils.generate_anchors(hyper_params)

#%%
weights_dir = os.getcwd() + "/frcnn_atmp"
weights_dir = weights_dir + "/" + os.listdir(weights_dir)[0]

rpn_model = model_utils.RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)
rpn_model.load_weights(weights_dir + '/rpn_weights/weights')

dtn_model = model_utils.DTN(hyper_params)
input_shape = (None, hyper_params['train_nms_topn'], 7, 7, 512)
dtn_model.build(input_shape)
dtn_model.load_weights(weights_dir + '/dtn_weights/weights')

#%%
file_num_light = [17, 67, 73, 104]
file_num_dark = [169, 186, 217, 219]
try_num = -1

colors_violet = tf.constant([[222, 1, 189, 59]])
colors_pink = tf.constant([[214,70, 119, 170]])
colors_skyblue = tf.constant([[54, 177, 235, 190]])
colors_kaki = tf.constant([[101, 158, 42, 42]])
colors = tf.concat([colors_skyblue, colors_pink, colors_violet, colors_kaki], axis=0)
colos_white = tf.constant([193, 181, 198, 136])

# colors= tf.random.uniform((4, 4), maxval=256, dtype=tf.int32)
fig = plt.figure(figsize=(30,30))
fig.tight_layout()
rows, cols = 2, 2
i = 1

while True:
    image, gt_boxes, gt_labels = next(dataset)
    if try_num in file_num_light: 
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(image)
        roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
        pooled_roi = postprocessing_utils.RoIAlign(roi_bboxes, feature_map, hyper_params)
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
        final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params)

        # indices = tf.where(final_bboxes != tf.constant([0.]))
        # tf.reshape(tf.gather_nd(final_bboxes, indices), (final_bboxes.shape)
        # final_labels

        image = tf.image.resize(image, (2160,3840), method="bicubic")
        image = tf.squeeze(image, axis=0)
        image = tf.keras.preprocessing.image.array_to_img(image)
        width, height = image.size
        draw = ImageDraw.Draw(image)

        y1 = gt_boxes[0][...,0] * height
        x1 = gt_boxes[0][...,1] * width
        y2 = gt_boxes[0][...,2] * height
        x2 = gt_boxes[0][...,3] * width

        gt_boxes = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

        y1 = final_bboxes[0][...,0] * height
        x1 = final_bboxes[0][...,1] * width
        y2 = final_bboxes[0][...,2] * height
        x2 = final_bboxes[0][...,3] * width

        denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

        gt_color = tf.constant([75,142,80,72], dtype=tf.int32)
        fnt = ImageFont.truetype(r"C:\Users\USER\AppData\Local\Microsoft\Windows\Fonts\FiraCode-Regular.ttf", 50)

        # for index, bbox in enumerate(gt_boxes):
        #     color = tf.constant([193, 181, 198, 136])
        #     y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)

        #     gt_labels_ = tf.reshape(gt_labels[0], shape=(gt_labels.shape[1]))
        #     label_index = int(gt_labels_[index])
        #     color = tuple(gt_color.numpy())
        #     label_text = labels[label_index]
        #     draw.text((x1 + 4, y1 - 60), label_text, fill=tuple(tf.constant([54, 177, 235, 190]).numpy()), font=fnt)
        #     draw.rectangle((x1, y1, x2, y2), outline=tuple(tf.constant([54, 177, 235, 190]).numpy()), width=5)


        for index, bbox in enumerate(denormalized_box):
            y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)

            final_labels_ = tf.reshape(final_labels[0], shape=(200,))
            final_scores_ = tf.reshape(final_scores[0], shape=(200,))
            label_index = int(final_labels_[index])
            color = tuple(colors[label_index].numpy())
            label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores_[index])
            draw.text((x1 + 4, y1 - 60), label_text, fill=color, font=fnt)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=5)

        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(image)
        ax.set_xticks([]), ax.set_yticks([])

        i += 1
    try_num += 1
plt.show()
# %%

# %%
