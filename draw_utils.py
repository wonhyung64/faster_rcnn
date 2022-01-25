#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import bbox_utils

from PIL import ImageDraw
#%%
def draw_nms_output(image, roi_bboxes, roi_scores, top_n, save_dir=None, save_num=None):

    image = tf.squeeze(image, axis=0)
    image = tf.keras.preprocessing.image.array_to_img(image)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    y1 = roi_bboxes[0][...,0] * height
    x1 = roi_bboxes[0][...,1] * width
    y2 = roi_bboxes[0][...,2] * height
    x2 = roi_bboxes[0][...,3] * width

    denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

    _, top_indices = tf.nn.top_k(roi_scores[0], top_n)
    #
    selected_rpn_bboxes = tf.gather(denormalized_box, top_indices, batch_dims=0)
    #
    selected_rpn_bboxes = tf.squeeze(selected_rpn_bboxes, axis=0)
    
    for bbox in selected_rpn_bboxes:
        y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)
        draw.rectangle((x1, y1, x2, y2), outline=234, width=3)

    if save_dir != None:    
        plt.figure()
        plt.imshow(image)
        plt.savefig(save_dir + r'\res_nms\\' + str(save_num) + '.png')

    else:
        plt.figure()
        plt.imshow(image)
        plt.show()
    
#%%
def draw_final_bbox_output(image, pred_deltas, roi_bboxes, hyper_params, save_dir=None, save_num=None):
    tmp = tf.reshape(pred_deltas[0], shape=(1, pred_deltas.shape[1], hyper_params['total_labels'], 4))
    tmp *= hyper_params['variances']

    expanded_roi_bboxes = tf.reshape(tf.tile(tf.expand_dims(roi_bboxes[0], -2), (1, hyper_params['total_labels'], 1)), shape=(1,hyper_params['test_nms_topn'], hyper_params['total_labels'], 4))
    final_bboxes_ = bbox_utils.delta_to_bbox(expanded_roi_bboxes, tmp)
    
    image = tf.squeeze(image, axis=0)
    image = tf.keras.preprocessing.image.array_to_img(image)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    y1 = final_bboxes_[...,0] * height
    x1 = final_bboxes_[...,1] * width
    y2 = final_bboxes_[...,2] * height
    x2 = final_bboxes_[...,3] * width

    denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

    colors = [57, 140, 234]

    for j in range(3):
        color = colors[j]
        tmp_box = denormalized_box[0][j]
        bbox = tmp_box[0]
        y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        
    if save_dir != None:    
        plt.figure()
        plt.imshow(image)
        plt.savefig(save_dir + r'\res_final_bbox\\' + str(save_num) + '.png')

    else:
        plt.figure()
        plt.imshow(image)
        plt.show()
#%%
def draw_frcnn_output(image, final_bboxes, labels, final_labels, final_scores, save_dir=None, save_num=None):

    image = tf.squeeze(image, axis=0)
    image = tf.keras.preprocessing.image.array_to_img(image)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    y1 = final_bboxes[0][...,0] * height
    x1 = final_bboxes[0][...,1] * width
    y2 = final_bboxes[0][...,2] * height
    x2 = final_bboxes[0][...,3] * width

    denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)

    
    for index, bbox in enumerate(denormalized_box):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)
        width = x2 - x1
        height = y2 - y1

        final_labels_ = tf.reshape(final_labels[0], shape=(200,))
        final_scores_ = tf.reshape(final_scores[0], shape=(200,))
        label_index = int(final_labels_[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores_[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    
    if save_dir != None:    
        plt.figure()
        plt.imshow(image)
        plt.savefig(save_dir + r'\res_frcnn\\' + str(save_num) + '.png')

    else:
        plt.figure()
        plt.imshow(image)
        plt.show()