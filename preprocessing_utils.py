#%%
import numpy as np
import tensorflow as tf
#%%
def preprocessing(img, gt_boxes, gt_labels):# resize, flip_left_right
    gt_labels = tf.cast(gt_labels + 1, tf.int32)
    if (np.random.uniform(0,1,1) > 0.5) == True:
        img, gt_boxes = flip_horizontal(img, gt_boxes)
    return img, gt_boxes, gt_labels
    
#%%
def flip_horizontal(img, gt_boxes):
    img = tf.image.flip_left_right(img)
    gt_boxes = tf.stack([gt_boxes[...,0],
                        1.0 - gt_boxes[...,3],
                        gt_boxes[...,2],
                        1.0 - gt_boxes[...,1]], -1)

    return img, gt_boxes
    
    