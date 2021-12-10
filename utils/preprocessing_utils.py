#%%
import numpy as np
import tensorflow as tf
#%%
def preprocessing(data, batch_size, final_height, final_width, evaluate, augmentation=False):
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
        gt_labels_[i] = tf.cast(gt_labels_[i] + 1, tf.int32)

        img_[i] = tf.image.convert_image_dtype(img_[i], tf.float32)
        img_[i] = tf.image.resize(img_[i], (final_height, final_width))

        img_[i] = tf.reshape(img_[i], shape=(1, final_height, final_width, 3))
        gt_boxes_[i] = tf.reshape(gt_boxes_[i], shape=(1, gt_boxes_[i].shape[0], 4))
        gt_labels_[i] = tf.reshape(gt_labels_[i], shape=(1, gt_labels_[i].shape[0]))

        if np.logical_and(augmentation == True, np.random.uniform(0,1,1) > 0.5) == True:
            img_[i] = tf.image.flip_left_right(img_[i])
            gt_boxes_[i] = tf.stack([gt_boxes_[i][...,0],
                                1.0 - gt_boxes_[i][...,3],
                                gt_boxes_[i][...,2],
                                1.0 - gt_boxes_[i][...,1]], -1)

    max_label_num = max([gt_labels_[i].shape[1] for i in range(batch_size)])

    for i in range(batch_size):
        gt_boxes_[i] = tf.concat([gt_boxes_[i], tf.constant(0, dtype=tf.float32, shape=(1, max_label_num - gt_boxes_[i].shape[1], 4))], axis = -2)
        gt_labels_[i] = tf.concat([gt_labels_[i], tf.constant(0, dtype=tf.int32, shape=(1, max_label_num - gt_labels_[i].shape[1]))], axis=-1)
    
    img = tf.concat([img_[i] for i in range(batch_size)], axis=0)
    gt_boxes = tf.concat([gt_boxes_[i] for i in range(batch_size)], axis=0)
    gt_labels = tf.concat([gt_labels_[i] for i in range(batch_size)], axis=0)
    
    return img, gt_boxes, gt_labels
    
    
    
    
    