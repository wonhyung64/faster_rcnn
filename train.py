#%% MODULE IMPORT
import os
import time
import tensorflow as tf

from tqdm import tqdm
from tensorflow import keras

import utils, loss_utils, model_utils, preprocessing_utils, postprocessing_utils, tune_utils, data_utils, anchor_utils, target_utils

#%% 

hyper_params = tune_utils.get_hyper_params()
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])

iters = hyper_params['iters']
batch_size = hyper_params['batch_size']
img_size = (hyper_params["img_size"], hyper_params["img_size"])

train, labels = data_utils.fetch_dataset("voc07", "train", img_size)

train = train.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))
data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
train = train.repeat().padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
dataset = iter(train)

labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)

anchors = anchor_utils.generate_anchors(hyper_params)

#%%
rpn_model = model_utils.RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)

optimizer1 = keras.optimizers.SGD(learning_rate=1e-5)

@tf.function
def train_step1(img, bbox_deltas, bbox_labels, hyper_params):
    with tf.GradientTape(persistent=True) as tape:
        '''RPN'''
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
        
        rpn_reg_loss = loss_utils.rpn_reg_loss_fn(rpn_reg_output, bbox_deltas, bbox_labels, hyper_params)
        rpn_cls_loss = loss_utils.rpn_cls_loss_fn(rpn_cls_output, bbox_labels)
        rpn_loss = rpn_reg_loss + rpn_cls_loss
        
    grads_rpn = tape.gradient(rpn_loss, rpn_model.trainable_weights)

    optimizer1.apply_gradients(zip(grads_rpn, rpn_model.trainable_weights))

    return rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map

#%%
dtn_model = model_utils.DTN(hyper_params)
input_shape = (None, hyper_params['train_nms_topn'], 7, 7, 512)
dtn_model.build(input_shape)

optimizer2 = keras.optimizers.SGD(learning_rate=1e-5)

@tf.function
def train_step2(pooled_roi, roi_deltas, roi_labels):
    with tf.GradientTape(persistent=True) as tape:
        '''DTN'''
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi, training=True)
        
        dtn_reg_loss = loss_utils.dtn_reg_loss_fn(dtn_reg_output, roi_deltas, roi_labels, hyper_params)
        dtn_cls_loss = loss_utils.dtn_cls_loss_fn(dtn_cls_output, roi_labels)
        dtn_loss = dtn_reg_loss + dtn_cls_loss

    grads_dtn = tape.gradient(dtn_loss, dtn_model.trainable_weights)
    optimizer2.apply_gradients(zip(grads_dtn, dtn_model.trainable_weights))

    return dtn_reg_loss, dtn_cls_loss


#%%
atmp_dir = os.getcwd()
atmp_dir = utils.generate_save_dir(atmp_dir, hyper_params)

step = 0
progress_bar = tqdm(range(hyper_params['iters']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, hyper_params['iters']))
start_time = time.time()

for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    bbox_deltas, bbox_labels = target_utils.rpn_target(anchors, gt_boxes, gt_labels, hyper_params)
    rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map = train_step1(img, bbox_deltas, bbox_labels, hyper_params)
    roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    pooled_roi = postprocessing_utils.RoIAlign(roi_bboxes, feature_map, hyper_params)
    roi_deltas, roi_labels = target_utils.dtn_target(roi_bboxes, gt_boxes, gt_labels, hyper_params)
    dtn_reg_loss, dtn_cls_loss = train_step2(pooled_roi, roi_deltas, roi_labels)

    step += 1
    
    progress_bar.set_description('iteration {}/{} | rpn_reg {:.4f}, rpn_cls {:.4f}, dtn_reg {:.4f}, dtn_cls {:.4f}, loss {:.4f}'.format(
        step, hyper_params['iters'], 
        rpn_reg_loss.numpy(), rpn_cls_loss.numpy(), dtn_reg_loss.numpy(), dtn_cls_loss.numpy(), (rpn_reg_loss + rpn_cls_loss + dtn_reg_loss + dtn_cls_loss).numpy()
    )) 
    
    if step % 500 == 0:
        print(progress_bar.set_description('iteration {}/{} | rpn_reg {:.4f}, rpn_cls {:.4f}, dtn_reg {:.4f}, dtn_cls {:.4f}, loss {:.4f}'.format(
            step, hyper_params['iters'], 
            rpn_reg_loss.numpy(), rpn_cls_loss.numpy(), dtn_reg_loss.numpy(), dtn_cls_loss.numpy(), (rpn_reg_loss + rpn_cls_loss + dtn_reg_loss + dtn_cls_loss).numpy()
        )))
    
    if step % 1000 == 0 :
        rpn_model.save_weights(atmp_dir + r'\rpn_weights\weights')
        dtn_model.save_weights(atmp_dir + r'\dtn_weights\weights')
        print("Weights Saved")

print("Time taken: %.2fs" % (time.time() - start_time))
#%%

