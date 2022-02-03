#%% MODULE IMPORT
import os
import time
import tensorflow as tf

from tqdm import tqdm
from tensorflow import keras

import utils, loss_utils, model_utils, preprocessing_utils, postprocessing_utils, data_utils, anchor_utils, target_utils, test_utils, ship

#%% 

hyper_params = utils.get_hyper_params()
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])

iters = hyper_params['iters']
batch_size = hyper_params['batch_size']
img_size = (hyper_params["img_size"], hyper_params["img_size"])

dataset, labels = ship.fetch_dataset("ship", "train", img_size)

dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing(x, y, z, w))
# dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing(x, y, z, w))
data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
dataset = dataset.repeat().padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values, drop_remainder=True)
dataset = iter(dataset)


labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)

anchors = anchor_utils.generate_anchors(hyper_params)

#%%
rpn_model = model_utils.RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)

optimizer1 = keras.optimizers.Adam(learning_rate=1e-5)

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

optimizer2 = keras.optimizers.Adam(learning_rate=1e-5)

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
    img.shape
    gt_boxes.shape
    gt_labels.shape
    bbox_deltas, bbox_labels = target_utils.rpn_target(anchors, gt_boxes, gt_labels, hyper_params)
    bbox_deltas.shape
    bbox_labels.shape
    rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map = train_step1(img, bbox_deltas, bbox_labels, hyper_params)
    rpn_reg_output.shape
    rpn_cls_output.shape
    feature_map.shape

    roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    roi_bboxes.shape
    pooled_roi = postprocessing_utils.RoIAlign(roi_bboxes, feature_map, hyper_params)
    pooled_roi.shape
    roi_deltas, roi_labels = target_utils.dtn_target(roi_bboxes, gt_boxes, gt_labels, hyper_params)
    roi_deltas.shape
    roi_labels.shape
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
        rpn_model.save_weights(atmp_dir + '/rpn_weights/weights')
        dtn_model.save_weights(atmp_dir + '/dtn_weights/weights')
        print("Weights Saved")

print("Time taken: %.2fs" % (time.time() - start_time))
utils.save_dict_to_file(hyper_params, atmp_dir + '/hyper_params')

#%%test
hyper_params["batch_size"] = batch_size = 1

dataset, _ = ship.fetch_dataset("coco17", "test", img_size)

dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing(x, y, z, w))
dataset = dataset.repeat().padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
dataset = iter(dataset)

rpn_model = model_utils.RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)
rpn_model.load_weights(atmp_dir + '/rpn_weights/weights')

dtn_model = model_utils.DTN(hyper_params)
input_shape = (None, hyper_params['train_nms_topn'], 7, 7, 512)
dtn_model.build(input_shape)
dtn_model.load_weights(atmp_dir + '/dtn_weights/weights')

total_time = []
mAP = []

progress_bar = tqdm(range(hyper_params['attempts']))
for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    img.shape
    gt_boxes.shape
    gt_labels.shape
    start_time = time.time()
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
    roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    pooled_roi = postprocessing_utils.RoIAlign(roi_bboxes, feature_map, hyper_params)
    dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params)
    time_ = float(time.time() - start_time)*1000
    AP = test_utils.calculate_AP(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
    total_time.append(time_)
    mAP.append(AP)

mAP_res = "%.2f" % (tf.reduce_mean(mAP))
total_time_res = "%.2fms" % (tf.reduce_mean(total_time))

result = {"mAP" : mAP_res,
          "total_time" : total_time_res}

utils.save_dict_to_file(result, atmp_dir + "/result")
#%%

for i in range(400):
    try:
        print(i)
        img, gt_boxes, gt_labels = next(dataset)
        print(gt_labels.shape)
    except:
        print("except")
        dataset, labels = ship.fetch_dataset("ship", "train", img_size)

        dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing(x, y, z, w))
        # dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing(x, y, z, w))
        data_shapes = ([None, None, None], [None, None], [None])
        padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
        dataset = dataset.repeat().padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values, drop_remainder=True)
        dataset = iter(dataset)

        print(i)
        img, gt_boxes, gt_labels = next(dataset)
        print(gt_labels.shape)
    # img[3]
    # gt_boxes[3]
    # gt_labels[3]
    # except: 
    #     img_, gt_boxes_, gt_labels_ = next(dataset)
    #     img_[3]
    #     gt_boxes_[3]
    #     gt_labels_[3]
    #     break
# %%
'''
194번째만 계속 튕습
data 불러올떄 try except 로 처리하여 학습
'''

