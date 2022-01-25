#%% MODULE IMPORT
import os
import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tensorflow import keras

import bbox_utils, loss_utils, model_utils, preprocessing_utils, tune_utils
#%% HYPER PARAMETERS
hyper_params = tune_utils.get_hyper_params()
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])
#
iters = hyper_params['iters']
batch_size = hyper_params['batch_size']
img_size = hyper_params["img_size"]

#%%
info_dir = r"C:\won\data\pascal_voc\voc2007_np"
info = np.load(info_dir + r"\info.npy", allow_pickle=True)

labels = info[0]["labels"]
train_filename = info[0]['train_filename'] + info[0]['val_filename']
test_filename = info[0]['test_filename']

train_total_items = len(train_filename)
test_total_items = len(test_filename)

labels = ["bg"] + labels

hyper_params["total_labels"] = len(labels)
#%%
anchors = bbox_utils.generate_anchors(hyper_params)
#%%
NMS = model_utils.RoIBBox(anchors, hyper_params)
NMS.build()

#%%
rpn_model = model_utils.RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)
rpn_model.summary()
rpn_model.load_weights(r'C:\won\frcnn\atmp1\rpn_weights\weights')

NMS = RoIBBox(anchors, hyper_params, test=False, name='roi_bboxes')
Pooling = RoIPooling(hyper_params, name="roi_pooling")
Delta = RoIDelta(hyper_params, name='roi_deltas')

frcnn_model = model_utils.DTN(hyper_params)
input_shape = (None, hyper_params['train_nms_topn'], 7, 7, 512)
frcnn_model.build(input_shape)
# frcnn_model.summary()
frcnn_model.load_weights(r'C:\won\frcnn\atmp1\frcnn_weights\weights')
#%%
optimizer1 = keras.optimizers.Adam(learning_rate=1e-5)
optimizer2 = keras.optimizers.Adam(learning_rate=1e-5)
#%%
@tf.function
def train_step1(img, bbox_deltas, bbox_labels, hyper_params):
    with tf.GradientTape(persistent=True) as tape:
        '''RPN'''
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
        
        rpn_reg_loss = loss_utils.region_reg_loss(rpn_reg_output, bbox_deltas, bbox_labels, hyper_params)
        rpn_cls_loss = loss_utils.region_cls_loss(rpn_cls_output, bbox_labels)
        rpn_loss = rpn_reg_loss + rpn_cls_loss
        
    grads_rpn = tape.gradient(rpn_loss, rpn_model.trainable_weights)

    optimizer1.apply_gradients(zip(grads_rpn, rpn_model.trainable_weights))

    return rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map

#%%
@tf.function
def train_step2(pooled_roi, roi_delta):
    with tf.GradientTape(persistent=True) as tape:
        '''DTNnition'''
        frcnn_pred = frcnn_model(pooled_roi, training=True)
        
        frcnn_reg_loss = loss_utils.dtn_reg_loss(frcnn_pred[0], roi_delta[0], roi_delta[1], hyper_params)
        frcnn_cls_loss = loss_utils.dtn_cls_loss(frcnn_pred[1], roi_delta[1])
        frcnn_loss = frcnn_reg_loss + frcnn_cls_loss

    grads_frcnn = tape.gradient(frcnn_loss, frcnn_model.trainable_weights)
    optimizer2.apply_gradients(zip(grads_frcnn, frcnn_model.trainable_weights))

    return frcnn_reg_loss, frcnn_cls_loss

#%%
def save_dict_to_file(dic,dict_dir):
    f = open(dict_dir + '.txt', 'w')
    f.write(str(dic))
    f.close()

#%%
train_dir = r"C:\won\data\pascal_voc\voc2007_np\train_val\\"

pos_num_lst = []
step = 0

progress_bar = tqdm(range(hyper_params['iters']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, hyper_params['iters']))

start_time = time.time()
for _ in progress_bar:
    chk_pos_num = []

    batch_data = np.array([np.load(train_dir + train_filename[i] + ".npy", allow_pickle=True) for i in list(np.random.randint(0, train_total_items, batch_size))])
    img, gt_boxes, gt_labels = preprocessing_utils.preprocessing(batch_data, hyper_params["batch_size"], hyper_params["img_size"], hyper_params["img_size"], evaluate=False, augmentation=True) 
    bbox_deltas, bbox_labels, chk_pos_num = rpn_utils.calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params, chk_pos_num)

    pos_num_lst.append(chk_pos_num)
    
    rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map = train_step1(img, bbox_deltas, bbox_labels, hyper_params)
    roi_bboxes, _ = NMS([rpn_reg_output, rpn_cls_output, gt_labels])
    pooled_roi = Pooling([feature_map, roi_bboxes])
    roi_delta = Delta([roi_bboxes, gt_boxes, gt_labels])
    frcnn_reg_loss, frcnn_cls_loss = train_step2(pooled_roi, roi_delta)

    step += 1
    
    progress_bar.set_description('iteration {}/{} | rpn_reg {:.4f}, rpn_cls {:.4f}, rpn {:.4f}, frcnn_reg {:.4f}, frcnn_cls {:.4f}, frcnn {:.4f}, loss {:.4f}'.format(
        step, hyper_params['iters'], 
        rpn_reg_loss.numpy(), rpn_cls_loss.numpy(), (rpn_reg_loss + rpn_cls_loss).numpy(), frcnn_reg_loss.numpy(), frcnn_cls_loss.numpy(), (frcnn_reg_loss + frcnn_cls_loss).numpy(), (rpn_reg_loss + rpn_cls_loss + frcnn_reg_loss + frcnn_cls_loss).numpy()
    )) 
    
    if step % 500 == 0:
        print(progress_bar.set_description('iteration {}/{} | rpn_reg {:.4f}, rpn_cls {:.4f}, rpn {:.4f}, frcnn_reg {:.4f}, frcnn_cls {:.4f}, frcnn {:.4f}, loss {:.4f}'.format(
            step, hyper_params['iters'], 
            float(rpn_reg_loss), float(rpn_cls_loss), float(rpn_reg_loss + rpn_cls_loss), float(frcnn_reg_loss), float(frcnn_cls_loss), float(frcnn_reg_loss + frcnn_cls_loss), float(rpn_reg_loss + rpn_cls_loss + frcnn_reg_loss + frcnn_cls_loss)
        )))

print("Time taken: %.2fs" % (time.time() - start_time))
print("pos num mean : ", np.mean(pos_num_lst), "pos num std : ", np.std(pos_num_lst))
#%%
i = 1
res_dir = r'C:\won\frcnn\atmp'

tmp = True
while tmp :
    if os.path.isdir(res_dir + str(i)) : 
        i+= 1
    else: 
        os.makedirs(res_dir + str(i))
        print("Generated atmp" + str(i))
        tmp = False

res_dir = res_dir + str(i) 

save_dict_to_file(hyper_params, res_dir + r'\hyper_params')
os.makedirs(res_dir + r'\rpn_weights')
os.makedirs(res_dir + r'\frcnn_weights')
os.makedirs(res_dir + r'\res_nms')
os.makedirs(res_dir + r'\res_final_bbox')
os.makedirs(res_dir + r'\res_frcnn')
#%%
rpn_model.save_weights(res_dir + r'\rpn_weights\weights')
frcnn_model.save_weights(res_dir + r'\frcnn_weights\weights')
print("Weights Saved")

