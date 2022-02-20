#%% MODULE IMPORT
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#
from tqdm import tqdm
from PIL import ImageDraw
from tensorflow import keras
#
import utils, data_utils, preprocessing_utils, anchor_utils, model_utils, postprocessing_utils
#%%
hyper_params = utils.get_hyper_params()
hyper_params["train_nms_topn"] = 10
hyper_params["test_nms_topn"] = 10
hyper_params["nms_iou_threshold"] = 0.9
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])

hyper_params["batch_size"] = batch_size = 1
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]

dataset, labels = data_utils.fetch_dataset(dataset_name, "train", img_size)
dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))

data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
dataset = dataset.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values, drop_remainder=True)
dataset = iter(dataset)

labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)

anchors = anchor_utils.generate_anchors(hyper_params)


#%%
weights_dir = os.getcwd() + "/frcnn_atmp"
weights_dir = weights_dir + "/" + os.listdir(weights_dir)[1]

rpn_model = model_utils.RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)
rpn_model.load_weights(weights_dir + '/rpn_weights/weights')

#%%
# save_dir = os.getcwd()
# save_dir = utils.generate_save_dir(save_dir, hyper_params)

total_time = []
mAP = []
num = 0
progress_bar = tqdm(range(hyper_params['attempts']))
for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    start_time = time.time()
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
    roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    pooled_roi = postprocessing_utils.RoIAlign(roi_bboxes, feature_map, hyper_params)
    dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params)
    time_ = float(time.time() - start_time)*1000
    AP = test_utils.calculate_AP(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
    print(num)
    test_utils.draw_dtn_output(img, final_bboxes, labels, final_labels, final_scores, )
    total_time.append(time_)
    mAP.append(AP)
    num += 1
    

print("mAP: %.2f" % (tf.reduce_mean(mAP)))
print("Time taken: %.2fms" % (tf.reduce_mean(total_time)))

#%%
#%%
i = atmp
roi_dir = r'C:\won\data\pascal_voc\voc_2007_roi_atmp'

tmp = True
while tmp :
    if os.path.isdir(roi_dir + str(i)) : 
        i+= 1
    else: 
        os.makedirs(roi_dir + str(i))
        print("Generated roi" + str(i))
        tmp = False

roi_dir = roi_dir + str(i) 

#%%
progress_bar = tqdm(range(train_total_items))
for attempt in progress_bar:
    extract_dic = dict()
    
    img, gt_boxes, gt_labels = next(dataset)
    img.resize
    tf.cast(gt_boxes * 416, tf.int32)
    
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
    roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    extract_roi = tf.cast(roi_bboxes * 500, tf.int32)

    img

    extract_dic["pooled_roi"] = pooled_roi.numpy()
    extract_dic["gt_boxes"] = gt_boxes.numpy()
    extract_dic["gt_labels"] = gt_labels.numpy()
    extract = np.array([extract_dic], dtype=object)

    iou_map = rpn_utils.generate_iou(roi_bboxes, gt_boxes)
    #
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # 1500개의 roi_bbox 와의 iou가 가장 큰 gtbox 인덱스
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    
    rpn_class = tf.where(merged_iou_map >= 0.5, tf.gather(gt_labels, max_indices_each_gt_box, axis=1, batch_dims=1), tf.constant(0, dtype=tf.int32))
    rpn_class = rpn_class.numpy()

    save_dir = roi_dir + r"\\" + res_filename[0]
    os.makedirs(save_dir)

    np.save((save_dir + "\\" + res_filename[0] + "_label.npy"), rpn_class, allow_pickle=True)
    np.save((save_dir + "\\" + res_filename[0] + "_feature.npy"), extract, allow_pickle=True)

    for i in range(10):
        roi_bboxes_ = roi_bboxes * img_size
        tmp = tf.round(roi_bboxes_[0][i])
        img_ = img[0][int(tmp[0]):int(tmp[2]), int(tmp[1]):int(tmp[3])]
        img_ = img_.numpy()
        np.save((save_dir + "\\" + res_filename[0] + "_" + str(i) + '.npy'), img_, allow_pickle=True)
        
        # img_ = tf.keras.preprocessing.image.array_to_img(img_)
        # plt.imshow(img_)
        # plt.savefig(save_dir + "\\" + res_filename[0] + "_" + str(i) + '.png')
# %%

# pooled_roi_ = pooled_roi.numpy()
# a = np.load()
# a = np.load(roi_dir + "\\" + res_filename[0] + ".npy", allow_pickle=True)

# rois = a[0]["pooled_roi"]
# rois = tf.convert_to_tensor(rois)
# roi = rois[0][0]

# classification = VGG16(include_top=True, input_shape=(hyper_params["img_size"], 
#                                                                 hyper_params["img_size"],
#                                                                 3))        
# classification.summay()
# classification = VGG16()
# classification.build(input_shape=(None, 500, 500, 3))
# classification.summary()

# model_input = classification.get_layer("flatten")
# model_output = classification.output
# model = tf.keras.Model(inputs=model_input, outputs=model_output)

# from keras.applications.vgg16 import decode_predictions

