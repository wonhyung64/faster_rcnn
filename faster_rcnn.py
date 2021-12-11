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
from tensorflow.keras.layers import Layer
#
from utils import bbox_utils, loss_utils, model_utils, rpn_utils, preprocessing_utils
#%% HYPER PARAMETERS
hyper_params = {"img_size": 500,
                "feature_map_shape": 31,
                "anchor_ratios": [1., 2., 1./2.],
                "anchor_scales": [128, 256, 512],
                "pre_nms_topn": 6000,
                "train_nms_topn": 1500,
                "test_nms_topn": 300,
                "nms_iou_threshold": 0.7,
                "total_pos_bboxes": 128,
                "total_neg_bboxes": 128,
                "pooling_size": (7,7),
                "variances": [0.1, 0.1, 0.2, 0.2],
                "pos_threshold" : 0.7,
                "neg_threshold" : 0.3,
                "batch_size" : 16,
                "iters" : 20000,
                }
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
class RoIBBox(Layer):
    
    def __init__(self, anchors, hyper_params, test=False, **kwargs):
        super(RoIBBox, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        self.anchors = tf.constant(anchors, dtype=tf.float32)
        self.test = test

    def get_config(self):
        config = super(RoIBBox, self).get_config()
        config.update({"hyper_params": self.hyper_params, "anchors": self.anchors.numpy()})
        return config
    def call(self, inputs):
        rpn_bbox_deltas = inputs[0]
        rpn_probs = inputs[1]
        gt_labels = inputs[2]
        anchors = self.anchors
        #
        pre_nms_topn = self.hyper_params["pre_nms_topn"] # pre_nms_topn : 6000
        post_nms_topn = self.hyper_params["train_nms_topn"]
        if self.test == True: post_nms_topn = self.hyper_params["test_nms_topn"]
        nms_iou_threshold = self.hyper_params["nms_iou_threshold"] # nms_iou_threshold : 0.7
        variances = self.hyper_params["variances"]
        total_anchors = anchors.shape[0]
        batch_size = tf.shape(rpn_bbox_deltas)[0]
        rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, total_anchors, 4))
        rpn_probs = tf.reshape(rpn_probs, (batch_size, total_anchors))
        #
        rpn_bbox_deltas *= variances
        #
        rpn_bboxes = bbox_utils.delta_to_bbox(anchors, rpn_bbox_deltas)

        _, pre_indices = tf.nn.top_k(rpn_probs, pre_nms_topn)
        #
        pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
        pre_roi_probs = tf.gather(rpn_probs, pre_indices, batch_dims=1)
        #
        pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
        pre_roi_probs = tf.reshape(pre_roi_probs, (batch_size, pre_nms_topn, 1))
        #
        
        roi_bboxes, roi_scores, _, _ = tf.image.combined_non_max_suppression(pre_roi_bboxes, pre_roi_probs,
                                                            max_output_size_per_class=post_nms_topn,
                                                            max_total_size = post_nms_topn,
                                                            iou_threshold=nms_iou_threshold)
        #
        return roi_bboxes, roi_scores

#%%
class RoIPooling(Layer):
    #
    def __init__(self, hyper_params, **kwargs):
        super(RoIPooling, self).__init__(**kwargs)
        self.hyper_params = hyper_params
    #
    def get_config(self):
        config = super(RoIPooling, self).get_config()
        config.update({"hyper_params": self.hyper_params})
        return config
    #
    def call(self, inputs):
        feature_map = inputs[0]
        roi_bboxes = inputs[1]
        pooling_size = self.hyper_params["pooling_size"]
        batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
        #
        row_size = batch_size * total_bboxes
        #
        pooling_bbox_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes))
        pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1, ))
        pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))
        #
        pooling_feature_map = tf.image.crop_and_resize(feature_map,
                                                        pooling_bboxes,
                                                        pooling_bbox_indices,
                                                        pooling_size) 

        final_pooling_feature_map = tf.reshape(pooling_feature_map, (batch_size,
                                                                     total_bboxes,
                                                                     pooling_feature_map.shape[1],
                                                                     pooling_feature_map.shape[2],
                                                                     pooling_feature_map.shape[3]))
        #
        return final_pooling_feature_map
#
#%%
class RoIDelta(Layer):
    def __init__(self, hyper_params, **kwargs):
        super(RoIDelta, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        
    def get_config(self):
        config = super(RoIDelta, self).get_config()
        config.update({"hyper_params": self.hyper_params})
        return config
    
    def call(self, inputs):
        roi_bboxes = inputs[0]
        gt_boxes = inputs[1]
        gt_labels = inputs[2]

        total_labels = self.hyper_params["total_labels"]
        total_pos_bboxes = self.hyper_params["total_pos_bboxes"]
        total_neg_bboxes = self.hyper_params["total_neg_bboxes"]
        variances = self.hyper_params["variances"]
        #
        iou_map = rpn_utils.generate_iou(roi_bboxes, gt_boxes)
        #
        max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
        merged_iou_map = tf.reduce_max(iou_map, axis=2)
        pos_mask = tf.greater(merged_iou_map, 0.5)
        pos_mask = rpn_utils.randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
        #
        neg_mask = tf.logical_and(tf.less(merged_iou_map, 0.5), tf.greater(merged_iou_map, 0.1))
        neg_mask = rpn_utils.randomly_select_xyz_mask(neg_mask, tf.constant([total_neg_bboxes], dtype=tf.int32))
        #
        gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
        expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, axis=-1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
        #
        gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)

        pos_gt_labels = tf.where(pos_mask, gt_labels_map, tf.constant(-1, dtype=tf.int32))
        neg_gt_labels = tf.cast(neg_mask, dtype=tf.int32)

        expanded_gt_labels = pos_gt_labels + neg_gt_labels 
        #
        roi_bbox_deltas = rpn_utils.bbox_to_delta(roi_bboxes, expanded_gt_boxes) / variances
        #
        roi_bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
        scatter_indices = tf.tile(tf.expand_dims(roi_bbox_labels, -1), (1, 1, 1, 4))
        roi_bbox_deltas = scatter_indices * tf.expand_dims(roi_bbox_deltas, -2)

        return roi_bbox_deltas, roi_bbox_labels

#%%
class Decoder(Layer):
    def __init__(self, hyper_params, max_total_size=200, score_threshold=0.7, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.variances = hyper_params["variances"]
        self.total_labels = hyper_params["total_labels"]
        self.max_total_size = max_total_size
        self.score_threshold = score_threshold
        
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            "variances" : self.variances,
            "total_labels": self.total_labels,
            "max_total_size": self.max_total_size,
            "score_threshold" : self.score_threshold
        })
        return config

    def call(self, inputs):
        roi_bboxes = inputs[0]
        pred_deltas = inputs[1]
        pred_label_probs = inputs[2]
        batch_size = tf.shape(pred_deltas)[0]

        pred_deltas = tf.reshape(pred_deltas, (batch_size, -1, self.total_labels, 4))
        pred_deltas *= self.variances

        expanded_roi_bboxes = tf.tile(tf.expand_dims(roi_bboxes, -2), (1, 1, self.total_labels, 1))
        
        pred_bboxes = bbox_utils.delta_to_bbox(expanded_roi_bboxes, pred_deltas)

        pred_labels_map = tf.expand_dims(tf.argmax(pred_label_probs, -1), -1)
        pred_labels = tf.where(tf.not_equal(pred_labels_map, 0), pred_label_probs, tf.zeros_like(pred_label_probs))
        
        final_bboxes, final_scores, final_labels, _ = tf.image.combined_non_max_suppression(
                            pred_bboxes, pred_labels,
                            max_output_size_per_class = self.max_total_size,
                            max_total_size = self.max_total_size,
                            score_threshold=self.score_threshold
                        )
        return final_bboxes, final_labels, final_scores

#%%
rpn_model = model_utils.RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)
# rpn_model.load_weights(r'C:\won\frcnn\atmp5\rpn_weights\weights')

NMS = RoIBBox(anchors, hyper_params, test=False, name='roi_bboxes')
Pooling = RoIPooling(hyper_params, name="roi_pooling")
Delta = RoIDelta(hyper_params, name='roi_deltas')

frcnn_model = model_utils.DTN(hyper_params)
input_shape = (None, hyper_params['train_nms_topn'], 7, 7, 512)
frcnn_model.build(input_shape)
# frcnn_model.load_weights(r'C:\won\frcnn\atmp5\frcnn_weights\weights')
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


#%%

batch_size = 2
hyper_params['batch_size'] = batch_size

NMS = RoIBBox(anchors, hyper_params, test=True, name='roi_bboxes')
Pooling = RoIPooling(hyper_params, name="roi_pooling")
decode = Decoder(hyper_params)

test_dir = r"C:\won\data\pascal_voc\voc2007_np\test\\"

for attempt in range(attempts):

    res_filename = [test_filename[i] for i in range(attempt*batch_size, attempt*batch_size + batch_size)]
    batch_data = np.array([np.load(test_dir + test_filename[i] + ".npy", allow_pickle=True) for i in range(attempt*batch_size, attempt*batch_size+batch_size)])

    img, gt_boxes, gt_labels = preprocessing_utils.preprocessing(batch_data, hyper_params["batch_size"], hyper_params["img_size"], hyper_params["img_size"], evaluate=True)
    
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model.predict(img)
    roi_bboxes, roi_scores = NMS([rpn_reg_output, rpn_cls_output, gt_labels])
    pooled_roi = Pooling([feature_map, roi_bboxes])
    pred_deltas, pred_label_probs = frcnn_model.predict(pooled_roi)
    final_bboxes, final_labels, final_scores = decode([roi_bboxes, pred_deltas, pred_label_probs])

    img_size = img.shape[1]
    
##### NMS OUTPUT #####
    for i, image in enumerate(img):

        y1 = roi_bboxes[i][...,0] * img_size
        x1 = roi_bboxes[i][...,1] * img_size
        y2 = roi_bboxes[i][...,2] * img_size
        x2 = roi_bboxes[i][...,3] * img_size

        denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

        _, top_indices = tf.nn.top_k(roi_scores[i], 2)
        #
        selected_rpn_bboxes = tf.gather(denormalized_box, top_indices, batch_dims=0)

        #
        colors = 234 

        image = tf.keras.preprocessing.image.array_to_img(image)
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        for bbox in selected_rpn_bboxes:
            y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)

            draw.rectangle((x1, y1, x2, y2), outline=colors, width=3)
        
        plt.figure()
        plt.imshow(image)
        plt.savefig(res_dir + r'\res_nms\\' + res_filename[i] + '.png')
    
    
##### FINAL BOUNDING BOXES #####
    for i , image in enumerate(img):

        tmp = tf.reshape(pred_deltas[i], shape=(1, pred_deltas.shape[1], hyper_params['total_labels'], 4))
        tmp *= hyper_params['variances']

        expanded_roi_bboxes = tf.reshape(tf.tile(tf.expand_dims(roi_bboxes[i], -2), (1, hyper_params['total_labels'], 1)), shape=(1,hyper_params['test_nms_topn'], hyper_params['total_labels'], 4))
        final_bboxes_ = bbox_utils.delta_to_bbox(expanded_roi_bboxes, tmp)

        y1 = final_bboxes_[...,0] * img_size
        x1 = final_bboxes_[...,1] * img_size
        y2 = final_bboxes_[...,2] * img_size
        x2 = final_bboxes_[...,3] * img_size

        denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

        colors = [57, 140, 234]

        image = tf.keras.preprocessing.image.array_to_img(image)
        width, height = image.size
        draw = ImageDraw.Draw(image)
        for j in range(3):
            color = colors[j]
            tmp_box = denormalized_box[0][j]
            bbox = tmp_box[0]
            y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            
        plt.figure()
        plt.imshow(image)
        plt.savefig(res_dir + r'\res_final_bbox\\' + res_filename[i] + '.png')

##### FASTER R-CNN RESULT #####
    for i , image in enumerate(img):

        y1 = final_bboxes[i][...,0] * img_size
        x1 = final_bboxes[i][...,1] * img_size
        y2 = final_bboxes[i][...,2] * img_size
        x2 = final_bboxes[i][...,3] * img_size

        denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

        colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)

        image = tf.keras.preprocessing.image.array_to_img(image)
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        for index, bbox in enumerate(denormalized_box):
            y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)
            width = x2 - x1
            height = y2 - y1

            final_labels_ = tf.reshape(final_labels[i], shape=(200,))
            final_scores_ = tf.reshape(final_scores[i], shape=(200,))
            label_index = int(final_labels_[index])
            color = tuple(colors[label_index].numpy())
            label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores_[index])
            draw.text((x1 + 4, y1 + 2), label_text, fill=color)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        
        plt.figure()
        plt.imshow(image)
        plt.savefig(res_dir + r'\res_frcnn\\' + res_filename[i] + '.png')

# %%