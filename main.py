#%%
import os
import time
import tensorflow as tf
from tqdm import tqdm
from utils import(
    generate_anchors,
    fetch_dataset,
    get_hyper_params,
    save_dict_to_file,
    generate_save_dir,
    rpn_reg_loss_fn,
    rpn_cls_loss_fn,
    dtn_reg_loss_fn,
    dtn_cls_loss_fn,
    RPN,
    DTN,
    RoIBBox,
    RoIAlign,
    Decode,
    preprocessing,
    rpn_target,
    dtn_target,
    draw_rpn_output,
    draw_dtn_output,
    calculate_AP,
    calculate_AP_const,
)

def build_graph(hyper_params):
    rpn_model = RPN(hyper_params)
    input_shape = (None, 500, 500, 3)
    rpn_model.build(input_shape)

    dtn_model = DTN(hyper_params)
    input_shape = (None, hyper_params['train_nms_topn'], 7, 7, 512)
    dtn_model.build(input_shape)

    return rpn_model, dtn_model


@tf.function
def train_step1(img, bbox_deltas, bbox_labels, hyper_params):
    with tf.GradientTape(persistent=True) as tape:
        '''RPN'''
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
        
        rpn_reg_loss = rpn_reg_loss_fn(rpn_reg_output, bbox_deltas, bbox_labels, hyper_params)
        rpn_cls_loss = rpn_cls_loss_fn(rpn_cls_output, bbox_labels)
        rpn_loss = rpn_reg_loss + rpn_cls_loss
        
    grads_rpn = tape.gradient(rpn_loss, rpn_model.trainable_weights)

    optimizer1.apply_gradients(zip(grads_rpn, rpn_model.trainable_weights))

    return rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map


@tf.function
def train_step2(pooled_roi, roi_deltas, roi_labels):
    with tf.GradientTape(persistent=True) as tape:
        '''DTN'''
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi, training=True)
        
        dtn_reg_loss = dtn_reg_loss_fn(dtn_reg_output, roi_deltas, roi_labels, hyper_params)
        dtn_cls_loss = dtn_cls_loss_fn(dtn_cls_output, roi_labels)
        dtn_loss = dtn_reg_loss + dtn_cls_loss

    grads_dtn = tape.gradient(dtn_loss, dtn_model.trainable_weights)
    optimizer2.apply_gradients(zip(grads_dtn, dtn_model.trainable_weights))

    return dtn_reg_loss, dtn_cls_loss

#%% 
hyper_params = get_hyper_params()
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])
iters = hyper_params['iters']
batch_size = hyper_params['batch_size']
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]

dataset, labels = fetch_dataset(dataset_name, "train", img_size)
dataset = dataset.map(lambda x, y, z: preprocessing(x, y, z))
data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
tf.random.set_seed(42)
dataset = dataset.shuffle(buffer_size=14000, seed=42)
dataset = dataset.repeat().padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values, drop_remainder=True)
dataset = iter(dataset)

labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)

anchors = generate_anchors(hyper_params)

#%%

rpn_model, dtn_model = build_graph(hyper_params)

boundaries = [100000, 200000, 300000]
values = [1e-5, 1e-6, 1e-7, 1e-8]
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

optimizer1 = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

#%%
if __name__ == "main":
    
#%%
atmp_dir = os.getcwd()
atmp_dir = generate_save_dir(atmp_dir, hyper_params)

step = 0
progress_bar = tqdm(range(hyper_params['iters']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, hyper_params['iters']))
start_time = time.time()

for _ in progress_bar:
    try: img, gt_boxes, gt_labels = next(dataset)
    except: continue
    bbox_deltas, bbox_labels = rpn_target(anchors, gt_boxes, gt_labels, hyper_params)
    rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map = train_step1(img, bbox_deltas, bbox_labels, hyper_params)

    roi_bboxes, _ = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    pooled_roi = RoIAlign(roi_bboxes, feature_map, hyper_params)
    roi_deltas, roi_labels = dtn_target(roi_bboxes, gt_boxes, gt_labels, hyper_params)
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
save_dict_to_file(hyper_params, atmp_dir + '/hyper_params')

#%%test
hyper_params["batch_size"] = batch_size = 1

dataset, labels = fetch_dataset(dataset_name, "train", img_size)
dataset = dataset.map(lambda x, y, z: preprocessing(x, y, z))

dataset = dataset.repeat().padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
dataset = iter(dataset)

rpn_model = RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)
rpn_model.load_weights(atmp_dir + '/rpn_weights/weights')

dtn_model = DTN(hyper_params)
input_shape = (None, hyper_params['train_nms_topn'], 7, 7, 512)
dtn_model.build(input_shape)
dtn_model.load_weights(atmp_dir + '/dtn_weights/weights')

total_time = []
mAP = []

img_num = 0
progress_bar = tqdm(range(hyper_params['attempts']))
for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    start_time = time.time()
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
    roi_bboxes, roi_scores = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    pooled_roi = RoIAlign(roi_bboxes, feature_map, hyper_params)
    dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
    final_bboxes, final_labels, final_scores = Decode(dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params)
    time_ = float(time.time() - start_time)*1000
    AP = calculate_AP_const(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
    total_time.append(time_)
    mAP.append(AP)

    draw_rpn_output(img, roi_bboxes, roi_scores, 5, atmp_dir, img_num)
    draw_dtn_output(img, final_bboxes, labels, final_labels, final_scores, atmp_dir, img_num)
    img_num += 1

mAP_res = "%.2f" % (tf.reduce_mean(mAP))
total_time_res = "%.2fms" % (tf.reduce_mean(total_time))

result = {"mAP" : mAP_res,
          "total_time" : total_time_res}

save_dict_to_file(result, atmp_dir + "/result")
