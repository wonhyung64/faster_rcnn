#%%
import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

try: import neptune.new as neptune
except:
    !pip install neptune-client
    import neptune.new as neptune

from utils import (
    generate_anchors,
    get_hyper_params,
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
    input_shape = (None, hyper_params["train_nms_topn"], 7, 7, 512)
    dtn_model.build(input_shape)

    return rpn_model, dtn_model


@tf.function
def train_step1(img, bbox_deltas, bbox_labels, hyper_params):
    with tf.GradientTape(persistent=True) as tape:
        """RPN"""
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)

        rpn_reg_loss = rpn_reg_loss_fn(
            rpn_reg_output, bbox_deltas, bbox_labels, hyper_params
        )
        rpn_cls_loss = rpn_cls_loss_fn(rpn_cls_output, bbox_labels)
        rpn_loss = rpn_reg_loss + rpn_cls_loss

    grads_rpn = tape.gradient(rpn_loss, rpn_model.trainable_weights)
    optimizer1.apply_gradients(zip(grads_rpn, rpn_model.trainable_weights))

    return rpn_reg_loss, rpn_cls_loss, rpn_reg_output, rpn_cls_output, feature_map


@tf.function
def train_step2(pooled_roi, roi_deltas, roi_labels):
    with tf.GradientTape(persistent=True) as tape:
        """DTN"""
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi, training=True)

        dtn_reg_loss = dtn_reg_loss_fn(
            dtn_reg_output, roi_deltas, roi_labels, hyper_params
        )
        dtn_cls_loss = dtn_cls_loss_fn(dtn_cls_output, roi_labels)
        dtn_loss = dtn_reg_loss + dtn_cls_loss

    grads_dtn = tape.gradient(dtn_loss, dtn_model.trainable_weights)
    optimizer2.apply_gradients(zip(grads_dtn, dtn_model.trainable_weights))

    return dtn_reg_loss, dtn_cls_loss


#%%
if __name__ == "__main__":
    run = neptune.init(
        project="wonhyung64/model-frcnn",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZTNlODVlMi0xMzIyLTQwYzQtYmNkYy1kNWYyZmM1MGFiMjcifQ==",
    )

    hyper_params = get_hyper_params()
    hyper_params["anchor_count"] = len(hyper_params["anchor_ratios"]) * len(
        hyper_params["anchor_scales"]
    )
    epochs = hyper_params["epochs"]
    batch_size = hyper_params["batch_size"]
    img_size = (hyper_params["img_size"], hyper_params["img_size"])
    dataset_name = hyper_params["dataset_name"]

    run["hyper_params"] = hyper_params
    run["sys/name"] = "frcnn-optimization"
    run["sys/tags"].add([dataset_name, str(img_size)])

    data_dir = hyper_params["data_dir"]
    train1, dataset_info = tfds.load(
        name="voc/2007", split="train", data_dir=data_dir, with_info=True
    )
    train2, _ = tfds.load(
        name="voc/2007",
        split="validation[100:]",
        data_dir=data_dir,
        with_info=True,
    )
    validation, _ = tfds.load(
        name="voc/2007",
        split="validation[:100]",
        data_dir=data_dir,
        with_info=True,
    )
    test, _ = tfds.load(
        name="voc/2007",
        split="train[:10%]",
        data_dir=data_dir,
        with_info=True,
    )
    train = train1.concatenate(train2)

    data_ck = iter(train)
    one_epoch = 0
    while True:
        try:
            next(data_ck)
        except:
            break
        one_epoch += 1

    labels = dataset_info.features["labels"].names
    labels = ["bg"] + labels
    hyper_params["total_labels"] = len(labels)

    data_shapes = ([None, None, None], [None, None], [None])
    padding_values = (
        tf.constant(0, tf.float32),
        tf.constant(0, tf.float32),
        tf.constant(-1, tf.int32),
    )

    tf.random.set_seed(42)
    train_set = train.map(lambda x, y=img_size, z=False: preprocessing(x, y, z))
    train_set = train_set.shuffle(buffer_size=one_epoch, seed=42)
    train_set = train_set.repeat().padded_batch(
        batch_size,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )
    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)
    train_set = iter(train_set)

    valid_set = validation.map(lambda x, y=img_size, z=True: preprocessing(x, y, z))
    valid_set = valid_set.repeat().padded_batch(
        batch_size=1,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )
    valid_set = valid_set.prefetch(tf.data.experimental.AUTOTUNE)
    valid_set = iter(valid_set)

    test_set = test.map(lambda x, y=img_size, z=True: preprocessing(x, y, z))
    test_set = test_set.repeat().padded_batch(
        batch_size=1,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)
    test_set = iter(test_set)

    anchors = generate_anchors(hyper_params)

    rpn_model, dtn_model = build_graph(hyper_params)

    boundaries = [100000, 200000, 300000]
    values = [1e-5, 1e-6, 1e-7, 1e-8]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )

    optimizer1 = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    if not (os.path.exists("model_ckpt")):
        os.mkdir("model_ckpt")
        os.mkdir("model_ckpt/rpn_weights")
        os.mkdir("model_ckpt/dtn_weights")

    step = 0
    best_mAP = 0
    iters = round(epochs * one_epoch / batch_size)
    progress_train = tqdm(range(iters))
    progress_train.set_description(
        "iteration {}/{} | current loss ?".format(step, iters)
    )
    start_time = time.time()

    for _ in progress_train:
        try:
            img, gt_boxes, gt_labels = next(train_set)
        except:
            continue
        bbox_deltas, bbox_labels = rpn_target(
            anchors, gt_boxes, gt_labels, hyper_params
        )
        (
            rpn_reg_loss,
            rpn_cls_loss,
            rpn_reg_output,
            rpn_cls_output,
            feature_map,
        ) = train_step1(img, bbox_deltas, bbox_labels, hyper_params)

        roi_bboxes, _ = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
        pooled_roi = RoIAlign(roi_bboxes, feature_map, hyper_params)
        roi_deltas, roi_labels = dtn_target(
            roi_bboxes, gt_boxes, gt_labels, hyper_params
        )
        dtn_reg_loss, dtn_cls_loss = train_step2(pooled_roi, roi_deltas, roi_labels)
        total_loss = rpn_reg_loss + rpn_cls_loss + dtn_reg_loss + dtn_cls_loss

        step += 1

        progress_train.set_description(
            "iteration {}/{} | rpn_reg {:.4f}, rpn_cls {:.4f}, dtn_reg {:.4f}, dtn_cls {:.4f}, loss {:.4f}".format(
                step,
                iters,
                rpn_reg_loss.numpy(),
                rpn_cls_loss.numpy(),
                dtn_reg_loss.numpy(),
                dtn_cls_loss.numpy(),
                (total_loss).numpy(),
            )
        )

        run["train/loss/rpn_reg_loss"].log(rpn_reg_loss.numpy())
        run["train/loss/rpn_cls_loss"].log(rpn_cls_loss.numpy())
        run["train/loss/dtn_reg_loss"].log(dtn_reg_loss.numpy())
        run["train/loss/dtn_cls_loss"].log(dtn_cls_loss.numpy())
        run["train/loss/total_loss"].log(total_loss.numpy())

        if step % round(one_epoch / batch_size) == 0:
            mAP = []
            progress_valid = tqdm(range(100))

            for _ in progress_valid:
                img, gt_boxes, gt_labels = next(test_set)
                rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
                roi_bboxes, roi_scores = RoIBBox(
                    rpn_reg_output, rpn_cls_output, anchors, hyper_params
                )
                pooled_roi = RoIAlign(roi_bboxes, feature_map, hyper_params)
                dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
                final_bboxes, final_labels, final_scores = Decode(
                    dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params
                )
                AP = calculate_AP_const(
                    final_bboxes,
                    final_labels,
                    gt_boxes,
                    gt_labels,
                    hyper_params,
                    mAP_threshold=0.7,
                )
                mAP.append(AP)

            mAP_res = tf.reduce_mean(mAP)
            run["validation/mAP"].log(mAP_res.numpy())

            if mAP_res.numpy() > best_mAP:
                best_mAP = mAP_res.numpy()
                ckpt_dir = f"model_ckpt/rpn_weights"
                rpn_model.save_weights(f"{ckpt_dir}/weights")
                ckpt = os.listdir(ckpt_dir)
                for i in range(len(ckpt)):
                    run[f"{ckpt_dir}/{ckpt[i]}"].upload(f"{ckpt_dir}/{ckpt[i]}")

                ckpt_dir = f"model_ckpt/dtn_weights"
                rpn_model.save_weights(f"{ckpt_dir}/weights")
                ckpt = os.listdir(ckpt_dir)
                for i in range(len(ckpt)):
                    run[f"{ckpt_dir}/{ckpt[i]}"].upload(f"{ckpt_dir}/{ckpt[i]}")

    train_time = time.time() - start_time

    rpn_model.load_weights("model_ckpt/rpn_weights/weights")
    dtn_model.load_weights("model_ckpt/dtn_weights/weights")

    total_time = []
    mAP = []
    progress_test = tqdm(range(20))

    for _ in progress_test:
        img, gt_boxes, gt_labels = next(test_set)
        start_time = time.time()
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
        roi_bboxes, roi_scores = RoIBBox(
            rpn_reg_output, rpn_cls_output, anchors, hyper_params
        )
        pooled_roi = RoIAlign(roi_bboxes, feature_map, hyper_params)
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
        final_bboxes, final_labels, final_scores = Decode(
            dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params
        )
        test_time = float(time.time() - start_time) * 1000
        AP = calculate_AP_const(
            final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params
        )
        total_time.append(test_time)
        mAP.append(AP)
        run["outputs/rpn"].log(
            neptune.types.File.as_image(draw_rpn_output(img, roi_bboxes, roi_scores, 5))
        )
        run["outputs/dtn"].log(
            neptune.types.File.as_image(
                draw_dtn_output(img, final_bboxes, labels, final_labels, final_scores)
            )
        )

    mAP_res = "%.3f" % (tf.reduce_mean(mAP))
    total_time_res = "%.2fms" % (tf.reduce_mean(total_time))
    result = {
        "mAP": mAP_res,
        "train_time": train_time,
        "inference_time": total_time_res,
    }
    run["results"] = result
