import os
import time
import tensorflow as tf
import neptune.new as neptune
from tqdm import tqdm
from utils import (
    build_args,
    RoIBBox,
    RoIAlign,
    Decode,
    draw_rpn_output,
    draw_dtn_output,
    calculate_ap,
    calculate_ap_const,
    plugin_neptune,
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
    load_dataset,
    build_dataset,
    build_anchors,
    build_models,
    build_optimizer,
    build_rpn_target,
    build_dtn_target,
    forward_backward_rpn,
    forward_backward_dtn,
    record_train_loss,
    record_result,
)


def main():
    args = build_args()
    os.makedirs("./data_chkr", exist_ok=True)
    run = plugin_neptune(NEPTUNE_API_KEY, NEPTUNE_PROJECT, args)

    experiment_name = run.get_run_url().split("/")[-1].replace("-", "_")
    experiment_dir = "./model_weights/experiment"
    os.makedirs(experiment_dir, exist_ok=True)
    weights_dir = f"{experiment_dir}/{experiment_name}"

    datasets, labels, train_num, test_num = load_dataset(name=args.name, data_dir=args.data_dir)
    train_set, valid_set, test_set = build_dataset(datasets, args.batch_size, args.img_size)
    anchors = build_anchors(args)

    rpn_model, dtn_model = build_models(args, len(labels))
    optimizer1, optimizer2 = build_optimizer(args.batch_size, train_num)

    train_time = train(run, args, train_num, train_set, valid_set, labels, anchors, rpn_model, dtn_model, optimizer1, optimizer2, weights_dir)

    rpn_model.load_weights(f"{weights_dir}_rpn.h5")
    dtn_model.load_weights(f"{weights_dir}_dtn.h5")

    mean_ap, mean_test_time = test(run, test_num, test_set, rpn_model, dtn_model, labels, anchors, args)

    record_result(run, weights_dir, train_time, mean_ap, mean_test_time)


def train(
    run,
    args,
    train_num,
    train_set,
    valid_set,
    labels,
    anchors,
    rpn_model,
    dtn_model,
    optimizer1,
    optimizer2,
    weights_dir,
    ):
    best_mean_ap = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_progress = tqdm(range(train_num//args.batch_size))
        for _ in epoch_progress:
            image, gt_boxes, gt_labels = next(train_set)

            true_rpn = build_rpn_target(anchors, gt_boxes, gt_labels, args)
            loss_rpn, rpn_reg_output, rpn_cls_output, feature_map = forward_backward_rpn(image, true_rpn, rpn_model, optimizer1, args.batch_size, args.feature_map_shape, args.anchor_ratios, args.anchor_scales, args.total_pos_bboxes, args.total_neg_bboxes)

            roi_bboxes, _ = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, args)
            pooled_roi = RoIAlign(roi_bboxes, feature_map, args)

            true_dtn = build_dtn_target(roi_bboxes, gt_boxes, gt_labels, len(labels), args)
            loss_dtn = forward_backward_dtn(pooled_roi, true_dtn, dtn_model, optimizer2, len(labels), args.batch_size, args.train_nms_topn)

            total_loss = tf.reduce_sum(loss_rpn + loss_dtn)
            record_train_loss(run, loss_rpn, loss_dtn, total_loss)

            epoch_progress.set_description(
                "Epoch {}/{} | rpn_reg {:.4f}, rpn_cls {:.4f}, dtn_reg {:.4f}, dtn_cls {:.4f}, total {:.4f}".format(
                    epoch+1,
                    args.epochs,
                    loss_rpn[0].numpy(),
                    loss_rpn[1].numpy(),
                    loss_dtn[0].numpy(),
                    loss_dtn[1].numpy(),
                    total_loss.numpy(),
                )
            )
        mean_ap = validation(valid_set, rpn_model, dtn_model, labels, anchors, args)

        run["validation/mAP"].log(mean_ap.numpy())

        if mean_ap.numpy() > best_mean_ap:
            best_mean_ap = mean_ap.numpy()
            rpn_model.save_weights(f"{weights_dir}_rpn.h5")
            dtn_model.save_weights(f"{weights_dir}_dtn.h5")

    train_time = time.time() - start_time

    return train_time


def validation(valid_set, rpn_model, dtn_model, labels, anchors, args):
    aps = []
    validation_progress = tqdm(range(100))
    for _ in validation_progress:
        image, gt_boxes, gt_labels = next(valid_set)
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(image)
        roi_bboxes, _ = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, args)
        pooled_roi = RoIAlign(roi_bboxes, feature_map, args)
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
        final_bboxes, final_labels, final_scores = Decode(
            dtn_reg_output, dtn_cls_output, roi_bboxes, args, len(labels)
        )

        ap = calculate_ap_const(final_bboxes, final_labels, gt_boxes, gt_labels, len(labels))
        validation_progress.set_description("Validation | Average_Precision {:.4f}".format(ap))
        aps.append(ap)

    mean_ap = tf.reduce_mean(aps)

    return mean_ap


def test(run, test_num, test_set, rpn_model, dtn_model, labels, anchors, args):
    test_times = []
    aps = []
    test_progress = tqdm(range(test_num))
    for step in test_progress:
        image, gt_boxes, gt_labels = next(test_set)
        start_time = time.time()
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(image)
        roi_bboxes, roi_scores = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, args)
        pooled_roi = RoIAlign(roi_bboxes, feature_map, args)
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
        final_bboxes, final_labels, final_scores = Decode(
            dtn_reg_output, dtn_cls_output, roi_bboxes, args, len(labels)
        )
        test_time = time.time() - start_time

        ap = calculate_ap_const(final_bboxes, final_labels, gt_boxes, gt_labels, len(labels))
        test_progress.set_description("Test | Average_Precision {:.4f}".format(ap))
        aps.append(ap)
        test_times.append(test_time)

        if step <= 20 == 0:
            run["outputs/rpn"].log(
                neptune.types.File.as_image(draw_rpn_output(image, roi_bboxes, roi_scores, 5))
            )
            run["outputs/dtn"].log(
                neptune.types.File.as_image(
                    draw_dtn_output(image, final_bboxes, labels, final_labels, final_scores)
                )
            )

    mean_ap = tf.reduce_mean(aps)
    mean_test_time = tf.reduce_mean(test_times)

    return mean_ap, mean_test_time


if __name__ == "__main__":
    main()
