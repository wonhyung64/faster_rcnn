import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, TimeDistributed, Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from typing import Dict, List
from .bbox_utils import delta_to_bbox


class RPN(Model):
    def __init__(self, args) -> None:
        """
        parameters

        Args:
            hyper_params (Dict): hyper parameters
        """
        super(RPN, self).__init__()
        self.args = args
        self.shape = args.img_size + [3]
        self.anchor_counts = len(self.args.anchor_ratios) * len(self.args.anchor_scales)
        if args.base_model == "vgg16":
            self.base_model = VGG16(
                include_top=False,
                input_shape=self.shape,
            )
        elif args.base_model == "vgg19":
            self.base_model = VGG19(
                include_top=False,
                input_shape=self.shape,
            )
        self.layer = self.base_model.get_layer("block5_conv3").output

        self.feature_extractor = Model(inputs=self.base_model.input, outputs=self.layer)
        self.feature_extractor.trainable = False

        self.conv = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            name="rpn_conv",
        )

        self.rpn_cls_output = Conv2D(
            filters=self.anchor_counts,
            kernel_size=(1, 1),
            activation="sigmoid",
            name="rpn_cls",
        )

        self.rpn_reg_output = Conv2D(
            filters=self.anchor_counts * 4,
            kernel_size=(1, 1),
            activation="linear",
            name="rpn_reg",
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> List:
        """
        batch of images pass RPN

        Args:
            inputs (tf.Tensor): batch of images

        Returns:
            List: list of RPN reg, cls, and feature map
        """
        feature_map = self.feature_extractor(inputs)
        x = self.conv(feature_map)
        rpn_reg_output = self.rpn_reg_output(x)
        rpn_cls_output = self.rpn_cls_output(x)

        return [rpn_reg_output, rpn_cls_output, feature_map]


class DTN(Model):
    def __init__(self, args, total_labels) -> None:
        """
        parameters

        Args:
            hyper_params (Dict): hyper parameters
        """
        super(DTN, self).__init__()
        self.args = args
        self.total_labels = total_labels
        #
        self.FC1 = TimeDistributed(Flatten(), name="frcnn_flatten")
        self.FC2 = TimeDistributed(Dense(4096, activation="relu"), name="frcnn_fc1")
        self.FC3 = TimeDistributed(Dropout(0.5), name="frcnn_dropout1")
        self.FC4 = TimeDistributed(Dense(4096, activation="relu"), name="frcnn_fc2")
        self.FC5 = TimeDistributed(Dropout(0.5), name="frcnn_dropout2")
        #
        self.cls = TimeDistributed(
            Dense(self.total_labels, activation="softmax"),
            name="frcnn_cls",
        )
        self.reg = TimeDistributed(
            Dense(self.total_labels * 4, activation="linear"),
            name="frcnn_reg",
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> List:
        """
        pass detection network

        Args:
            inputs (tf.Tensor): pooled RoI

        Returns:
            List: list of detection reg, cls outputs
        """
        fc1 = self.FC1(inputs)
        fc2 = self.FC2(fc1)
        fc3 = self.FC3(fc2)
        fc4 = self.FC4(fc3)
        fc5 = self.FC5(fc4)
        dtn_reg_output = self.reg(fc5)
        dtn_cls_output = self.cls(fc5)

        return [dtn_reg_output, dtn_cls_output]


def build_models(args, total_labels):
    rpn_model = RPN(args)
    input_shape = [None] + args.img_size + [3]
    rpn_model.build(input_shape)

    dtn_model = DTN(args, total_labels)
    input_shape = [None, args.train_nms_topn, 7, 7, 512]
    dtn_model.build(input_shape)

    return rpn_model, dtn_model


def Decode(
    dtn_reg_output,
    dtn_cls_output,
    roi_bboxes,
    args,
    total_labels,
    max_total_size=200,
    score_threshold=0.7,
    iou_threshold=0.5,
):
    # batch_size = args.batch_size
    variances = args.variances

    dtn_reg_output = tf.reshape(dtn_reg_output, (1, -1, total_labels, 4))
    dtn_reg_output *= variances

    expanded_roi_bboxes = tf.tile(
        tf.expand_dims(roi_bboxes, -2), (1, 1, total_labels, 1)
    )

    pred_bboxes = delta_to_bbox(expanded_roi_bboxes, dtn_reg_output)

    pred_labels_map = tf.expand_dims(tf.argmax(dtn_cls_output, -1), -1)
    pred_labels = tf.where(
        tf.not_equal(pred_labels_map, 0), dtn_cls_output, tf.zeros_like(dtn_cls_output)
    )

    final_bboxes, final_scores, final_labels, _ = tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        max_output_size_per_class=max_total_size,
        max_total_size=max_total_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )

    return final_bboxes, final_labels, final_scores


def RoIBBox(
    rpn_reg_output,
    rpn_cls_output,
    anchors,
    args,
    nms_iou_threshold=0.7,
    test=False,
):
    pre_nms_topn = args.pre_nms_topn
    post_nms_topn = args.train_nms_topn
    if test == True:
        post_nms_topn = args.test_nms_topn
    variances = args.variances
    total_anchors = (
        args.feature_map_shape[0]
        * args.feature_map_shape[1]
        * len(args.anchor_ratios)
        * len(args.anchor_scales)
    )
    batch_size = tf.shape(rpn_reg_output)[0]

    rpn_reg_output = tf.reshape(rpn_reg_output, (batch_size, total_anchors, 4))
    rpn_cls_output = tf.reshape(rpn_cls_output, (batch_size, total_anchors))

    rpn_reg_output *= variances

    rpn_bboxes = delta_to_bbox(anchors, rpn_reg_output)

    _, pre_indices = tf.nn.top_k(rpn_cls_output, pre_nms_topn)

    pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
    pre_roi_probs = tf.gather(rpn_cls_output, pre_indices, batch_dims=1)

    pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
    pre_roi_probs = tf.reshape(pre_roi_probs, (batch_size, pre_nms_topn, 1))

    roi_bboxes, roi_scores, _, _ = tf.image.combined_non_max_suppression(
        pre_roi_bboxes,
        pre_roi_probs,
        max_output_size_per_class=post_nms_topn,
        max_total_size=post_nms_topn,
        iou_threshold=nms_iou_threshold,
    )

    return roi_bboxes, roi_scores


def RoIAlign(roi_bboxes, feature_map, args):
    pooling_size = args.pooling_size
    batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]

    row_size = batch_size * total_bboxes

    pooling_bbox_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes)
    )
    pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1,))
    pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))

    pooled_roi = tf.image.crop_and_resize(
        feature_map, pooling_bboxes, pooling_bbox_indices, pooling_size
    )

    pooled_roi = tf.reshape(
        pooled_roi,
        (
            batch_size,
            total_bboxes,
            pooled_roi.shape[1],
            pooled_roi.shape[2],
            pooled_roi.shape[3],
        ),
    )

    return pooled_roi
