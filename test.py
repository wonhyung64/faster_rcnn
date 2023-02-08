#%%
import tensorflow as tf
import os
import neptune.new as neptune
from utils import (
    build_models,
    build_anchors,
    build_args,
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
    RoIBBox,
    RoIAlign,
    Decode,
    draw_dtn_output,
    calculate_ap_const,
    load_dataset,
    build_dataset,
)
from tqdm import tqdm
import json
import numpy as np
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#%%
if __name__ == "__main__":
    args = build_args()
    run = neptune.init(
        project=NEPTUNE_PROJECT,
        api_token=NEPTUNE_API_KEY,
        run="MOD-202"
    )
    args.data_dir = "/Users/wonhyung64/data"
    args.batch_size = 1

    datasets, labels, train_num, valid_num, test_num = load_dataset(
        name=args.name, data_dir=args.data_dir
    )
    train_set, valid_set, test_set = build_dataset(
        datasets, args.batch_size, args.img_size
    )

    experiment_name = run.get_run_url().split("/")[-1].replace("-", "_")
    model_name = NEPTUNE_PROJECT.split("-")[1]
    experiment_dir = f"./model_weights/{model_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    weights_dir = f"{experiment_dir}/{experiment_name}"

    '''
    run["rpn_model"].download(f"{weights_dir}_rpn.h5")
    run["dtn_model"].download(f"{weights_dir}_dtn.h5")
    '''

    rpn_model, dtn_model = build_models(args, len(labels))
    rpn_model.load_weights(f"{weights_dir}_rpn.h5")
    dtn_model.load_weights(f"{weights_dir}_dtn.h5")
    anchors = build_anchors(args)

#%%
    test_progress = tqdm(range(train_num))
    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)
    for step in test_progress:
        # for _ in range(30):
        #     next(test_set)
        image, gt_boxes, gt_labels = next(train_set)
        
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(image)
        roi_bboxes, roi_scores = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, args)
        pooled_roi = RoIAlign(roi_bboxes, feature_map, args)
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
        # final_bboxes, final_labels, final_scores = Decode(
        #     dtn_reg_output, dtn_cls_output, roi_bboxes, args, len(labels)
        # )
        pred_bboxes, pred_labels, final_bboxes, final_labels, final_scores = Decode(
            dtn_reg_output, dtn_cls_output, roi_bboxes, args, len(labels)
        )

        sample = {
            "gt_boxes": gt_boxes[0].numpy(),
            "gt_labels": gt_labels[0].numpy(),
            "roi_bboxes": roi_bboxes[0].numpy(),
            "roi_scores": roi_scores[0].numpy(),
            "pred_bboxes": pred_bboxes[0].numpy(),
            "pred_labels": pred_labels[0].numpy(),
            "final_bboxes": final_bboxes[0].numpy(),
            "final_labels": final_labels[0].numpy(),
            "final_scores": final_scores[0].numpy()
        }

        #save
        path = "/Users/wonhyung64/data/diagnosis"
        with open(f"{path}/sample{step}.json", "w") as f:
            json.dump(sample, f, cls=NumpyEncoder)
            json.dumps(sample, cls=NumpyEncoder)

        #load
        with open(f"{path}/sample{step}.json", encoding="UTF-8") as f:
            json_load = json.load(f)
        json_load = {k: np.asarray(json_load[k]) for k in json_load.keys()}


        json.load(f"{path}/sample{step}.json")

        json_load = json.loads(json_string)

        restored = {k: np.asarray(json_load[k]) for k in json_load.keys()}

        result = draw_dtn_output(
            image, final_bboxes, labels, final_labels, final_scores, colors
        )
        ap50 = calculate_ap_const(final_bboxes, final_labels, gt_boxes, gt_labels, len(labels))
        print(ap50)
        # result
        if ap50 >= 0.4: break
        result.save(f"res_{step}.jpg")
# %%
from PIL import Image
import xml.etree.ElementTree as elemTree
import numpy as np


def extract_img(dir):
    image = Image.open(f"{dir}.jpg")
    image = tf.convert_to_tensor(image)
    img_size = tf.shape(image)
    image = tf.image.resize(image, [500,500])
    image = image / 255.
    image = tf.expand_dims(image, axis=0)
    return image, img_size


def extract_annot(dir, org_img_size):
    tree = elemTree.parse(f"{dir}.xml")
    root = tree.getroot()
    bboxes_ = []
    labels_ = []
    for x in root:
        if x.tag == "object":
            for y in x:
                if y.tag == "bndbox":
                    bbox_ = [int(z.text) for z in y]
                    bbox = [
                        bbox_[1] / org_img_size[0],
                        bbox_[0] / org_img_size[1],
                        bbox_[3] / org_img_size[0],
                        bbox_[2] / org_img_size[1],
                    ]
                    bboxes_.append(bbox)
                if y.tag == "category_id":
                    label = int(y.text)
                    labels_.append(label)
    bboxes = np.array(bboxes_, dtype=np.float32)
    labels = np.array(labels_, dtype=np.int32)
    bboxes = tf.expand_dims(tf.cast(bboxes, dtype=tf.float32), axis=0)
    labels = tf.expand_dims(tf.cast(labels, dtype=tf.int32), axis=0)

    return bboxes, labels

#%%
path = "/Users/wonhyung64/data/voucher/test"
filenames = os.listdir(path)
detector = ShipDetector(args,labels, weights_dir)
evaluate = Evaluate(labels, colors)

for f in filenames:
    f_iter = iter(filenames)
    f = next(f_iter)
    # if f == ".DS_Store": continue
    filename = f.split(".")[0]
    dir = f"{path}/{filename}"

    image, img_size = extract_img(dir)
    gt_boxes, gt_labels = extract_annot(dir, img_size)
    gt_labels = gt_labels - 1
    final_bboxes, final_labels, final_scores = detector.predict(image)
    ap_50 = evaluate.cal_ap(final_bboxes, final_labels, gt_boxes, gt_labels)
    res = evaluate.visualize(image, final_bboxes, final_labels, final_scores)
    print(ap_50)
    res

rpn_reg_output, rpn_cls_output, feature_map = rpn_model(image)
roi_bboxes, roi_scores = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, args)
pooled_roi = RoIAlign(roi_bboxes, feature_map, args)
dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
np.sum(tf.argmax(dtn_cls_output, axis=-1).numpy())
gt_boxes
final_bboxes, final_labels, final_scores = Decode(
    dtn_reg_output, dtn_cls_output, roi_bboxes, args, total_labels
)


#%%
class ShipDetector():
    def __init__(self, args, labels, weights_dir):
        self.total_labels = len(labels)
        self.rpn_model, self.dtn_model = build_models(args, self.total_labels)
        rpn_model.load_weights(f"{weights_dir}_rpn.h5")
        dtn_model.load_weights(f"{weights_dir}_dtn.h5")
        self.anchors = build_anchors(args)

    def predict(self, image):
        rpn_reg_output, rpn_cls_output, feature_map = self.rpn_model(image)
        roi_bboxes, roi_scores = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, args)
        pooled_roi = RoIAlign(roi_bboxes, feature_map, args)
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
        final_bboxes, final_labels, final_scores = Decode(
            dtn_reg_output, dtn_cls_output, roi_bboxes, args, self.total_labels
        )

        return final_bboxes, final_labels, final_scores


class Evaluate():
    def __init__(self, labels, colors):
        self.labels = labels
        self.total_labels = len(labels)
        self.colors = colors
    
    def cal_ap(self, final_bboxes, final_labels, gt_boxes, gt_labels):
        ap50 = calculate_ap_const(final_bboxes, final_labels, gt_boxes, gt_labels, self.total_labels)

        return ap50

    def visualize(self, image, final_bboxes, final_labels, final_scores):
        fig = draw_dtn_output(
            image, final_bboxes, self.labels, final_labels, final_scores, self.colors
        )

        return fig



# %%
