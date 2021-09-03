#%%
import tensorflow as tf
import math
from utils import bbox_utils
RPN = {
    "vgg16" : {
        "img_size": 500, 
        "feature_map_shape" : 31,
        "anchor_ratios" : [1., 2., 1./2.],
        "anchor_scales" : [128, 256, 512],
    },
    "mobilenet_v2":{
        "img_size": 500,
        "feature_map_shape": 32,
        "anchor_ratis": [1., 2., 1./2.],
        "anchor_scales": [128, 256, 512],
    }
}

def get_hyper_params(backbone, **kwargs):
    
    hyper_params = RPN[backbone]
    hyper_params["pre_nms_topn"] = 6000
    hyper_params["train_nms_topn"] = 1500
    hyper_params["test_nms_topn"] = 300
    hyper_params["nms_iou_threshold"] = 0.7
    hyper_params["total_pos_bboxes"] = 128
    hyper_params["total_neg_bboxes"] = 128
    hyper_params["pooling_size"] = (7,7)
    hyper_params["variances"] = [0.1, 0.1, 0.2, 0.2]
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value
    hyper_params["anchor_count"] = len(hyper_params["anchor_ratios"]) * len(hyper_params["anchor_scales"])
    return hyper_params