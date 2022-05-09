#%%
import os
from typing import Dict
from utils.variable import BASE_MODEL_LIST, DATASET_NAME_LIST

def get_hyper_params() -> Dict:
    """
    return hyper parameters dictionary

    Returns:
        Dict: hyper parameters
    """
    default_hyper_perams_dict = {
        "img_size": 500,
        "feature_map_shape": 31,
        "anchor_ratios": [1.0, 2.0, 1.0 / 2.0],
        "anchor_scales": [64, 128, 256],
        "pre_nms_topn": 6000,
        "train_nms_topn": 1500,
        "test_nms_topn": 300,
        "nms_iou_threshold": 0.7,
        "total_pos_bboxes": 128,
        "total_neg_bboxes": 128,
        "pooling_size": (7, 7),
        "variances": [0.1, 0.1, 0.2, 0.2],
        "pos_threshold": 0.65,
        "neg_threshold": 0.25,
        "batch_size": 4,
        "epochs": 150,
        "base_model": "vgg16",
        "dataset_name": "coco/2017",
        # "data_dir": "D:/won/data/tfds",
        "data_dir": "/home1/wonhyung64/data/tfds",
    }

    hyper_params_dict = {}
    
    for base_model in BASE_MODEL_LIST:
        for dataset_name in DATASET_NAME_LIST:

            hyper_params = default_hyper_perams_dict
            hyper_params["base_model"] = base_model
            hyper_params["dataset_name"] = dataset_name
            hyper_params_dict.update(
                {
                    f"{base_model}_{dataset_name}" : hyper_params
                }
            )

    return hyper_params_dict
