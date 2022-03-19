#%%
import os
from typing import Dict

#%%
def get_hyper_params() -> Dict:
    """
    return hyper parameters dictionary

    Returns:
        Dict: hyper parameters
    """
    hyper_params = {
        "img_size": 500,
        "feature_map_shape": 31,
        "anchor_ratios": [1.0, 2.0, 1.0 / 2.0],
        "anchor_scales": [32, 64, 128],
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
        "batch_size": 2,
        "iters": 320000,
        "attempts": 100,
        "base_model": "vgg16",
        "mAP_threshold": 0.5,
        "dataset_name": "ship",
    }

    return hyper_params


def save_dict_to_file(dict: Dict, dict_dir: str) -> None:
    """
    save dictionary on local directory

    Args:
        dict (Dict): dictionary to save
        dict_dir (str): local directory to save
    """
    f = open(f"{dict_dir}.txt", "w")
    f.write(str(dict))
    f.close()


def generate_save_dir(atmp_dir: str) -> str:
    """
    generate local directory to save weights and outputs

    Args:
        atmp_dir (str): base local directory

    Returns:
        str: save directory
    """
    atmp_dir = f"{atmp_dir}/frcnn_atmp"
    num = 1
    while True:
        if os.path.isdir(f"{atmp_dir}/{str(num)}"):
            num += 1
        else:
            os.makedirs(f"{atmp_dir}/{str(num)}")
            print(f"Generated atmp{str(num)}")
            break

    atmp_dir = f"{atmp_dir}/{str(num)}"
    os.makedirs(f"{atmp_dir}/rpn_weights")
    os.makedirs(f"{atmp_dir}/dtn_weights")
    os.makedirs(f"{atmp_dir}/rpn_output")
    os.makedirs(f"{atmp_dir}/dtn_output")

    return atmp_dir
