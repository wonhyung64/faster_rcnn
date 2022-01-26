#%%

#%%
def get_hyper_params():

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
                    "attempts" : 30,
                    "base_model" : "vgg16"
                    }
    return hyper_params