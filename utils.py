#%%
import os
#%%
def get_hyper_params():

    hyper_params = {"img_size": 500,
                    "feature_map_shape": 31,
                    "anchor_ratios": [1., 2., 1./2.],
                    "anchor_scales": [32, 64, 128],
                    "pre_nms_topn": 6000,
                    "train_nms_topn": 1500,
                    "test_nms_topn": 300,
                    "nms_iou_threshold": 0.7,
                    "total_pos_bboxes": 128,
                    "total_neg_bboxes": 128,
                    "pooling_size": (7,7),
                    "variances": [0.1, 0.1, 0.2, 0.2],
                    "pos_threshold" : 0.65,
                    "neg_threshold" : 0.25,
                    "batch_size" : 2,
                    "iters" : 320000,
                    "attempts" : 100,
                    "base_model" : "vgg16",
                    "mAP_threshold" : 0.5,
                    "dataset_name" : "ship"
                    }
    return hyper_params
#%%
def save_dict_to_file(dic,dict_dir):
    f = open(dict_dir + '.txt', 'w')
    f.write(str(dic))
    f.close()
#%%
def generate_save_dir(atmp_dir, hyper_params):
    atmp_dir = atmp_dir + '/frcnn_atmp'

    i = 1
    tmp = True
    while tmp :
        if os.path.isdir(atmp_dir + '/' + str(i)) : 
            i+= 1
        else: 
            os.makedirs(atmp_dir + '/' + str(i))
            print("Generated atmp" + str(i))
            tmp = False
    atmp_dir = atmp_dir + '/' + str(i)

    os.makedirs(atmp_dir + '/rpn_weights')
    os.makedirs(atmp_dir + '/dtn_weights')
    os.makedirs(atmp_dir + '/rpn_output')
    os.makedirs(atmp_dir + '/dtn_output')

    return atmp_dir