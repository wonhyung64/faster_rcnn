#%%
import tensorflow_datasets as tfds

train_data, dataset_info = tfds.load("voc/2007", split="train+validation", data_dir = "E:\Data\\tensorflow_datasets", with_info=True)
val_data, _ = tfds.load("voc/2007", split="test", data_dir = "E:\Data\\tensorflow_datasets", with_info=True)
# %%
import numpy as np

feature_map_size = 16


ratios = [0.5, 1, 2]
scales = [8, 16, 32]

base_anchor = np.array([1, 1, feature_map_size, feature_map_size]) - 1

width = base_anchor[2] - base_anchor[0] + 1
height = base_anchor[3] - base_anchor[1] + 1
x_ctr = base_anchor[0] + 0.5 * (width - 1)
y_ctr = base_anchor[1] + 0.5 * (height - 1)

size = width * height
size_ratios = size / ratios

widths = np.round(np.sqrt(size_ratios))
heights = np.round(widths * ratios)

widths = widths[:, np.newaxis]
heights = heights[:, np.newaxis]

ratio_anchors = np.hstack((x_ctr - 0.5 * (widths - 1),
                     y_ctr - 0.5 * (heights - 1),
                     x_ctr + 0.5 * (widths - 1),
                     y_ctr + 0.5 * (heights - 1)))


for i in range(ratio_anchors.shape[0]):
    width = ratio_anchors[:,2] - ratio_anchors[:,0] + 1
    height = ratio_anchors[:,3] - ratio_anchors[:,1] + 1
    x_ctr = ratio_anchors[:,0] + 0.5 * (width - 1)
    y_ctr = ratio_anchors[:,1] + 0.5 * (height - 1)
    
    widths = width * scales
    heights = height * scales

    anchors_tmp = np.hstack((x_ctr - 0.5 * (widths - 1),
                         y_ctr - 0.5 * (height - 1),
                         x_ctr + 0.5 * (widths - 1),
                         y_ctr + 0.5 * (heights - 1)))
    print(anchors_tmp)
    anchors = np.vstack(anchors_tmp)
    