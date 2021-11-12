#%%
import os
import numpy as np
import tensorflow_datasets as tfds
#%%

data_dir = r"C:\won\data\pascal_voc"
#
train_data, dataset_info = tfds.load("voc/2007", split="train", data_dir = data_dir, with_info=True)

val_data, _ = tfds.load("voc/2007", split="validation", data_dir = data_dir, with_info=True)

test_data, _ = tfds.load("voc/2007", split="test", data_dir = data_dir, with_info=True)
#
#
#
#%%
info = dict()
info['labels'] = dataset_info.features['labels'].names
info['dataset_columns'] = ['image','bbox','label','is_difficult']

info['dataset_columns']
info['train_filename'] = []
info['val_filename'] = []
info['test_filename'] = []

#%%
for i in tfds.as_numpy(train_data):
    tmp = [] 
    info['train_filename'].append(str(i['image/filename'])[2:-5])
    tmp.append(i['image'])
    tmp.append(i['objects']['bbox'])
    tmp.append(i['objects']['label'])
    tmp.append(i['objects']['is_difficult'])
    tmp = np.array(tmp)
    np.save((r"C:\won\data\pascal_voc\voc2007_np\train_val\\"+str(i['image/filename'])[2:-5] + ".npy"), tmp, allow_pickle=True)

#%%
for i in tfds.as_numpy(val_data):
    tmp = []
    info['val_filename'].append(str(i['image/filename'])[2:-5])
    tmp.append(i['image'])
    tmp.append(i['objects']['bbox'])
    tmp.append(i['objects']['label'])
    tmp.append(i['objects']['is_difficult'])
    tmp = np.array(tmp)
    np.save((r"C:\won\data\pascal_voc\voc2007_np\train_val\\"+str(i['image/filename'])[2:-5] + ".npy"), tmp, allow_pickle=True)

#%%
for i in tfds.as_numpy(test_data):
    tmp = []
    info['test_filename'].append(str(i['image/filename'])[2:-5])
    tmp.append(i['image'])
    tmp.append(i['objects']['bbox'])
    tmp.append(i['objects']['label'])
    tmp.append(i['objects']['is_difficult'])
    tmp = np.array(tmp)
    np.save((r"C:\won\data\pascal_voc\voc2007_np\test\\"+str(i['image/filename'])[2:-5] + ".npy"), tmp, allow_pickle=True)

#%%
info_numpy = np.array([info])
#%%
save_dir = data_dir + r'\voc2007_np'
os.makedirs(save_dir, exist_ok=True)
np.save(save_dir + r'\info.npy', info_numpy, allow_pickle=True)

# %%
