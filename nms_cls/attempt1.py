#%%
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from utils import bbox_utils, data_utils, hyper_params_utils


#%%
batch_size = 4

hyper_params = hyper_params_utils.get_hyper_params()

epochs = hyper_params['epochs']

train_data, dataset_info = tfds.load("voc/2007", split="train+validation", data_dir = "E:\Data\\tensorflow_datasets", with_info=True)
val_data, _ = tfds.load("voc/2007", split="test", data_dir = "E:\Data\\tensorflow_datasets", with_info=True)

train_total_items = dataset_info.splits["train"].num_examples + dataset_info.splits["validation"].num_examples
val_total_items = dataset_info.splits["test"].num_examples