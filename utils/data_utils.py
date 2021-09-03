#%%
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np

#%%
def get_dataset(name, split, data_dir = "~/tensorflow_datasets"):
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    