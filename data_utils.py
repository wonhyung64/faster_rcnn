# %%
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

#%%
def download_dataset(dataset_name, data_dir):

    if dataset_name == "coco17":
        dataset, dataset_info = tfds.load(name="coco/2017", data_dir=data_dir, with_info=True)
        labels = dataset_info.features["objects"]["label"].names
        
    elif dataset_name == "voc07":
        dataset, dataset_info = tfds.load(name="voc/2007", data_dir=data_dir, with_info=True)
        labels = dataset_info.features["labels"].names

    elif dataset_name == "voc12":
        dataset, dataset_info = tfds.load(name="voc/2012", data_dir=data_dir, with_info=True)
        labels = dataset_info.features["labels"].names

    train = dataset["train"]
    validation = dataset["validation"]
    test = dataset["test"]


    return train, validation, test, labels
#%%
def serialize_example(example, img_size):
    image = example["image"]
    image = tf.image.resize(image, img_size)
    image_shape = image.shape
    
    image = np.array(image).tobytes()
    image_shape = np.array(image_shape).tobytes()

    bbox = example["bbox"]
    bbox_shape = bbox.shape

    bbox = np.array(bbox).tobytes()
    bbox_shape = np.array(bbox_shape).tobytes()

    label = example['label']
    label = np.array(label).tobytes()
    feature_dict={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'image_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_shape])),
        'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox])),
        'bbox_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_shape])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict)) 

    return example.SerializeToString()
#%%
def deserialize_example(serialized_string):
    image_feature_description = { 
        'image': tf.io.FixedLenFeature([], tf.string), 
        'image_shape': tf.io.FixedLenFeature([], tf.string), 
        'bbox': tf.io.FixedLenFeature([], tf.string),
        'bbox_shape': tf.io.FixedLenFeature([], tf.string), 
        'label': tf.io.FixedLenFeature([], tf.string), 
    } 

    example = tf.io.parse_single_example(serialized_string, image_feature_description) 

    image = tf.io.decode_raw(example["image"], tf.float32)
    image_shape = tf.io.decode_raw(example["image_shape"], tf.int32)
    bbox = tf.io.decode_raw(example["bbox"], tf.float32)
    bbox_shape = tf.io.decode_raw(example["bbox_shape"], tf.int32)
    label = tf.io.decode_raw(example["label"], tf.int64) 

    image = tf.reshape(image,image_shape)
    bbox = tf.reshape(bbox, bbox_shape)
    return image, bbox, label

# %%
def write_labels(save_dir, labels):
    with open(f"{save_dir}/labels.txt", "w") as f:
        for label in labels:
            f.write(label+",")

#%%
def read_labels(label_dir):
    f = open(f"{label_dir}/labels.txt", "r")
    labels = f.read().split(",")
    del labels[-1]
    return labels

#%%
def fetch_dataset(dataset, split, img_size, data_dir=r"C:\won", save_dir=r"D:\won"):

    save_dir = save_dir + r"\data\\" + dataset + "_tfrecord_" + str(img_size[0]) + "_" + str(img_size[1])
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)

        data_dir = data_dir + r"\data\tfds"
        if os.path.exists(data_dir) == False: os.mkdir(data_dir)

        train, validation, test, labels = download_dataset(dataset, data_dir)

        write_labels(save_dir, labels)

        start_time = time.time()
        for name, split_dataset in [("train", train), ("validation", validation), ("test",test)]:
            print("Fetch", name, "dataset")
            try_num = 0
            writer = tf.io.TFRecordWriter(f"{save_dir}/{name}.tfrecord".encode("utf-8"))
            for sample in split_dataset:
                example = {"image":sample["image"]/255, "bbox":sample["objects"]["bbox"], "label":sample["objects"]["label"]}
                x = serialize_example(example, img_size)
                writer.write(x)
                try_num += 1
                print(try_num)

        print("total time :", time.time() - start_time)

    datasets = tf.data.TFRecordDataset(f"{save_dir}/{split}.tfrecord".encode("utf-8")).map(deserialize_example)
    labels = read_labels(save_dir)
    return datasets, labels

