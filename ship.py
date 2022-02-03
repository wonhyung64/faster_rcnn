#%%
import os 
import json
import numpy as np
import xml.etree.ElementTree as elemTree
import re
import tensorflow as tf
from PIL import Image

#%%
def serialize_example(dic):
    image = dic["image"].tobytes()
    image_shape = np.array(dic["image_shape"]).tobytes()
    bbox = dic["bbox"].tobytes()
    bbox_shape = np.array(dic["bbox_shape"]).tobytes()
    label = dic["label"].tobytes()
    filename = dic["filename"].tobytes()

    dic = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'image_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_shape])),
        'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox])),
        'bbox_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_shape])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
    })) 
    return dic.SerializeToString()

#%%
def deserialize_example(serialized_string):
    image_feature_description = { 
        'image': tf.io.FixedLenFeature([], tf.string), 
        'image_shape': tf.io.FixedLenFeature([], tf.string), 
        'bbox': tf.io.FixedLenFeature([], tf.string), 
        'bbox_shape': tf.io.FixedLenFeature([], tf.string), 
        'label': tf.io.FixedLenFeature([], tf.string), 
        'filename': tf.io.FixedLenFeature([], tf.string),
    } 
    example = tf.io.parse_single_example(serialized_string, image_feature_description) 

    image = tf.io.decode_raw(example["image"], tf.float32)
    image_shape = tf.io.decode_raw(example["image_shape"], tf.int32)
    bbox = tf.io.decode_raw(example["bbox"], tf.float32)
    bbox_shape = tf.io.decode_raw(example["bbox_shape"], tf.int32)
    label = tf.io.decode_raw(example["label"], tf.int32) 
    filename = tf.io.decode_raw(example["filename"], tf.int32)

    image = tf.reshape(image, image_shape)
    bbox = tf.reshape(bbox, bbox_shape)
    
    return image, bbox, label, filename

#%%
def save_dict_to_file(dic,dict_dir):
    f = open(dict_dir + '/label_dict.txt', 'w')
    f.write(str(dic))
    f.close()

#%%
def read_labels(label_dir):
    f = open(f"{label_dir}/labels.txt", "r")
    labels = f.read().split(",")
    del labels[-1]
    return labels

#%%
def fetch_dataset(dataset, split, img_size, file_dir="C:/won/data", save_dir="D:/won/data"):
    save_dir = f"{save_dir}/{dataset}_tfrecord_{img_size[0]}_{img_size[1]}"

    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)
        try_num = 0
        label_dict = {}
        filename_lst = []
        file_dir1 = file_dir

        writer1 = tf.io.TFRecordWriter(f'{save_dir}/train.tfrecord'.encode("utf-8"))
        writer2 = tf.io.TFRecordWriter(f'{save_dir}/test.tfrecord'.encode("utf-8"))

        file_dir2 = file_dir1 + "/ship_detection/train/남해_여수항1구역_BOX"
        file_dir2_conts = os.listdir(file_dir2)
        if ".DS_Store" in file_dir2_conts: file_dir2_conts.remove(".DS_Store")

        for i in range(len(file_dir2_conts)):
            file_dir3 = file_dir2 + "/" + file_dir2_conts[i]
            file_dir3_conts = os.listdir(file_dir3)
            if ".DS_Store" in file_dir3_conts: file_dir3_conts.remove(".DS_Store")

            for j in range(len(file_dir3_conts)):
                file_dir4 = file_dir3 + "/" + file_dir3_conts[j]
                file_dir4_conts = os.listdir(file_dir4)
                if ".DS_Store" in file_dir4_conts: file_dir4_conts.remove(".DS_Store")
                filename_lst = list(set([file_dir4_conts[l][:25] for l in range(len(file_dir4_conts))]))

                for k in range(len(filename_lst)):
                    try_num += 1
                    if try_num % 3 == 0:
                        print(try_num)
                        file_dir5 = file_dir4 + "/" + filename_lst[k]
                        filename = re.sub(r'[^0-9]', '', filename_lst[k])
                        filename_lst.append(filename)

                        #jpg
                        img_ = Image.open(file_dir5 + ".jpg")
                        img_ = tf.convert_to_tensor(np.array(img_, dtype=np.int32)) / 255 # image
                        img_ = tf.image.resize(img_, img_size)
                        img = np.array(img_)

                        #xml
                        tree = elemTree.parse(file_dir5 + ".xml")
                        root = tree.getroot()
                        bboxes_ = []
                        labels_ = []
                        for x in root:
                            # print(x.tag)
                            if x.tag == "object":
                                for y in x:
                                    # print("--", y.tag)
                                    if y.tag == "bndbox":
                                        bbox_ = [int(z.text) for z in y] 
                                        bbox = [bbox_[0] / 2160, bbox_[1] / 3840, bbox_[2] / 2160, bbox_[3] / 3840]
                                        # print("----", bbox)
                                        bboxes_.append(bbox)
                                    if y.tag == "category_id":
                                        # print("----", y.text)
                                        label = int(y.text)
                                        labels_.append(label)
                                    if y.tag == "name": 
                                        label_dict[str(label)] = y.text
                        bboxes = np.array(bboxes_, dtype=np.float32)
                        labels = np.array(labels_, dtype=np.int32)

                        #json
                        with open(file_dir5 + "_meta.json", "r", encoding="UTF8") as st_json:
                            st_python = json.load(st_json)
                        st_python["Date"]
                        time = st_python["Date"][11:-1]
                        weather = st_python["Weather"]
                        season = st_python["Season"]

                        #to_dictionary
                        dic = {
                            "image":img,
                            "image_shape":img.shape,
                            "bbox":bboxes,
                            "bbox_shape":bboxes.shape,
                            "label":labels,
                            "filename":np.array(filename)
                        }

                        info_ = {
                            "time":time,
                            "weather":weather,
                            "season":season,
                        }

                        info = np.array([info_])
                        if try_num % 50 == 0 : writer2.write(serialize_example(dic))
                        else:writer1.write(serialize_example(dic))

                        if os.path.isdir(save_dir + "/meta") == False: os.mkdir(save_dir + "/meta")

                        info_dir = save_dir + "/meta/" + filename
                        np.save(info_dir + ".npy", info, allow_pickle=True)

        save_dict_to_file(label_dict, save_dir)

    dataset = tf.data.TFRecordDataset(f"{save_dir}/{split}.tfrecord".encode("utf-8")).map(deserialize_example)
    labels = read_labels(save_dir)

    return dataset, labels

# %%