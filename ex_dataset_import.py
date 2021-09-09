# %%
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree

data_path = "E:\Data\pascal_voc\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007"
os.chdir(data_path)

IMAGE_FOLDER = "JPEGImages"
ANNOTATIONS_FOLDER = "Annotations"

ann_root, ann_dir, ann_files = next(os.walk(os.path.join(os.getcwd(), ANNOTATIONS_FOLDER)))
img_root, img_dir, img_files = next(os.walk(os.path.join(os.getcwd(), IMAGE_FOLDER)))

for xml_file in ann_files:
    
    img_name = img_files[img_files.index('.'.join([xml_file.split('.')[0], 'jpg']))]
    img_file = os.path.join(img_root, img_name)
    img = tf.io.read_file(img_file) 
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img = img.numpy()

    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()
    
    size = root.find("size")
    
    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    print("Image properties\nwidth : {}\nheight  {}\nchannels : {}\n".format(width, height, channels))

    objects = root.findall('object')
    print('Objects Description')

    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
