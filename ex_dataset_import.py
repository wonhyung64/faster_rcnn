#%%
import os
import sys
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

image_path = 'C:\won\data\pascal_voc\VOCdevkit\VOC2007\JPEGImages\\000005.jpg'
image = Image.open(image_path).convert('RGB')

plt.figure(figsize=(25,20))
plt.imshow(image)
plt.show()
plt.close()

#%%
import sys 
import os
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree

xml_path = 'C:\won\data\pascal_voc\VOCdevkit\VOC2007\Annotations\\000005.xml'

print("XML parsing Start\n")
xml = open(xml_path, "r")
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
    
    print('class : {}\nxmin : {}\nymin : {}\nxmax : {}\nymax : {}\n'.format(name, xmin, ymin, xmax, ymax))

print('XML parsing END')

# %%
