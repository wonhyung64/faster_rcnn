#%%
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dropout, Dense, TimeDistributed
from tensorflow.python.keras.backend import softmax

from tqdm import tqdm
import time
#%%
info_dir = r"C:\won\data\pascal_voc\voc2007_np"
info = np.load(info_dir + r"\info.npy", allow_pickle=True)

labels = info[0]["labels"]
train_filename = info[0]['train_filename'] + info[0]['val_filename']
test_filename = info[0]['test_filename']

train_total_items = len(train_filename)
test_total_items = len(test_filename)

labels = ["bg"] + labels
# %%
base_model = VGG19()

backbone = Model(inputs=base_model.input, outputs=base_model.get_layer("block5_pool").output)
backbone.trainable=False

FC1 = Flatten(name='frcnn_flatten')(backbone.output)
FC2 = Dense(4096, activation='relu', name='frcnn_fc1')(FC1)
FC3 = Dropout(0.5, name='frcnn_dropout1')(FC2)
FC4 = Dense(4096, activation='relu', name='frcnn_fc2')(FC3)
FC5 = Dropout(0.5, name='frcnn_dropout2')(FC4)
cls = Dense(21, activation='softmax', name='frcnn_cls')(FC5)

my_model = Model(inputs=backbone.input, outputs=cls)
my_model.load_weights(r'C:\won\frcnn\compare_weights\weights')
# my_model.summary()
#%%
# backbone2 = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)

# predict = Dense(21, activation="softmax", name="vgg19_cls")(backbone2.output)
# compare_model = Model(inputs=base_model.input, outputs=predict)
# compare_model.load_weights(r'C:\won\frcnn\compare_weights\weights')
#%%
optimizer = tf.keras.optimizers.Adam(1e-5)

def softmax_loss(pred, true):
    return -tf.math.reduce_sum(true * tf.math.log(pred + 1e-7) )

def sigmoid_loss(pred, true):
    return -tf.reduce_sum(true * tf.math.log(pred + 1e-7) + (1-true) * tf.math.log(1 - pred + 1e-7))

#%%
@tf.function
def train_step(images, true):
    with tf.GradientTape(persistent=True) as tape:
        pred = my_model(images)
        loss = softmax_loss(pred, true)
        
    grads = tape.gradient(loss, my_model.trainable_weights)

    optimizer.apply_gradients(zip(grads, my_model.trainable_weights))

    return loss
#%% train compare model

#%%

train_dir = r"C:\won\data\pascal_voc\voc_2007_roi_atmp11\\"
step = 0
batch = 8
iter = 20010

progress_bar = tqdm(range(iter))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, iter))

start_time = time.time()
for _ in progress_bar:
    
    img_arr = []
    label_arr = []
    for i in range(batch):
        filename = train_filename[int(np.random.random_integers(0,5010, 1))]
        img_index = int(np.random.random_integers(0,9,1))

        img_dir = train_dir + filename + "\\" + filename + "_" + str(img_index) + ".npy"
        label_dir = train_dir + filename + "\\" + filename + "_label" + ".npy"
        img_ = np.load(img_dir)
        img_ = tf.convert_to_tensor(img_)
        img_ = tf.image.resize(img_, (224,224))
        img_arr.append(img_)

        label = np.load(label_dir)[0][img_index]        
        label_arr.append(label)

    images = tf.convert_to_tensor(img_arr)
    images_labels = tf.convert_to_tensor(label_arr)

    true_ = tf.one_hot(images_labels, 21)
    true_ = tf.reshape(true_, shape=(batch, 21))
    true = tf.cast(true_, dtype=tf.float32)

    loss = train_step(images, true)

    step += 1
    
    progress_bar.set_description('iteration {}/{} | loss {:.4f}'.format(
        step, iter, loss.numpy()
    )) 
    
    if step % 500 == 0:
        print(progress_bar.set_description('iteration {}/{} | loss {:.4f}'.format(
            step, iter, loss.numpy()
        )))
    if step % 10000 == 0:
        my_model.save_weights(r'C:\won\frcnn\my_weights\weights')
        print("weights saved")
# print("Time taken: %.2fs" % (time.time() - start_time))


# compare_model.save_weights(r'C:\won\frcnn\compare_weights\weights')

#%% test
# print(tf.argmax(compare_model(images), axis=1, output_type=tf.int32))
# print(tf.argmax(true, axis=1, output_type=tf.int32))


print(tf.argmax(my_model(images), axis=1, output_type=tf.int32))
print(tf.argmax(true, axis=1, output_type=tf.int32))
#%%


