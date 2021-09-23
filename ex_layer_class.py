#%%
import tensorflow as tf
from tensorflow import keras

# %%
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype='float32'),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype='float32'), trainable=True
        )
    
    def call(self,inputs):
        return tf.matmul(inputs, self.w) + self.b
# %%
x = tf.ones((2,2))
linear_layer = Linear(4,2)
y = linear_layer(x)
print(y)

#%%
assert linear_layer.weights == [linear_layer.w, linear_layer.b]

#%%
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units, ), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(units,), initializer='zeros', trainable=True)
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
# %%
x = tf.ones((2,2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
# %%

class ComputeSum(keras.layers.Layer):
    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim, )), trainable=False)
        
    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total
    
# %%
x = tf.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())


# %%
print("weights:", len(my_sum.weights))
print("non_trainable weights:", len(my_sum.non_trainable_weights))
print("trainable_weights:", my_sum.trainable_weights)

# %%
class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
# %%
linear_layer = Linear(32)
y = linear_layer(x)
print(y)

#%%
class MLPBlock(keras.layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)
        
    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)
    

# %%
mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))
print("weights:", len(mlp.weights))
print("trainable weights:", len(mlp.trainable_weights))
# %%
class ActivityRegularizationLayer(keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate
        
    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

#%%
class OuterLayer(keras.layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)
        
    def call(self, inputs):
        return self.activity_reg(inputs)
# %%
layer = OuterLayer()
assert len(layer.losses) == 0

_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1

_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1


# %%
class OuterLayerWithKernelRegularizer(keras.layers.Layer):
    def __init__(self):
        super(OuterLayerWithKernelRegularizer, self).__init__()
        self.dense = keras.layers.Dense(
            32, kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )
    
    def call(self, inputs):
        return self.dense(inputs)
# %%
layer = OuterLayerWithKernelRegularizer()
_ = layer(tf.zeros((1, 1)))
print(layer.losses)
# %%
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for x_batch_train, y_batch_train in train_data