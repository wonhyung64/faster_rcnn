#%%
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential

#%%

def get_model(hyper_params):

    """Generating rpn model 

    Args:
        hyper_params (dictionary): key - "img_size", "anchor_count"

    Returns:
        rpn_model (model) : rpn model 
        feature_extractor (layer) : block5_conv3 layer of VGG16
    """

    img_size = hyper_params['img_size']
    base_model = VGG16(include_top=False, input_shape=(img_size, img_size, 3))
    # include_top ?
    # input shape : (img_size) x  (img_size) x 3channel

    feature_extractor = base_model.get_layer("block5_conv3")
    # VGG16의 마지막 합성곱 층인 "blcok5_conv3" 를 feature_extractor에 할당
    
    output = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="rpn_conv")(feature_extractor.output)
    # output의 차원이 512 ? 직관적으로 이해 안 됨... 
    
    rpn_cls_output = Conv2D(filters=hyper_params["anchor_count"], kernel_size=(1,1), activation="sigmoid", name="rpn_cls")(output)
    # rpn model 에서 나오는 classification layer
    
    rpn_reg_output = Conv2D(filters=hyper_params["anchor_cout"]*4, kernel_size=(1,1), activation="linear", name="rpn_reg")(output)
    # rpn model 에서 나오는 regression layer
    
    rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])
    # rpn model 구축
    
    return rpn_model, feature_extractor
    # feature_extractor 를 왜 return ?
    
#%%
def init_model(model):
    
    model(tf.random.uniform((1, 500, 500, 3)))
    # Initializing model with dummy data for load weights with optimizer state and also graph construction
    # 이라는데 무슨말인지 모름...