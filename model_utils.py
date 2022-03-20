#%%
from msilib.schema import Class
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, TimeDistributed, Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from typing import Dict, List

#%%
class RPN(Model):
    def __init__(self, hyper_params: Dict) -> None:
        """
        parameters

        Args:
            hyper_params (Dict): hyper parameters
        """
        super(RPN, self).__init__()
        self.hyper_params = hyper_params
        if hyper_params["base_model"] == "vgg16":
            self.base_model = VGG16(
                include_top=False,
                input_shape=(
                    self.hyper_params["img_size"],
                    self.hyper_params["img_size"],
                    3,
                ),
            )
        elif hyper_params["base_model"] == "vgg19":
            self.base_model = VGG19(
                include_top=False,
                input_shape=(
                    self.hyper_params["img_size"],
                    self.hyper_params["img_size"],
                    3,
                ),
            )
        self.layer = self.base_model.get_layer("block5_conv3").output

        self.feature_extractor = Model(inputs=self.base_model.input, outputs=self.layer)
        self.feature_extractor.trainable = False

        self.conv = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            name="rpn_conv",
        )

        self.rpn_cls_output = Conv2D(
            filters=self.hyper_params["anchor_count"],
            kernel_size=(1, 1),
            activation="sigmoid",
            name="rpn_cls",
        )

        self.rpn_reg_output = Conv2D(
            filters=self.hyper_params["anchor_count"] * 4,
            kernel_size=(1, 1),
            activation="linear",
            name="rpn_reg",
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> List:
        """
        batch of images pass RPN

        Args:
            inputs (tf.Tensor): batch of images

        Returns:
            List: list of RPN reg, cls, and feature map
        """
        feature_map = self.feature_extractor(inputs)
        x = self.conv(feature_map)
        rpn_reg_output = self.rpn_reg_output(x)
        rpn_cls_output = self.rpn_cls_output(x)

        return [rpn_reg_output, rpn_cls_output, feature_map]


#%%
class DTN(Model):
    def __init__(self, hyper_params: Dict) -> None:
        """
        parameters

        Args:
            hyper_params (Dict): hyper parameters
        """
        super(DTN, self).__init__()
        self.hyper_params = hyper_params
        #
        self.FC1 = TimeDistributed(Flatten(), name="frcnn_flatten")
        self.FC2 = TimeDistributed(Dense(4096, activation="relu"), name="frcnn_fc1")
        self.FC3 = TimeDistributed(Dropout(0.5), name="frcnn_dropout1")
        self.FC4 = TimeDistributed(Dense(4096, activation="relu"), name="frcnn_fc2")
        self.FC5 = TimeDistributed(Dropout(0.5), name="frcnn_dropout2")
        #
        self.cls = TimeDistributed(
            Dense(self.hyper_params["total_labels"], activation="softmax"),
            name="frcnn_cls",
        )
        self.reg = TimeDistributed(
            Dense(self.hyper_params["total_labels"] * 4, activation="linear"),
            name="frcnn_reg",
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> List:
        """
        pass detection network

        Args:
            inputs (tf.Tensor): pooled RoI

        Returns:
            List: list of detection reg, cls outputs
        """
        fc1 = self.FC1(inputs)
        fc2 = self.FC2(fc1)
        fc3 = self.FC3(fc2)
        fc4 = self.FC4(fc3)
        fc5 = self.FC5(fc4)
        dtn_reg_output = self.reg(fc5)
        dtn_cls_output = self.cls(fc5)

        return [dtn_reg_output, dtn_cls_output]
