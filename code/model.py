from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input


def conv_block(inputs, filters):
    filter1, filter2 = filters
    block = Conv2D(filters=filter1, kernel_size=(1, 1), strides=1)(inputs)
    block = BatchNormalization(axis=-1)(block)
    block = Activation("relu")(block)

    block = Conv2D(filters=filter1, kernel_size=(3, 3), strides=1,padding='same')(block)
    block = BatchNormalization(axis=-1)(block)
    block = Activation("relu")(block)

    block = Conv2D(filters=filter2, kernel_size=(1, 1), strides=1)(block)
    block = BatchNormalization(axis=-1)(block)
    block = Activation("relu")(block)

    block1 = Conv2D(filters=filter2,kernel_size=(1,1), strides=1)(inputs)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation("relu")(block1)

    final_block=Add()([block, block1])
    final_block=Activation("relu")(final_block)

    return final_block


def identity_block(inputs, filters):
    filter1, filter2 = filters
    block = Conv2D(filters=filter1, kernel_size=(1, 1), strides=1)(inputs)
    block = BatchNormalization(axis=-1)(block)
    block = Activation("relu")(block)

    block = Conv2D(filters=filter1, kernel_size=(3, 3), strides=1, padding='same')(block)
    block = BatchNormalization(axis=-1)(block)
    block = Activation("relu")(block)

    block = Conv2D(filters=filter2, kernel_size=(1, 1), strides=1)(block)
    block = BatchNormalization(axis=-1)(block)
    block = Activation("relu")(block)

    final_block = Add()([block, inputs])
    final_block = Activation("relu")(final_block)

    return final_block


def create_model_resnet50(input_shape, out_shape):
    inputs = Input(input_shape)
    model = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")(inputs)


    model = BatchNormalization(axis = -1)(model)
    model = Activation("relu")(model)
    model = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(model)

    model = conv_block(model, [64, 256])
    model = identity_block(model, [64, 256])
    model = identity_block(model, [64, 256])

    model = conv_block(model, [128, 512])
    model = identity_block(model, [128, 512])
    model = identity_block(model, [128, 512])
    model = identity_block(model, [128, 512])

    model = conv_block(model, [256, 1024])
    model = identity_block(model, [256, 1024])
    model = identity_block(model, [256, 1024])
    model = identity_block(model, [256, 1024])
    model = identity_block(model, [256, 1024])
    model = identity_block(model, [256, 1024])

    model = conv_block(model, [512, 2048])
    model = identity_block(model, [512, 2048])
    model = identity_block(model, [512, 2048])

    model = AveragePooling2D(pool_size=(7, 7), strides=1)(model)
    model = Flatten()(model)
    model = Dense(out_shape)(model)
    model = Activation("softmax")(model)
    #model = Activation("sigmoid")(model)
    final_model = Model(inputs=inputs, outputs=model)
    return final_model
