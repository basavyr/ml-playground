import tensorflow as tf
import tensorflow.keras.layers as tfl


def sequential():

    # Create the Sequential model and add only the ZeroPadding2D layer
    model = tf.keras.Sequential([
        tfl.ZeroPadding2D(padding=3, input_shape=(64, 64, 3)),
        tfl.Conv2D(filters=32, kernel_size=(7, 7), strides=1, padding="valid"),
        # use batch norm along the channel axis
        tfl.BatchNormalization(axis=-1),
        tfl.ReLU(),
        # keep moving in 2x2 size within the max pool layer
        tfl.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'),
        tfl.Flatten(),
        tfl.Dense(units=1, activation="sigmoid")
    ])

    # Display the model's architecture
    model.summary()


def conv_model(input_shape):
    input_img = tf.keras.Input(shape=input_shape)

    # can be considered as Z0, A0, or even P0
    A0 = input_img

    # FIRST LAYER
    Z1 = tfl.Conv2D(filters=8, kernel_size=(4, 4),
                    strides=(1, 1), padding="same")(A0)
    # activation of the first layer
    A1 = tfl.ReLU()(Z1)

    # max pooling in the first layer
    P1 = tfl.MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding='same')(A1)

    # SECOND LAYER
    Z2 = tfl.Conv2D(filters=16, kernel_size=(2, 2),
                    strides=(1, 1), padding="same")(P1)
    # activation of the second layer
    A2 = tfl.ReLU()(Z2)
    # max pooling of the second layer
    P2 = tfl.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same')(A2)

    # Final flattened layer
    F = tfl.Flatten()(P2)

    # output layer
    outputs = tfl.Dense(units=6, activation="softmax")(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


if __name__ == "__main__":
    input_shape = (64, 64, 3)
    model = conv_model(input_shape)
    model.summary()
