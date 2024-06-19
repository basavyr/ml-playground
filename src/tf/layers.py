import tensorflow as tf
import tensorflow.keras.layers as tfl

# Create the Sequential model and add only the ZeroPadding2D layer
model = tf.keras.Sequential([
    tfl.ZeroPadding2D(padding=3, input_shape=(64, 64, 3)),
    tfl.Conv2D(filters=32, kernel_size=(7, 7), strides=1, padding="valid"),

])

# Display the model's architecture
model.summary()
