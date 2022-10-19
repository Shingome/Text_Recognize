import keras
import tensorflow_datasets as tfds
import numpy as np
import os
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils.vis_utils import plot_model
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def show_image(image_array):
    image_array.shape = (20, 16)
    image = Image.fromarray(image_array)
    image.show()


def rebuild(x):
    new_x = list()
    for i in x:
        new_x.append(np.reshape(i, (20, 16, 1)))
    return np.asarray(new_x)


if __name__ == "__main__":
    dataset, test = tfds.load('BinaryAlphaDigits', split=["train[:80%]", "train[80%:]"])

    dataset = tfds.as_dataframe(dataset)
    test = tfds.as_dataframe(test)

    x = dataset['image']
    y = dataset['label']
    x = rebuild(x)

    x_test = test['image']
    y_test = test['label']
    x_test = rebuild(x)

    model = keras.Sequential()
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(20, 16, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(36, activation='softmax'))

    plot_model(model,
               to_file="model_plot.png",
               show_dtype=True,
               show_shapes=True,
               show_layer_names=True,
               show_layer_activations=True)

    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    model.fit(x, y, epochs=50, shuffle=True)

    print(model.evaluate(x_test[:len(y_test)], y_test[:len(y_test)]))