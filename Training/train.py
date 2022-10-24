import keras
import tensorflow_datasets as tfds
import numpy as np
import os
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, InputLayer
from keras.utils.vis_utils import plot_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def rebuild(x):
    new_x = list()
    for i in x:
        new_x.append(np.reshape(i, (20, 16, 1)))
    return np.asarray(new_x)


if __name__ == "__main__":
    train_dataset, test_dataset = tfds.load('BinaryAlphaDigits', split=["train[:80%]", "train[80%:]"])

    train_dataset = tfds.as_dataframe(train_dataset)
    test_dataset = tfds.as_dataframe(test_dataset)

    x_train = train_dataset['image']
    y_train = train_dataset['label']
    x_train = rebuild(x_train)

    x_test = test_dataset['image']
    y_test = test_dataset['label']
    x_test = rebuild(x_test)

    model = keras.Sequential()
    model.add(InputLayer((20, 16, 1)))
    model.add(Conv2D(32, (2, 2), activation='relu'))
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

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, shuffle=True, batch_size=32)
