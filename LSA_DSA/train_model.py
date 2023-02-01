import argparse
import json
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import model_from_json
from keras.models import load_model
from keras.datasets import mnist, cifar10, fashion_mnist, cifar100
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D,BatchNormalization
from keras.regularizers import l2
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from tensorflow.tools.docs.doc_controls import T
# from tensorflow.keras import layers
# from tensorflow.keras.layers import AveragePooling2D
# from tensorflow.keras.layers import BatchNormalization
from keras import optimizers

CLIP_MIN = -0.5
CLIP_MAX = 0.5
Your_google_drive_path="/content/drive/MyDrive/DeepGDPaper"
def train(args):
    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        layers = [
            # Conv2D(6, (5, 5), padding="same", input_shape=(28, 28, 1)),
            # Activation("relu"),
            # MaxPooling2D(pool_size=(2, 2)),
            # Conv2D(16, (5, 5), padding="same"),
            # Activation("relu"),
            # MaxPooling2D(pool_size=(2, 2)),
            # # Dropout(0.5),
            # Flatten(),
            # Dense(120),
            # Activation("relu"),
            # Dense(84),
            # Activation("relu"),
            # Dense(10),
            Conv2D(4, (5, 5), activation='relu', padding="same", input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(12, (5, 5), activation = 'relu', padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(10, activation= 'softmax')
        ]
    if args.d == "fashion_mnist":
      (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
      x_train = x_train.reshape(-1, 28, 28, 1)
      x_test = x_test.reshape(-1, 28, 28, 1)

      layers = [
          Conv2D(6, (5, 5),activation='relu', padding="same", input_shape=(28, 28, 1)),
          MaxPooling2D(pool_size=(2, 2)),
          Conv2D(16, (5, 5),activation='relu', padding="same"),
          MaxPooling2D(pool_size=(2, 2)),
          Flatten(),
          Dense(84, activation = 'relu'),
          Dense(10, activation = 'softmax'),
      ]
      y_train = np_utils.to_categorical(y_train, 10)
      y_test = np_utils.to_categorical(y_test, 10)

    if args.d=="SVHN":
      
      train_raw = loadmat(str(Your_google_drive_path)+'/Data/train_32x32.mat')
      test_raw = loadmat(str(Your_google_drive_ath)+'/Data/test_32x32.mat')
      x_train = np.array(train_raw['X'])
      x_test = np.array(test_raw['X'])
      y_train = train_raw['y']
      y_test = test_raw['y']
      x_train = np.moveaxis(x_train, -1, 0)
      x_test = np.moveaxis(x_test, -1, 0)
     
      x_test= x_test.reshape (-1,32,32,3)
      x_train= x_train.reshape (-1,32,32,3)
      lb = LabelBinarizer()
      y_train = lb.fit_transform(y_train)
      y_test = lb.fit_transform(y_test)
      
      layers = [
          Conv2D(6, (5, 5),activation='relu', padding="same", input_shape=(32, 32, 3)),
          MaxPooling2D(pool_size=(2, 2)),
          Conv2D(16, (5, 5),activation='relu', padding="same"),
          MaxPooling2D(pool_size=(2, 2)),
          Flatten(),
          Dense(120, activation = 'relu'),
          Dense(84, activation = 'relu'),
          Dense(10, activation = 'softmax'),
      ]
    if args.d == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        layers = [
            Conv2D(64, (3, 3), strides=(1, 1), input_shape=(32, 32, 3), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),

            Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),     
            Conv2D(64, (3, 3), strides=1, padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),

            Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),     
            Conv2D(64, (3, 3), strides=1, padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),

            Conv2D(128, 1, strides=2, padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Conv2D(128, (3, 3), strides=2, padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(128, (3, 3), strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),


            Conv2D(128, (3, 3), strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),     
            Conv2D(128, (3, 3), strides=1, padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),

            Conv2D(256, 1, strides=2, padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Conv2D(256, (3, 3), strides=2, padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(256, (3, 3), strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),

            Conv2D(256, (3, 3), strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),     
            Conv2D(256, (3, 3), strides=1, padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),

            Conv2D(512, 1, strides=2, padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Conv2D(512, (3, 3), strides=2, padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(512, (3, 3), strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),

            Conv2D(512, (3, 3), strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),     
            Conv2D(512, (3, 3), strides=1, padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),

            AveragePooling2D(pool_size=(4, 4), padding='same'),
            Flatten(),
            Dense(10, activation='softmax')
        ]
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)


    elif args.d == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        layers = [
            Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)),
            Activation("relu"),
            Conv2D(32, (3, 3), padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), padding="same"),
            Activation("relu"),
            Conv2D(64, (3, 3), padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), padding="same"),
            Activation("relu"),
            Conv2D(128, (3, 3), padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation("relu"),
            Dropout(0.5),
            Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation("relu"),
            Dropout(0.5),
            Dense(10),
        ]
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)


    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    model = Sequential()
    for layer in layers:
        model.add(layer)
      # model.add(Activation("softmax"))

    # print(model.summary())

    #For all datasets
    # model.compile(
    #     loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
    # )
    #for SVHN dataset
    # opt = tf.keras.optimizers.SGD(lr= 1e-1, momentum=0.9, nesterov=False)
    # model.compile(optimizer=opt,
    #                 loss='categorical_crossentropy',
    #                 metrics=['accuracy'])
    model.compile(
    optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        x_train,
        y_train,

 ##SVHN

        # epochs=100,
        # batch_size=125,
        
        ##fashion_mnist
        # epochs=20,
        # batch_size=258,

        ##Cifar10, MNIST(LeNet5, LeNet4)

        # epochs=150,
        # batch_size=128,
        # ModelN cifar10 12conv 1 == 26%
        # epochs=10,
        # batch_size=250,
        #  ModelN cifar10 12conv 1 == 26%
        # N2
        # epochs=20,
        # batch_size=128,
        # N3
        # epochs=30,
        # batch_size=128,
        # N4
        # epochs=40,
        # batch_size=128,
        # N5
        # epochs=50,
        # batch_size=128,
        # N6
        # epochs=80,
        # batch_size=128, 86%
        # epochs=100,
        # batch_size=128,89 33
        # N8 last compiler 
        # epochs=120,
        # batch_size=128,
        # N9
        epochs=5,
        batch_size=128,




        shuffle=True,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    

    # serialize model to JSON
    #  the keras model which is trained is defined as 'model' in this example
    # model_json = model.to_json()
    model.save("./model/modelN9_{}_.h5".format(args.d))

    # with open("model3_num.json", "w") as json_file:
    #   json.dump(model_json, json_file)

    # serialize weights to HDF5
    # model.save_weights('model_weights.h5')
    # model.save_weights("./model/model3_{}.h5".format(args.d))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str)
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", "fashion_mnist","cifar10", "SVHN"], "Dataset "

    train(args)