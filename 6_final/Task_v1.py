import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

"--------------------------------------------------------------------------------------------"
"""
load the dataset
"""
def load_data():
  (X_train, y_train),(X_test, y_test) = cifar10.load_data()
  X_train  = X_train / 255
  X_test  = X_test / 255
  return (X_train,X_test), (y_train, y_test)

"--------------------------------------------------------------------------------------------"

def create_model():
  """
   Create the model 1
   """
  (X_train, X_test), (y_train, y_test) = load_data()


  model_1 = Sequential([
    layers.Conv2D(32, (3,3),input_shape=(32, 32, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
  ])
  model_1.summary()
  model_1.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
  epochs=10

  "fit the model"
  history = model_1.fit(X_train,y_train,
    validation_data=(X_test,y_test),
    epochs=epochs
  )




  x = tf.constant(X_train[:1])
  y = tf.constant(y_train[:1])
  loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  with tf.GradientTape() as tape:
    tape.watch(x)
    prediction = model_1(x)
    loss = loss_fn(prediction, to_categorical(y, 10))
    # Get the gradients of the loss w.r.t to the input image.
  perturbations = tape.gradient(loss, x)
  # Get the sign of the gradients to create the perturbation
  epsilons = [0, 0.01, 0.1, 0.15]
  descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

  for i, eps in enumerate(epsilons):
    adv_x = x + eps * perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)


create_model()

"""

  ------------------------------------------------------------------------------------------------
  """

"""



  model_2 = Sequential()
  model_2.add(Dense(256, activation='relu', input_dim=3072))
  model_2.add(Dense(256, activation='relu'))
  model_2.add(Dense(10, activation='softmax'))
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

  model_2.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

  history = model_2.fit(X_train, y_train, epochs=15, batch_size=32, verbose=2, validation_split=0.2)

  """"""
  score = model_2.evaluate(X_test, y_test, batch_size=128, verbose=0)
  print(model_2.metrics_names)
  print(score)

 """"""
  ------------------------------------------------------------------------------------------------
  """"""
  Create the data from the third model
  """"""
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()
  y_train = to_categorical(y_train, num_classes=10)
  y_test = to_categorical(y_test, num_classes=10)
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255
  X_test /= 255

  """"""
  Create the  model n3
  """"""

  model_3 = Sequential()

  model_3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
  model_3.add(Conv2D(32, (3, 3), activation='relu'))
  model_3.add(MaxPooling2D(pool_size=(2, 2)))

  model_3.add(Flatten())
  model_3.add(Dense(256, activation='relu'))
  model_3.add(Dense(10, activation='softmax'))

  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model_3.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

  history = model_3.fit(X_train, y_train, batch_size=32, epochs=15, verbose=2, validation_split=0.2)

  score = model_3.evaluate(X_test, y_test, batch_size=128, verbose=0)
  """
