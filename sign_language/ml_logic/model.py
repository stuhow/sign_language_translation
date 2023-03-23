import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers  import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback

def load_transfer_model():
    model = VGG19(include_top=False,
    weights="imagenet",
    input_shape=(56,56,3))
    return model

def set_nontrainable_layers(model):
    model.trainable = True
    return model

def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
    model = set_nontrainable_layers(model)
    flattening_layer = Flatten()
    dense_layer_1 = layers.Dense(500, activation='relu')
    dense_layer_2 = layers.Dense(300, activation='relu')
    dense_layer_3 = layers.Dense(200, activation='relu')
    prediction_layer = layers.Dense(29, activation='softmax')
    model = Sequential([
    model,
    flattening_layer,
    dense_layer_1,
    dense_layer_2,
    dense_layer_3,
    prediction_layer
    ])
    return model

def initiate_model():
    lr_schedule = ExponentialDecay(
    0.001,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)
    model = load_transfer_model()
    model = add_last_layers(model)
    opt = Adam(learning_rate = lr_schedule)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    model = initiate_model()
    es = EarlyStopping(monitor='val_loss', verbose=1, patience = 3, restore_best_weights = True)
    history = model.fit(X_train,
                        y_train,
                        epochs = 100,
                        callbacks = [es, TqdmCallback(verbose = 0)],
                        verbose = 0,
                        batch_size=32)

    return model, history

def evaluate_model(model, X_test, y_test):
    accuracy = model.evaluate(X_test, y_test)
    return accuracy[1]
