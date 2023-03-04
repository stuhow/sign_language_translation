import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers  import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping

def initiate_model():
    model = Sequential()

    model.add(Conv2D(128, (3, 3), padding='same', input_shape = (56, 56, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # #model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # #model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(29, activation='softmax'))

    return model

def compile_model(model):
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):

    es = EarlyStopping(patience = 3, restore_best_weights = True)
    history = model.fit(X_train,
                y_train,
                batch_size = 512,
                epochs=25,
                validation_split = 0.2,
                callbacks = [es],
                verbose=0)

    return model, history

def evaluate_model(model, X_test, y_test):
    accuracy = model.evaluate(X_test, y_test)
    return accuracy[1]
