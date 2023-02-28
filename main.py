from load_data import get_images
import os
from preprocessing import train_val_test_split, preprocessing, balancing
from model import initiate_model, compile_model, train_model, evaluate_model

# os.environ['DIRECTORY']

print('Start')

directory = os.environ.get('DIRECTORY')

print(directory)

images, labels = get_images(directory)


print('step 1 done')

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(images, labels)

print('step 2 done')

X_train, y_train = balancing(X_train, y_train)
X_train, y_train = preprocessing(X_train, y_train)
X_val, y_val = preprocessing(X_val, y_val)
X_test, y_test = preprocessing(X_test, y_test)

model = initiate_model()

print('step 3 done')

model = compile_model(model)

print('step 4 done')

model, history = train_model(model, X_train, y_train)

print('step 5 done')

accuracy = evaluate_model(model, X_test, y_test)

print(accuracy)
