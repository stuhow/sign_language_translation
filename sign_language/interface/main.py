from data_sources.load_data import get_images, process_images
import os
from ml_logic.preprocessing import train_val_test_split, preprocessing, balancing
from ml_logic.model import initiate_model, compile_model, train_model, evaluate_model
from ml_logic.preprocessing import train_val_test_split, preprocessing

# os.environ['DIRECTORY']

print('Start')
def preprocess():
    '''
    processes the images and then saves train and test data sets
    '''

    # create a folder with all the cleaned images
    directory = os.environ.get('DIRECTORY')
    saving_dir = os.environ.get("SAVE_DIR")
    process_images(directory,saving_dir)


    print('step 1 done')


def train():
    '''trains an image on the training dataset'''

    loading_images = os.environ.get("SAVE_DIR") # path where the images will be saved

    images, labels = get_images(loading_images)

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

def evaluate(model,):
    '''evaluates the model on the test set'''

    accuracy = evaluate_model(model, X_test, y_test)

    print(accuracy)

# ready for us to create the below functions once
# we can save the trained model and load it for evaluation
# if __name__ == '__main__':
#     preprocess()
#     train()
#     pred()
#     evaluate()
