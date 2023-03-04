from sign_language.data_sources.load_data import get_images
import os
from sign_language.ml_logic.preprocessing import process_images, split_training_and_test_images, preprocessing
from sign_language.ml_logic.model import initiate_model, compile_model, train_model, evaluate_model
from sign_language.ml_logic.registry import save_model_local, load_model_local
from sign_language.data_sources.bucket_data import upload_processed_images

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

    print('Images processed')

    # create train and test data sets
    split_training_and_test_images(saving_dir)

    # save preprocessed data into bukets in the cloud
    upload_processed_images()

    print('Images saved as train and test sets')

def train():
    '''trains an image on the training dataset'''

    if os.environ.get('LOAD_DATA') == 'local':
        loading_images = os.environ.get("LOCAL_TRAINING_DATA") # path where the images will be saved
        images, labels = get_images(loading_images) # loads the images

        images, labels = preprocessing(images, labels) # preprecoesses the images

    print('Data loaded')

    model = initiate_model()

    print('model initiated')

    model = compile_model(model)

    print('model compiled')

    model, history = train_model(model, images, labels)

    print('model trained')

    if os.environ.get('SAVE_MODEL') == 'local':
        save_model_local(model)

    print('Model trained')

def evaluate():
    '''evaluates the model on the test set'''
    if os.environ.get('LOAD_DATA') == 'local':
        loading_images = os.environ.get("LOCAL_TESTING_DATA") # path where the images will be saved
        images, labels = get_images(loading_images) # loads the images
        images, labels = preprocessing(images, labels) # preprecoesses the images

    print('Data loaded')

    if os.environ.get('LOAD_MODEL') == 'local':
        model = load_model_local()

    print('MOdel loaded')

    accuracy = evaluate_model(model, images, labels)

    print(accuracy)

# ready for us to create the below functions once
# we can save the trained model and load it for evaluation
if __name__ == '__main__':
    preprocess()
    # train()
    # evaluate()
