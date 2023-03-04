import os
from google.cloud import storage
from tensorflow.keras.models import load_model
import datetime


def load_cloud_model():
    bucket_name = os.environ["BUCKET"]
    model_name = os.environ["MODEL_NAME"]
    model_dir = os.environ["MODEL_DIR"]

    # Create a client object for Google Cloud Storage
    client = storage.Client()

    # Get a bucket object for the bucket
    bucket = client.get_bucket(bucket_name)

    # Get a blob object for the Keras model file
    blob = bucket.blob(model_name)

    # Download the Keras model file to a local file
    blob.download_to_filename(model_dir + model_name)
    
    model = load_model(model_dir + model_name)
    
    return model


def save_cloud_model(modelh5):
    bucket_name = os.environ["BUCKET"]
    model_dir = os.environ["MODEL_DIR"]
    current_time = datetime.datetime.now()
    save_name = f"model{(current_time)}.h5"
    modelh5.save(model_dir + save_name)

    # Create a client object for Google Cloud Storage
    client = storage.Client()

    # Get a bucket object for the bucket
    bucket = client.get_bucket(bucket_name)

    # Get a blob object for the Keras model file
    blob = bucket.blob(save_name)

    # Download the Keras model file to a local file
    blob.upload_from_filename(model_dir + save_name, timeout = 300)
    
    return "Success!"